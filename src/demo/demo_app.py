import os
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from peft import PeftModel

# Configure Streamlit page
st.set_page_config(
    page_title="Biomedical NER",
    page_icon="🧬",
    layout="wide"
)

# --- 1. Define the paths and labels ---
base_model_name = "dmis-lab/biobert-base-cased-v1.2"
# Default to current directory if not found, or use a known path if it exists locally
adapter_path = os.environ.get("NER_ADAPTER_PATH", "E:/Semetser 5/final-biobert-ner-model")

label_list = [
    'B-ANATOMY', 'B-CELL_LINE', 'B-CHEMICAL', 'B-DIAGNOSTIC_TEST', 'B-DISEASE', 
    'B-DRUG', 'B-GENE_PROTEIN', 'B-MISC', 'B-QUANTITY', 'B-SYMPTOM', 'B-TREATMENT', 
    'I-ANATOMY', 'I-CELL_LINE', 'I-CHEMICAL', 'I-DIAGNOSTIC_TEST', 'I-DISEASE', 
    'I-DRUG', 'I-GENE_PROTEIN', 'I-MISC', 'I-QUANTITY', 'I-SYMPTOM', 'I-TREATMENT', 
    'O'
]

# Create mappings
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}


# --- 2. Application Logic ---
@st.cache_resource(show_spinner="Loading NLP Models...")
def load_ner_pipeline():
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load base model configured for token classification with custom labels
        base_model = AutoModelForTokenClassification.from_pretrained(
            base_model_name,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id,
        )

        final_model = base_model
        model_status = "Base BioBERT loaded (unfine-tuned)"
        
        # Try loading PEFT adapter weights
        if os.path.exists(adapter_path):
            try:
                model = PeftModel.from_pretrained(base_model, adapter_path)
                final_model = model.merge_and_unload()
                model_status = "Fine-tuned BioBERT loaded successfully!"
            except Exception as e:
                model_status = f"Failed to load adapters: {e}. Falling back to Base model."
        else:
            model_status = f"Adapter path '{adapter_path}' not found. Using Base BioBERT."
            
        # Create pipeline
        ner_pipe = pipeline(
            "ner", 
            model=final_model, 
            tokenizer=tokenizer, 
            aggregation_strategy="simple"
        )
        
        return ner_pipe, model_status
        
    except Exception as e:
        return None, f"Critical model load failure: {e}"

# Load the pipeline
ner_pipeline, pipeline_status = load_ner_pipeline()

# --- 3. UI Layout ---
st.title("🧬 Biomedical NER Extractor")
st.markdown("Extract medical entities such as **Diseases, Drugs, Symptoms, and Treatments** from clinical abstracts.")

# Sidebar status
with st.sidebar:
    st.header("System Status")
    if ner_pipeline:
        if "successfully" in pipeline_status:
            st.success(pipeline_status)
        else:
            st.warning(pipeline_status)
            st.info("Note: Without the fine-tuned adapter weights, the underlying BioBERT model might predominantly output 'O' (Outside) tokens or incorrect labels for specific domain terms.")
    else:
        st.error(pipeline_status)
        
    st.markdown("---")
    st.markdown("**Supported Entities:**")
    st.markdown(f"*{', '.join([l.replace('B-', '') for l in label_list if l.startswith('B-')])}*")

# Main Interface
user_text = st.text_area(
    "Enter a biomedical abstract or clinical sentence:", 
    height=150,
    value="The patient, a 54-year-old male, was diagnosed with type 2 diabetes mellitus and hypertension. He was prescribed metformin 500mg twice daily and lisinopril 10mg."
)

if st.button("Extract Entities", type="primary", use_container_width=True):
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
    elif ner_pipeline is None:
        st.error("The NER pipeline failed to initialize.")
    else:
        with st.spinner("Analyzing text..."):
            entities = ner_pipeline(user_text)
            
            st.subheader("Extraction Results")
            if not entities:
                st.info("No entities were extracted from the provided text.")
            else:
                # Format output into a nice dataframe
                df_data = []
                for ent in entities:
                    df_data.append({
                        "Entity Group": ent.get("entity_group", ent.get("entity", "Unknown")),
                        "Word": ent.get("word", ""),
                        "Confidence": f"{ent.get('score', 0.0):.1%}",
                        "Start": ent.get("start", 0),
                        "End": ent.get("end", 0)
                    })
                
                df = pd.DataFrame(df_data)
                
                # Display metrics
                col1, col2 = st.columns(2)
                col1.metric("Total Entities", len(df))
                col2.metric("Unique Entity Types", df["Entity Group"].nunique())
                
                # Show results table
                st.dataframe(
                    df[["Entity Group", "Word", "Confidence"]], 
                    use_container_width=True,
                    hide_index=True
                )
                
                # Show raw JSON expansible block
                with st.expander("View Raw JSON Output"):
                    st.json(entities)
