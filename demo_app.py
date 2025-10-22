from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel

# --- 1. Define the paths ---
# The original, full-sized model you started with
base_model_name = "dmis-lab/biobert-base-cased-v1.2"
# The path to the folder where you saved your LoRA adapters
adapter_path = "E:/Semetser 5/final-biobert-ner-model"

label_list = [
    'B-ANATOMY', 'B-CELL_LINE', 'B-CHEMICAL', 'B-DIAGNOSTIC_TEST', 'B-DISEASE', 
    'B-DRUG', 'B-GENE_PROTEIN', 'B-MISC', 'B-QUANTITY', 'B-SYMPTOM', 'B-TREATMENT', 
    'I-ANATOMY', 'I-CELL_LINE', 'I-CHEMICAL', 'I-DIAGNOSTIC_TEST', 'I-DISEASE', 
    'I-DRUG', 'I-GENE_PROTEIN', 'I-MISC', 'I-QUANTITY', 'I-SYMPTOM', 'I-TREATMENT', 
    'O'
]

# --- 2. Load the Base Model and Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# You must configure the base model with YOUR custom labels first
# (Assume label_list, id2label, and label2id are already defined)
base_model = AutoModelForTokenClassification.from_pretrained(
    base_model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
    device_map="auto"
)

# --- 3. Load and Merge the LoRA Adapters ---
# This command loads your saved adapter weights and merges them into the base model
model = PeftModel.from_pretrained(base_model, adapter_path)
# For faster inference, you can fully merge the adapters
model = model.merge_and_unload()

# --- 4. Run Inference ---
# You can now use the 'model' object in a Hugging Face pipeline
# ... (add all the code to load your final, merged model and tokenizer) ...

st.title("Biomedical NER Extractor")

# Load your fine-tuned NER pipeline (this should only run once)
@st.cache_resource
def load_ner_pipeline():
    # ... (your model loading and merging code here) ...
    return pipeline("ner", model=merged_model, tokenizer=tokenizer, aggregation_strategy="simple")

ner_pipeline = load_ner_pipeline()

# Create a text area for user input
user_text = st.text_area("Enter a biomedical abstract or sentence:", height=150)

if st.button("Extract Entities"):
    if user_text:
        entities = ner_pipeline(user_text)
        st.subheader("Extracted Entities:")
        st.write(entities)
    else:
        st.warning("Please enter some text.")
