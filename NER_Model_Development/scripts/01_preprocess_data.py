import json
from collections import defaultdict
import spacy
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import numpy as np

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
#  STEP 1: DEFINE MAPPING and PRECEDENCE HIERARCHY
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# This dictionary is the rulebook that converts raw UMLS Type IDs
# into your desired human-readable categories.
TYPE_MAPPING = {
    # DISEASE
    "T047": "DISEASE", "T191": "DISEASE", "T046": "DISEASE",
    "T020": "DISEASE", "T019": "DISEASE", "T048": "DISEASE",
    # SYMPTOM
    "T184": "SYMPTOM",
    # GENE_PROTEIN
    "T028": "GENE_PROTEIN", "T116": "GENE_PROTEIN", "T085": "GENE_PROTEIN",
    "T086": "GENE_PROTEIN", "T087": "GENE_PROTEIN",
    # DRUG
    "T121": "DRUG", "T127": "DRUG", "T122": "DRUG", "T131": "DRUG",
    # CHEMICAL
    "T109": "CHEMICAL", "T120": "CHEMICAL", "T103": "CHEMICAL",
    "T104": "CHEMICAL", "T197": "CHEMICAL",
    # ANATOMY
    "T023": "ANATOMY", "T022": "ANATOMY", "T024": "ANATOMY",
    "T030": "ANATOMY", "T029": "ANATOMY",
    # CELL_LINE
    "T025": "CELL_LINE", "T026": "CELL_LINE",
    # DIAGNOSTIC_TEST
    "T060": "DIAGNOSTIC_TEST", "T059": "DIAGNOSTIC_TEST", "T034": "DIAGNOSTIC_TEST",
    # TREATMENT
    "T061": "TREATMENT",
    # QUANTITY
    "T081": "QUANTITY", "T080": "QUANTITY", "T082": "QUANTITY"
}

# This list defines the tie-breaking rule (highest priority first).
PRECEDENCE_HIERARCHY = [
    "GENE_PROTEIN", "DRUG", "DISEASE", "SYMPTOM", "ANATOMY",
    "CELL_LINE", "DIAGNOSTIC_TEST", "TREATMENT", "CHEMICAL", "QUANTITY", "MISC"
]



def normalize_text(text):
    return text.lower().strip()


def parse_and_map_pubtator(file_path):
    """Parses the PubTator file and applies the initial type mapping."""
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        blocks = f.read().strip().split('\n\n')

        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 2: continue

            title_line, abstract_line = lines[0], lines[1]
            pmid, _, title = title_line.partition('|t|')
            _, _, abstract = abstract_line.partition('|a|')
            full_text = title + " " + abstract

            entities = []
            for line in lines[2:]:
                parts = line.split('\t')
                if len(parts) == 6:
                    try:
                        start, end = int(parts[1]), int(parts[2])
                        entity_text, original_types, concept_id,  = parts[3], parts[4], parts[5]

                        # Use the first type for mapping if multiple are present
                        first_type = original_types.split(',')[0]

                        # Apply mapping: If type not in map, default to 'MISC'
                        mapped_type = TYPE_MAPPING.get(first_type, "MISC")

                        entities.append({
                            "start": start, "end": end,
                            "text": entity_text, "type": mapped_type,
                            "concept_id": concept_id
                        })
                    except ValueError:
                        continue

            documents.append({"pmid": pmid, "text": full_text, "entities": entities})
    print("✅ Step 1: Parsing  initial mapping and concept id extraction complete.")
    return documents



def resolve_ambiguity(documents, precedence_list):
    """Resolves entity type ambiguity using the precedence hierarchy."""
    precedence_map = {name: i for i, name in enumerate(precedence_list)}

    text_to_labels = defaultdict(set)
    for doc in documents:
        for entity in doc['entities']:
            text_to_labels[entity['text']].add(entity['type'])

    resolution_map = {}
    for text, labels in text_to_labels.items():
        if len(labels) > 1:
            best_label = min(labels, key=lambda label: precedence_map.get(label, 999))
            resolution_map[text] = best_label

    cleaned_documents_temp1 = []
    for doc in documents:
        new_entities = []
        added_entities_in_doc = set()
        for entity in doc['entities']:
            entity_key = (entity['start'], entity['end'], entity['text'])
            if entity['text'] in resolution_map:
                resolved_type = resolution_map[entity['text']]
                if entity['type'] == resolved_type and entity_key not in added_entities_in_doc:
                    new_entity = entity.copy()
                    new_entity['type'] = resolved_type
                    new_entities.append(new_entity)
                    added_entities_in_doc.add(entity_key)
            elif entity_key not in added_entities_in_doc:
                new_entities.append(entity)
                added_entities_in_doc.add(entity_key)

        doc['entities'] = new_entities
        cleaned_documents_temp1.append(doc)

    print(f"✅ Step 2: Ambiguity resolution complete. {len(resolution_map)} conflicts resolved.")
    return cleaned_documents_temp1



def enforce_concept_id_consistency(documents):
    """Ensures that all entities with the same Concept ID have the same final entity type."""
    concept_id_to_types = defaultdict(list)
    for doc in documents:
        for entity in doc['entities']:
          if entity['concept_id'] != "-1":
            concept_id_to_types[entity['concept_id']].append(entity['type'])

    concept_id_resolution = {}
    for concept_id, types in concept_id_to_types.items():
        if len(set(types)) > 1:
            # Find the most common type for this concept ID
            majority_type = Counter(types).most_common(1)[0][0]
            concept_id_resolution[concept_id] = majority_type

    for doc in documents:
        for entity in doc['entities']:
            if entity['concept_id'] in concept_id_resolution:
                entity['type'] = concept_id_resolution[entity['concept_id']]

    print(f"✅ Step 3: Concept ID consistency enforced. {len(concept_id_resolution)} conflicts resolved.")
    return documents

def resolve_overlapping_entities(documents):
    """Resolves overlapping entities using the 'longest entity wins' rule."""
    for doc in documents:
        entities = sorted(doc['entities'], key=lambda x: x['start'])
        if not entities: continue

        non_overlapping_entities = [entities[0]]
        for i in range(1, len(entities)):
            current_entity = entities[i]
            last_entity = non_overlapping_entities[-1]

            # Check for overlap
            if current_entity['start'] < last_entity['end']:
                # Overlap detected, apply the 'longest entity wins' rule
                if (current_entity['end'] - current_entity['start']) > (last_entity['end'] - last_entity['start']):
                    # Current entity is longer, so it replaces the last one
                    non_overlapping_entities[-1] = current_entity
                # Otherwise, the current (shorter) entity is simply skipped
            else:
                # No overlap
                non_overlapping_entities.append(current_entity)
        doc['entities'] = non_overlapping_entities
    print("✅ Step 4: Overlapping entities resolved.")
    return documents

def enforce_consistency_and_filter_noise(documents, nlp_model):
    """
    Refines entity boundaries using linguistic rules and filters out noisy labels.
    """
    consistent_docs = []
    POS_TO_TRIM = {"ADP", "DET", "CCONJ"} # Prepositions, Articles, Conjunctions
    STOP_LIST_ENTITIES = {"the", "a", "an", "of", "in", "is", "was", "and", "or", "to", "for", "study", "results", "analysis"}

    for doc in documents:
        full_text_doc = nlp_model(doc['text'])
        new_entities = []
        for entity in doc['entities']:
            original_label = entity['type']
            try:
                span = full_text_doc.char_span(entity['start'], entity['end'])
                if span is not None:
                    # --- Rule 1: Refine Boundaries ---
                    while len(span) > 1 and span[0].pos_ in POS_TO_TRIM:
                        span = span[1:]

                    # --- Rule 2: Filter by Length ---
                    if len(span.text) <= 2:
                        continue # Skip this entity if it's too short

                    # --- Rule 3: Filter by Stop List ---
                    if span.text.lower() in STOP_LIST_ENTITIES:
                        continue # Skip this entity if it's a common stop word

                    new_entities.append({
                        "text": span.text,
                        "type": original_label,
                        "start": span.start_char,
                        "end": span.end_char
                    })
            except Exception:
                continue
        doc['entities'] = new_entities
        consistent_docs.append(doc)
    print("✅ Step 5: Boundary consistency and noise filtering complete.")
    return consistent_docs

import os
import json
import spacy

# --- (All of your function definitions go here) ---

if __name__ == "__main__":
    # --- Define Paths ---
    input_file = 'E:/Semetser 5/Text Analytics/BioBertNERProject/NER_Model_Development/data/raw/corpus_pubtator1.txt'
    output_file = 'E:/Semetser 5/Text Analytics/BioBertNERProject/NER_Model_Development/data/processed/Cleaned_dataset_final.json'

    # --- Run the full, enhanced preprocessing pipeline ---
    docs = parse_and_map_pubtator(input_file)
    docs = resolve_ambiguity(docs, PRECEDENCE_HIERARCHY)
    docs = enforce_concept_id_consistency(docs)
    
    # This was the intermediate result before the final cleaning step
    intermediate_docs = resolve_overlapping_entities(docs)

    # --- This is the corrected section ---
    
    # 1. Load the spaCy model
    print("Loading spaCy model for final cleaning steps...")
    nlp = spacy.load("en_core_web_sm")

    # 2. Call the function correctly (quotes removed)
    final_docs = enforce_consistency_and_filter_noise(intermediate_docs, nlp)

    # 3. Create the output directory before saving the file
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # 4. Save the final, fully cleaned dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_docs, f, indent=2)

    print(f"\n🎉 Success! Enhanced, clean dataset saved to '{output_file}'")    



