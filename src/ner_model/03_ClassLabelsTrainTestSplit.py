import json
from collections import Counter
from sklearn.model_selection import train_test_split

with open('E:/Semetser 5/Text Analytics/BioBertNERProject/NER_Model_Development/data/processed/Cleaned_dataset_final.json', 'r', encoding='utf-8') as f:
    documents = json.load(f)

# Create a representative label for each document for stratification
labels = []
for doc in documents:
    if doc['entities']:
        doc_labels = [entity['type'] for entity in doc['entities']]
        labels.append(Counter(doc_labels).most_common(1)[0][0])
    else:
        labels.append('NO_ENTITY')

# First split: 80% train, 20% temp
train_docs_final, temp_docs, _, _ = train_test_split( 
    documents, labels, test_size=0.2, random_state=42, stratify=labels)

# Re-stratify the temp set for the final split
temp_labels = []
for doc in temp_docs:
    if doc['entities']:
        doc_labels = [entity['type'] for entity in doc['entities']]
        temp_labels.append(Counter(doc_labels).most_common(1)[0][0])
    else:
        temp_labels.append('NO_ENTITY')

val_docs_final, test_docs_final, _, _ = train_test_split(
    temp_docs, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

# Save the splits to new files
with open('E:/Semetser 5/Text Analytics/BioBertNERProject/NER_Model_Development/data/processed/train_dataset_final.json', 'w') as f: json.dump(train_docs_final, f)
with open('E:/Semetser 5/Text Analytics/BioBertNERProject/NER_Model_Development/data/processed/validation_dataset_final.json', 'w') as f: json.dump(val_docs_final, f)
with open('E:/Semetser 5/Text Analytics/BioBertNERProject/NER_Model_Development/data/processed/test_dataset_final.json', 'w') as f: json.dump(test_docs_final, f)

print(f"\nDataset split complete:")
print(f"Training set size: {len(train_docs_final)}")
print(f"Validation set size: {len(val_docs_final)}")
print(f"Test set size: {len(test_docs_final)}")




import json
import spacy

nlp = spacy.blank("en")

def convert_to_bio_format(documents):
    bio_data = []
    for doc in documents:
        spacy_doc = nlp(doc['text'])
        tokens = [token.text for token in spacy_doc]
        tags = ['O'] * len(tokens)

        for entity in doc['entities']:
            span = spacy_doc.char_span(entity['start'], entity['end'], label=entity['type'])
            if span:
                tags[span.start] = f"B-{entity['type']}"
                for i in range(span.start + 1, span.end):
                    tags[i] = f"I-{entity['type']}"

        bio_data.append({"tokens": tokens, "ner_tags": tags})
    return bio_data

# Convert all three splits
train_bio = convert_to_bio_format(train_docs_final)
val_bio = convert_to_bio_format(val_docs_final)
test_bio = convert_to_bio_format(test_docs_final)

# Save the final, model-ready data
with open('E:/Semetser 5/Text Analytics/BioBertNERProject/NER_Model_Development/data/processed/train_bio_final.json', 'w') as f: json.dump(train_bio, f)
with open('E:/Semetser 5/Text Analytics/BioBertNERProject/NER_Model_Development/data/processed/validation_bio_final.json', 'w') as f: json.dump(val_bio, f)
with open('E:/Semetser 5/Text Analytics/BioBertNERProject/NER_Model_Development/data/processed/test_bio_final.json', 'w') as f: json.dump(test_bio, f)

print("\nBIO conversion complete. Data is ready for model training.")


from datasets import load_dataset

# 1. Define the paths to your final, BIO-formatted JSON files
data_files = {
    "train": "E:/Semetser 5/Text Analytics/BioBertNERProject/NER_Model_Development/data/processed/train_bio_final.json",
    "validation": "E:/Semetser 5/Text Analytics/BioBertNERProject/NER_Model_Development/data/processed/validation_bio_final.json",
    "test": "E:/Semetser 5/Text Analytics/BioBertNERProject/NER_Model_Development/data/processed/test_bio_final.json"
}

# 2. Load the data from the files into a single DatasetDict object
# The 'json' argument tells the library how to read the files.
raw_datasets = load_dataset("json", data_files=data_files)

# 3. (Optional) Print the loaded object to see its structure
print("--- Datasets loaded successfully! ---")
print(raw_datasets)

# You can also inspect a single example from the training set
print("\n--- First example from the training set ---")
print(raw_datasets["train"][0])


import json

def prepare_label_list(bio_json_file):
    """
    Reads a BIO-formatted JSON file and creates a sorted list of unique
    NER tags and the corresponding id/label mappings.

    Args:
        bio_json_file (str): Path to the BIO-formatted training data file.

    Returns:
        tuple: A tuple containing (label_list, label_to_id, id_to_label).
    """

    # Load the BIO-formatted training data
    with open(bio_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Use a set to automatically handle duplicates and find all unique tags
    unique_labels = set()
    for item in data:
        for tag in item['ner_tags']:
            unique_labels.add(tag)

    # Convert the set to a sorted list for consistent ordering
    label_list = sorted(list(unique_labels))

    # Create the required mappings for the Hugging Face model
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for i, label in enumerate(label_list)}

    return label_list, label_to_id, id_to_label

# --- Main Execution ---
if __name__ == "__main__":
    training_file = 'E:/Semetser 5/Text Analytics/BioBertNERProject/NER_Model_Development/data/processed/train_bio_final.json'

    # Generate the label list and mappings
    LabelList, label2id, id2label = prepare_label_list(training_file)

    print("--- Complete List of Unique BIO Tags ---")
    print(LabelList)


    print(f"\nTotal number of unique labels: {len(LabelList)}")

    print("\n--- Mapping from Label to ID (label2id) ---")
    print(label2id)
    print("\n--- Mapping from ID TO label (id2label) ---")
    print(id2label)


