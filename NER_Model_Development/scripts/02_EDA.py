#02_EDA

import json
from collections import Counter

with open('E:/Semetser 5/Text Analytics/BioBertNERProject/NER_Model_Development/data/processed/Cleaned_dataset_final.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

entity_counts = Counter()
for doc in data:
    for entity in doc['entities']:
        entity_counts[entity['type']] += 1

print("--- Final Entity Class Distribution ---")
for entity_type, count in entity_counts.most_common():
    print(f"{entity_type}: {count} examples")




import pandas as pd

# Calculate total number of examples (entities)
total_entities = sum(entity_counts.values())

print("\n--- Cost Sensitive Learning Weights ---")
print(f"Total entities: {total_entities}")
print("-" * 35)

# Calculate weights using the user-specified formula
# Weight = Total number of examples / Number of examples for that class

print(f"Formula: Weight = Total entities ({total_entities}) / Number of examples for a specific class")
print("-" * 35)

weights_data = []
weights = {} # Initialize the weights dictionary
for entity_type, count in entity_counts.most_common():
    # Avoid division by zero if a class has no examples (though unlikely with most_common)
    if count > 0:
        weight = total_entities / count
        weights[entity_type] = weight
        weights_data.append({
            "Entity Type": entity_type,
            "Example Count": count,
            "Calculation": f"{total_entities} / {count}",
            "Weight": round(weight, 2)
        })
    else:
        weights[entity_type] = 0.00 # Assign 0 weight to classes with no examples
        weights_data.append({
            "Entity Type": entity_type,
            "Example Count": count,
            "Calculation": "N/A",
            "Weight": 0.00
        })

# Create a pandas DataFrame
weights_df = pd.DataFrame(weights_data)

# Display the DataFrame
print(weights_df)

print("-" * 35)

# You can now use the 'weights' dictionary in your model training
# Example: print(weights)
