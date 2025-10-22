# =============================================================================
# 1. IMPORTS
# All necessary libraries go at the top of the script.
# =============================================================================
import torch
import numpy as np
import evaluate
from datasets import load_dataset # You need this to load your data
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from torch.nn import CrossEntropyLoss
from bitsandbytes.optim import PagedAdamW8bit

# =============================================================================
# 2. DEFINE CUSTOM CLASSES AND FUNCTIONS
# Define all your helper tools before you use them.
# =============================================================================


data_files = {
    "train": "/content/train_bio_final.json",
    "validation": "/content/validation_bio_final.json",
    "test": "/content/test_bio_final.json"
}

class WeightedLossTrainer(Trainer):
    """
    A custom Trainer that uses weighted cross-entropy loss to handle
    class imbalance in Named Entity Recognition tasks.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = class_weights.to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# This function will be used by the .map() method later
def tokenize_and_align_labels(examples, tokenizer, label2id):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=512
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Define the metric and the compute_metrics function
metric = evaluate.load("seqeval")

def compute_metrics(p, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[pred] for (pred, lbl) in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[lbl] for (pred, lbl) in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
    }

# =============================================================================
# 3. MAIN EXECUTION SCRIPT
# This is the main workflow that runs from top to bottom.
# =============================================================================

if __name__ == "__main__":
    # --- A. Data Loading and Preparation ---
    # (Assuming these steps are here from your other scripts)
    # raw_datasets = load_dataset(...)
    # label_list, label2id, id2label = prepare_label_list(...)
    # class_weights_tensor = ...

    # --- B. Model and Tokenizer Loading (with QLoRA) ---
    model_name = "dmis-lab/biobert-base-cased-v1.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )
    peft_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["query", "value"],
        lora_dropout=0.05, bias="none", task_type="TOKEN_CLS"
    )
    qlora_model = get_peft_model(model, peft_config)
    qlora_model.print_trainable_parameters()

    # --- C. Tokenize and Align Datasets ---
    tokenized_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "label2id": label2id}
    )

    # --- D. Define Trainer Configuration ---
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=15,
        learning_rate=2e-5,
        max_grad_norm=1.0,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        bf16=True,
        report_to="none",
    )
    optimizer = PagedAdamW8bit(params=qlora_model.parameters(), lr=2e-5)
    optimizers = (optimizer, None)
    
    # --- E. Initialize and Run the Trainer ---
    trainer = WeightedLossTrainer(
        model=qlora_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, label_list), # Pass label_list correctly
        optimizers=optimizers,
        class_weights=class_weights_tensor
    )
    
    print("--- Starting the fine-tuning process ---")
    trainer.train()
    print("✅ Training complete!")

    # --- F. Save the Final Model ---
    final_model_path = "E:/Semetser 5/Text Analytics/BioBertNERProject/NER_Model_Development/output/final-biobert-ner-model"
    print(f"\n--- Saving the best model to '{final_model_path}' ---")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print("✅ Final model and tokenizer saved successfully.")



