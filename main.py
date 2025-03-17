import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, set_seed

# For reproducibility
set_seed(42)
torch.cuda.empty_cache() 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# ----------------------------
# 1. Data Loading and Splitting
# ----------------------------
dataset = load_dataset("json", data_files="data/processed_quotes.json")["train"]
split_dataset = dataset.train_test_split(test_size=0.01, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# ----------------------------
# 2. Tokenization
# ----------------------------
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

def tokenize_function(examples):
    tokenized_output = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=50)
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()  # Ensure labels are set
    return tokenized_output


train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

# ----------------------------
# 3. Data Collator
# ----------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ----------------------------
# 4. Model Setup and LoRA Fine-Tuning
# ----------------------------
# model = AutoModelForCausalLM.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="./quote_generator",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
)

# ----------------------------
# 7. Metrics Calculation
# ----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    mask = labels != -100
    if mask.sum() == 0:
        accuracy = 0.0
    else:
        accuracy = (predictions[mask] == labels[mask]).astype(np.float32).mean().item()
    return {"accuracy": accuracy}

# ----------------------------
# 8. Trainer Setup and Training
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Fix: Use train_dataset instead of tokenized_dataset
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)


def safe_evaluate(trainer):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Prevent gradient storage
        eval_results = trainer.evaluate()
    model.train()  # Set model back to training mode
    return eval_results

trainer.train()
eval_results = trainer.evaluate()
print("Final evaluation results:", eval_results)

model.save_pretrained("./saved_gpt2")
print("Model saved to ./saved_gpt2")
