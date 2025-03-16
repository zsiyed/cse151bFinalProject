import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
                          Trainer, DataCollatorForLanguageModeling, set_seed, TrainerCallback)
from peft import LoraConfig, get_peft_model

# For reproducibility
set_seed(42)

# ----------------------------
# 1. Data Loading and Splitting
# ----------------------------
dataset = load_dataset("json", data_files="data/processed_quotes.json")["train"]
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# ----------------------------
# 2. Tokenization
# ----------------------------
model_name = "BEE-spoke-data/smol_llama-101M-GQA"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    if "quote" in examples:
        key = "quote"
    elif "text" in examples:
        key = "text"
    else:
        key = list(examples.keys())[0]
    prompts = [
        "Generate a personalized motivational quote for [user_input].\n" + quote
        for quote in examples[key]
    ]
    return tokenizer(prompts, truncation=True, max_length=512, padding="max_length")

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

# ----------------------------
# 3. Data Collator
# ----------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ----------------------------
# 4. Model Setup and LoRA Fine-Tuning
# ----------------------------
model = AutoModelForCausalLM.from_pretrained(model_name)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ----------------------------
# 5. Custom Callback for Gradient Logging
# ----------------------------
class GradientLogger(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % args.logging_steps == 0 and model is not None:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Step {state.global_step}: Gradient Norm = {total_norm:.4f}")

# ----------------------------
# 6. Training Setup with Modified Hyperparameters
# ----------------------------
training_args = TrainingArguments(
    output_dir="./lora_smol_llama_finetuned",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    num_train_epochs=5,           # Increased number of epochs
    learning_rate=1e-3,           # Increased learning rate
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    fp16=True if torch.cuda.is_available() else False,
    evaluation_strategy="steps",
    eval_steps=50,
    eval_accumulation_steps=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
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
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[GradientLogger]
)

trainer.train()
eval_results = trainer.evaluate()
print("Final evaluation results:", eval_results)

model.save_pretrained("./lora_smol_llama_finetuned")
print("Model saved to ./lora_smol_llama_finetuned")
