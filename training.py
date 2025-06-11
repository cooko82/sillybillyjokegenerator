# === Patch DTensor issue BEFORE importing transformers ===
import torch
import types

# Prevent crash from missing DTensor
if not hasattr(torch.distributed, "tensor"):
    torch.distributed.tensor = types.SimpleNamespace()
if not hasattr(torch.distributed.tensor, "DTensor"):
    torch.distributed.tensor.DTensor = type("FakeDTensor", (), {})()

# Patch pytorch_utils.id_tensor_storage to avoid DTensor crash
import transformers.pytorch_utils as pytorch_utils
_original_id_tensor_storage = pytorch_utils.id_tensor_storage

def safe_id_tensor_storage(tensor):
    try:
        return _original_id_tensor_storage(tensor)
    except ImportError:
        return id(tensor.storage()), 0

pytorch_utils.id_tensor_storage = safe_id_tensor_storage

# === Now import everything else ===
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import os

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Load dataset
reddit_path = "/Users/lauraleone/Documents/Github_Repos/sillybillyjokegenerator/data/reddit_jokes.txt"
dataset = load_dataset('text', data_files=reddit_path)

# Check how many jokes are loaded
with open(reddit_path) as f:
    lines = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(lines)} jokes")

# Tokenize dataset
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training configuration
training_args = TrainingArguments(
    output_dir="./sillybilly-model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_strategy="no",
    logging_dir='./logs',
    logging_steps=100,
)

os.makedirs(training_args.output_dir, exist_ok=True)

# Set up Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

# Optional: disable Trainer save
trainer.save_model = lambda *args, **kwargs: None

# Train!
trainer.train()

# Save manually
model.save_pretrained("final_model")
tokenizer.save_pretrained("final_model")
