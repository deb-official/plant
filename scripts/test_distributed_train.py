from accelerate import Accelerator
from accelerate.state import AcceleratorState
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import torch

# Ensure clean AcceleratorState
if AcceleratorState._shared_state:
    AcceleratorState._reset_state()

# Initialize Accelerator
accelerator = Accelerator(mixed_precision="fp16")

# Verify distributed setup
if accelerator.is_main_process:
    print(f"Distributed type: {accelerator.distributed_type}")

# Load dataset (IMDb)
dataset = load_dataset("imdb")
train_dataset = dataset["train"].select(range(1000))  # Subset for speed
eval_dataset = dataset["test"].select(range(200))    # Subset for speed

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    quantization_config={"load_in_4bit": True},  # 4-bit quantization
    trust_remote_code=True
)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Define TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    learning_rate=2e-5,
    fp16=True,
)

# Prepare model and datasets
model, train_dataset, eval_dataset = accelerator.prepare(model, train_dataset, eval_dataset)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train
trainer.train()

# Clean up
accelerator.end_training()