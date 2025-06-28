import os

# Set working directory
wd_path = "/home/ajay/Python_Projects/project_synapse_v1/mistral_ft_experiments"
try:
    os.chdir(wd_path)
    print(f"Working directory set to: {wd_path}")
except FileNotFoundError:
    print(f"ERROR: Directory '{wd_path}' not found. Create it first.")
    exit()

#==========================
import logging

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

#==========================

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

# Configuration
MODEL_ID = "microsoft/phi-2"
DATA_PATH = "data/pubmedqa_mistral_format.json"
OUTPUT_DIR = "phi2_medical_ft"
BATCH_SIZE = 4

# Load dataset
dataset = load_dataset("json", data_files=DATA_PATH)

# Create validation split if needed
if "validation" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.1)
    dataset["validation"] = dataset.pop("test")
    
print(f"Loaded {len(dataset['train'])} train, {len(dataset['validation'])} eval examples")

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    padding_side="right",
    use_fast=True,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# 4-bit config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Prepare for training
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

# CORRECTED LoRA config based on ACTUAL architecture
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "dense"],  # Verified in your model
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,
    learning_rate=3e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    fp16=True,
    gradient_checkpointing=True,
    report_to="none",
    max_grad_norm=0.3,
    warmup_ratio=0.03
)

# Formatting function
def format_chat(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    formatting_func=format_chat,
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args
)

# Start training
print("Starting training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR}")