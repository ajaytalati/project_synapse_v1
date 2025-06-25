#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 18:06:44 2025

@author: ajay
"""

import torch
import logging
import sys
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers.utils import logging as hf_logging

# ----------------------------------------------------------------------------
# Setup Logging (file only, suppress unwanted stdout spam)
# ----------------------------------------------------------------------------
LOG_FILE = "gemma_finetune.log"

file_handler = logging.FileHandler(LOG_FILE, mode='w')
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
file_handler.setLevel(logging.INFO)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(file_handler)

# Suppress all console logging from HF and tokenizer init
hf_logging.set_verbosity_error()
transformers_logger = logging.getLogger("transformers.tokenization_utils_base")
transformers_logger.setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.CRITICAL)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.CRITICAL)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
base_model_name = "google/gemma-3-4b-it"
dataset_path = "/home/ajay/Python_Projects/project_synapse_v1/test_fine_tuning_Gemma_4b/stoic_emperor_dataset.jsonl"
output_dir = "./gemma4b_qlora_output"
merged_output_dir = "./gemma4b_merged"
hf_token = # copy token here

logger.info("Started fine-tuning Gemma-3-4b-it...")

# ----------------------------------------------------------------------------
# Load and format dataset
# ----------------------------------------------------------------------------
def format_instruction(example):
    return (
        f"<start_of_turn>user\n{example['instruction']}<end_of_turn>\n"
        f"<start_of_turn>model\n{example['output']}<end_of_turn>"
    )

logger.info("Loading dataset...")
dataset = load_dataset("json", data_files=dataset_path, split="train")
dataset = dataset.map(lambda x: {"text": format_instruction(x)})
logger.info(f"Dataset loaded: {len(dataset)} examples")

# ----------------------------------------------------------------------------
# Load tokenizer and model
# ----------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    token=hf_token,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    token=hf_token,
)
model.config.use_cache = False

logger.info("Tokenizer and model loaded")

# ----------------------------------------------------------------------------
# LoRA config
# ----------------------------------------------------------------------------
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, peft_config)
logger.info("LoRA model wrapped")
model.print_trainable_parameters()

# ----------------------------------------------------------------------------
# TrainingArguments and Trainer setup
# ----------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=20,
    learning_rate=5e-5,
    bf16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    optim="paged_adamw_32bit",
    report_to="none",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

def formatting_func(example):
    return example["text"]

data_collator = DataCollatorForCompletionOnlyLM(
    response_template="<start_of_turn>model", tokenizer=tokenizer
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=peft_config,
    formatting_func=formatting_func,
    data_collator=data_collator,
)

# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------
logger.info("Starting training...")
trainer.train()
logger.info("Training completed")

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
logger.info("LoRA model and tokenizer saved")

# ----------------------------------------------------------------------------
# Merge LoRA weights into base model and save with safetensors
# ----------------------------------------------------------------------------
logger.info("Merging and saving final model using safetensors...")
merged_model = model.merge_and_unload()

# Patch to clone lm_head weight and avoid shared tensor error
if hasattr(merged_model, "lm_head") and merged_model.lm_head.weight is merged_model.get_input_embeddings().weight:
    logger.info("Cloning lm_head.weight to avoid safetensors shared memory error...")
    merged_model.lm_head.weight = torch.nn.Parameter(merged_model.lm_head.weight.clone())

merged_model.save_pretrained(
    merged_output_dir,
    safe_serialization=True,
    use_safetensors=True
)


tokenizer.save_pretrained(merged_output_dir)
logger.info("Merged model saved using safetensors format")
