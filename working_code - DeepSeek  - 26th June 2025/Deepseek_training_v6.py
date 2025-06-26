#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 08:27:52 2025

@author: ajay
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 05:25:53 2025
@author: ajay
"""

import torch
import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DataCollatorForCompletionOnlyLM
from transformers.utils import logging as hf_logging
import os
import gc

# ----------------------------------------------------------------------------
# Setup Logging
# ----------------------------------------------------------------------------
LOG_FILE = "gemma_finetune.log"

file_handler = logging.FileHandler(LOG_FILE, mode='w')
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
file_handler.setLevel(logging.INFO)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(file_handler)

# Suppress noisy logs
hf_logging.set_verbosity_error()
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.CRITICAL)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
base_model_name = "google/gemma-3-4b-it"
dataset_path = "/home/ajay/Python_Projects/project_synapse_v1/test_fine_tuning_Gemma_4b/stoic_emperor_gemma_ready_cleaned.jsonl"
output_dir = "./gemma4b_qlora_output"
merged_output_dir = "./gemma4b_merged"
hf_token = # add

logger.info("Started fine-tuning Gemma-3-4b-it...")

# Set working directory
wd_path = "/home/ajay/Python_Projects/project_synapse_v1/test_fine_tuning_Gemma_4b"
os.chdir(wd_path)

# ----------------------------------------------------------------------------
# Load and clean dataset
# ----------------------------------------------------------------------------
logger.info("Loading and cleaning dataset...")
dataset = load_dataset("json", data_files=dataset_path, split="train")

def clean_dataset(example):
    """Fix formatting and ensure proper Gemma structure"""
    text = example["text"]
    
    # Ensure BOS token at start
    if not text.startswith("<bos>"):
        text = "<bos>" + text
        
    # Fix question prefixes
    if "Tell me about is" in text:
        text = text.replace("Tell me about is", "What is")
    if "Tell me about" in text:
        text = text.replace("Tell me about", "What is")
    if "What if one were to should" in text:
        text = text.replace("What if one were to should", "How should")
    if "What if one were to do" in text:
        text = text.replace("What if one were to do", "How should")
    if "Can a wise person answer:" in text:
        text = text.replace("Can a wise person answer:", "")
    
    # Ensure consistent structure
    parts = text.split("<end_of_turn>")
    if len(parts) < 2:
        text += "<end_of_turn>"
    elif len(parts) > 2:
        text = "<end_of_turn>".join(parts[:2]) + "<end_of_turn>"
    
    example["text"] = text
    return example

dataset = dataset.map(clean_dataset)

# Validate dataset structure
def validate_dataset(dataset):
    for i, ex in enumerate(dataset):
        text = ex["text"]
        if not text.startswith("<bos>"):
            raise ValueError(f"Example {i} missing BOS token: {text[:50]}...")
        if text.count("<start_of_turn>user") != text.count("<start_of_turn>model"):
            raise ValueError(f"Example {i} has unbalanced turn tokens")
        if text.count("<end_of_turn>") % 2 != 0:
            raise ValueError(f"Example {i} has unpaired EOT token")

try:
    validate_dataset(dataset)
    logger.info("Dataset validation passed")
except ValueError as e:
    logger.error(f"Dataset validation failed: {str(e)}")
    raise

logger.info(f"Cleaned dataset loaded: {len(dataset)} examples")

# ----------------------------------------------------------------------------
# Tokenizer Setup
# ----------------------------------------------------------------------------
logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    token=hf_token,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# ----------------------------------------------------------------------------
# Tokenize Dataset
# ----------------------------------------------------------------------------
logger.info("Tokenizing dataset...")
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding=False,
        add_special_tokens=False
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=8,
    remove_columns=["text"],
    load_from_cache_file=False
)

# Log sample tokenization
sample_text = tokenizer.decode(tokenized_dataset[0]['input_ids'])
logger.info(f"Sample tokenized text: {sample_text}")

logger.info("Dataset tokenized")

# Clear memory
del dataset
gc.collect()

# ----------------------------------------------------------------------------
# Load Model
# ----------------------------------------------------------------------------
logger.info("Loading model...")
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

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

logger.info("Model loaded")

# ----------------------------------------------------------------------------
# LoRA Configuration
# ----------------------------------------------------------------------------
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "o_proj"],
    inference_mode=False
)

model = get_peft_model(model, peft_config)
logger.info("LoRA model configured")

# Verify trainable parameters
trainable_params = 0
all_params = 0
for name, param in model.named_parameters():
    all_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()

logger.info(f"Trainable params: {trainable_params} || All params: {all_params} || Trainable%: {100 * trainable_params / all_params}")

# ----------------------------------------------------------------------------
# Data Collator
# ----------------------------------------------------------------------------
response_template_str = "<start_of_turn>model"
response_template = tokenizer.encode(
    response_template_str, 
    add_special_tokens=False
)

if response_template[0] == tokenizer.bos_token_id:
    response_template = response_template[1:]

data_collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer
)

logger.info(f"Response template: {response_template} ({tokenizer.decode(response_template)})")

# ----------------------------------------------------------------------------
# Training Configuration
# ----------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    learning_rate=1e-4,
    bf16=True,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=1,
    optim="adafactor",
    report_to="none",
    gradient_checkpointing=True,
    max_grad_norm=0.5,
    lr_scheduler_type="constant",
    warmup_ratio=0.05,
    remove_unused_columns=False,
    logging_dir="./logs",
    ddp_find_unused_parameters=False,
)

# ----------------------------------------------------------------------------
# Trainer Setup
# ----------------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
logger.info("Starting training...")
try:
    model.train()
    train_result = trainer.train()
    logger.info("Training completed")
    
    metrics = train_result.metrics
    metrics["train_samples"] = len(tokenized_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("LoRA model and tokenizer saved")
    
    # ----------------------------------------------------------------------------
    # Merge and Save Final Model
    # ----------------------------------------------------------------------------
    logger.info("Merging and saving final model...")
    merged_model = model.merge_and_unload()
    
    # Vocabulary size handling
    logger.info(f"Original vocab size: {len(tokenizer)}")
    logger.info(f"Model vocab size before resize: {merged_model.get_input_embeddings().num_embeddings}")
    
    if merged_model.get_input_embeddings().num_embeddings != len(tokenizer):
        logger.info(f"Resizing token embeddings to match tokenizer ({len(tokenizer)})")
        merged_model.resize_token_embeddings(len(tokenizer))
        merged_model.config.vocab_size = len(tokenizer)

    # Break shared weights
    if hasattr(merged_model, "lm_head") and hasattr(merged_model, "model"):
        if hasattr(merged_model.model, "embed_tokens"):
            if merged_model.lm_head.weight.data_ptr() == merged_model.model.embed_tokens.weight.data_ptr():
                logger.info("Breaking shared weights by cloning lm_head weights")
                merged_model.lm_head.weight = torch.nn.Parameter(
                    merged_model.lm_head.weight.clone().detach()
                )
                
                if merged_model.lm_head.weight.data_ptr() == merged_model.model.embed_tokens.weight.data_ptr():
                    logger.error("Failed to break shared weights connection!")
                else:
                    logger.info("Successfully broke shared weights connection")

    # Save model
    try:
        merged_model.save_pretrained(merged_output_dir, safe_serialization=True)
        logger.info("Merged model saved with safetensors")
    except RuntimeError as e:
        if "shared memory" in str(e) or "duplicate memory" in str(e):
            logger.warning("Safetensors saving failed, falling back to PyTorch format")
            merged_model.save_pretrained(merged_output_dir, safe_serialization=False)
            logger.info("Merged model saved with PyTorch format")
        else:
            raise
    
    tokenizer.save_pretrained(merged_output_dir)
    logger.info("Tokenizer saved")
    
    # ----------------------------------------------------------------------------
    # Verification Test (Updated)
    # ----------------------------------------------------------------------------
    logger.info("Running verification test...")
    try:
        sample_dataset = load_dataset("json", data_files=dataset_path, split="train[:1]")
        test_sample = sample_dataset[0]["text"]
        inputs = tokenizer(test_sample, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        
        # Move model to GPU without changing dtype
        merged_model = merged_model.to("cuda")
        
        with torch.no_grad():
            outputs = merged_model.generate(
                **inputs.to("cuda"), 
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=False)
            logger.info(f"Training sample replication test:\n{response}")
    except Exception as e:
        logger.error(f"Verification test failed: {str(e)}")

except Exception as e:
    logger.error(f"Training failed: {str(e)}")
    raise

logger.info("Script completed")
