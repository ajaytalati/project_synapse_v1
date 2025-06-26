#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 18:15:01 2025

@author: ajay
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------
# Configuration
# -------------------------------
use_base_model_only = False  # ✅ TOGGLE: True = base Gemma, False = merged fine-tune
base_model = "google/gemma-3-4b-it"
merged_model_path = "./gemma4b_merged"

if use_base_model_only:
    model_path = base_model
    print("📦 Loading BASE Gemma-3-4b-it model...")
    local_files = False
    trust_remote_code = True
else:
    model_path = merged_model_path
    print("📦 Loading MERGED fine-tuned model...")
    local_files = True
    trust_remote_code = True

# -------------------------------
# Load Tokenizer and Model
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=trust_remote_code,
    local_files_only=local_files,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=local_files,
    trust_remote_code=trust_remote_code,
    use_safetensors=True,
)

# -------------------------------
# Tokenizer & Model Setup
# -------------------------------
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model.config.pad_token_id = tokenizer.pad_token_id

print("✅ Model and tokenizer loaded.")

# -------------------------------
# Prompts to Test
# -------------------------------
# Replace your prompts with:
prompts = [
    "<bos><start_of_turn>user\nHow should a Stoic respond to betrayal?<end_of_turn>\n<start_of_turn>model\n",
    "<bos><start_of_turn>user\nWhat would Marcus Aurelius say about dealing with anger?<end_of_turn>\n<start_of_turn>model\n",
    "<bos><start_of_turn>user\nAs a Stoic philosopher, how should I handle success?<end_of_turn>\n<start_of_turn>model\n",
]
# -------------------------------
# Run Inference
# -------------------------------
for i, prompt in enumerate(prompts, 1):
    print(f"\n====== PROMPT {i} ======")
    print(prompt)

    inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=1024).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            min_new_tokens=10,
            temperature=0.3, # Instead of 0.7 - for more focused responses
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0, input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print("-------- RESPONSE --------")
    print(response.strip())
    print("==========================")
