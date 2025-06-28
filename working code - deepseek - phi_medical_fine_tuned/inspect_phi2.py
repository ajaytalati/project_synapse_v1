#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 23:04:22 2025

@author: ajay
"""

import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "microsoft/phi-2"

# Load model (no quantization for inspection)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

# Print all module names
print("="*80)
print("All Module Names in Phi-2:")
print("="*80)
all_modules = [name for name, _ in model.named_modules()]
for name in all_modules:
    print(name)

# Find potential attention modules
print("\n" + "="*80)
print("Potential Attention Modules:")
print("="*80)
attention_candidates = [
    name for name in all_modules 
    if "q_" in name or "k_" in name or "v_" in name or "out_" in name or "attn" in name
]
for name in attention_candidates:
    print(name)

# Check if specific modules exist
print("\n" + "="*80)
print("Module Existence Check:")
print("="*80)
for module in ["Wqkv", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj", "query", "key", "value"]:
    exists = any(module in name for name in all_modules)
    print(f"{module}: {exists}")