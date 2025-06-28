#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 01:40:46 2025

@author: ajay
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from peft import PeftModel
import time

def load_model(model_id, adapter_path=None, merge=False):
    """Load model with optional adapter merging"""
    # 4-bit config for efficient loading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Merge adapter if provided
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        if merge:
            print("Merging adapter with base model...")
            model = model.merge_and_unload()
            print("Merge complete!")
    
    return model

def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    """Generate response with timing and token count"""
    start_time = time.time()
    
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    
    # Generate response
    outputs = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Extract response
    full_output = outputs[0]['generated_text']
    response = full_output[len(prompt):].strip()
    
    # Calculate stats
    duration = time.time() - start_time
    input_tokens = len(tokenizer.encode(prompt))
    output_tokens = len(tokenizer.encode(response))
    
    return response, duration, input_tokens, output_tokens

# Configuration
MODEL_ID = "microsoft/phi-2"
ADAPTER_PATH = "phi2_medical_ft"  # Your saved adapter directory
MERGE_MODEL = True  # Set to False to keep adapter separate

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    padding_side="right",
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
print("Loading base model...")
base_model = load_model(MODEL_ID)
print("Base model loaded\n")

# Load fine-tuned model (merged or with adapter)
print("Loading fine-tuned model...")
if MERGE_MODEL:
    ft_model = load_model(MODEL_ID, ADAPTER_PATH, merge=True)
else:
    ft_model = load_model(MODEL_ID, ADAPTER_PATH, merge=False)
print("Fine-tuned model loaded\n")

# Test prompts
MEDICAL_PROMPTS = [
    "What's the first-line treatment for hypertension?",
    "How does metformin work in diabetes management?",
    "Explain the mechanism of action for statins",
    "What are the diagnostic criteria for type 2 diabetes?",
    "Describe the side effects of chemotherapy"
]

# Compare responses
for prompt in MEDICAL_PROMPTS:
    # Format prompt with Phi-2 style
    formatted_prompt = f"Instruct: {prompt}\nOutput:"
    
    print(f"\n{'='*80}")
    print(f"PROMPT: {prompt}")
    print(f"{'='*80}")
    
    # Base model response
    print("\n[ BASE MODEL RESPONSE ]")
    base_response, base_time, base_in, base_out = generate_response(
        base_model, tokenizer, formatted_prompt
    )
    print(base_response)
    print(f"⏱️ {base_time:.2f}s | 📥 {base_in} tokens → 📤 {base_out} tokens")
    
    # Fine-tuned model response
    print("\n[ FINE-TUNED MODEL RESPONSE ]")
    ft_response, ft_time, ft_in, ft_out = generate_response(
        ft_model, tokenizer, formatted_prompt
    )
    print(ft_response)
    print(f"⏱️ {ft_time:.2f}s | 📥 {ft_in} tokens → 📤 {ft_out} tokens")
    
    print(f"\n{'='*80}")
    print("SUMMARY:")
    print(f"- Response length: Base {len(base_response)} chars vs FT {len(ft_response)} chars")
    print(f"- Speed: Base {base_time:.2f}s vs FT {ft_time:.2f}s")
    print(f"- Output tokens: Base {base_out} vs FT {ft_out}")
    print("="*80 + "\n\n")

# Interactive mode
print("Entering interactive mode. Type 'quit' to exit.")
while True:
    prompt = input("\nYour question: ")
    if prompt.lower() in ['quit', 'exit']:
        break
        
    formatted_prompt = f"Instruct: {prompt}\nOutput:"
    
    print("\n[ BASE MODEL ]")
    base_response, _, _, _ = generate_response(base_model, tokenizer, formatted_prompt)
    print(base_response)
    
    print("\n[ FINE-TUNED MODEL ]")
    ft_response, _, _, _ = generate_response(ft_model, tokenizer, formatted_prompt)
    print(ft_response)