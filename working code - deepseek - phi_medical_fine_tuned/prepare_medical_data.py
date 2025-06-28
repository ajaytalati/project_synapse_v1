#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 20:42:39 2025

@author: ajay
"""

import json
from datasets import load_dataset
from tqdm import tqdm

# 1. Download PubMedQA dataset from Hugging Face
print("Loading PubMedQA dataset from Hugging Face...")
dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")

# 2. Filter and format data for Mistral fine-tuning
print("Processing dataset for Mistral fine-tuning...")
mistral_dataset = []

for example in tqdm(dataset, desc="Processing examples"):
    # Only use examples with detailed answers
    if example["final_decision"] != "no" and example["long_answer"]:
        # Format for Mistral's chat template
        conversation = [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["long_answer"]}
        ]
        mistral_dataset.append({"messages": conversation})

# 3. Save as JSON file
output_path = "pubmedqa_mistral_format.json"
print(f"Saving formatted dataset to {output_path}...")
with open(output_path, "w") as f:
    json.dump(mistral_dataset, f, indent=2)

# 4. Dataset statistics
print("\n" + "="*50)
print(f"Total examples processed: {len(mistral_dataset)}")
print(f"Sample structure:")
print(json.dumps(mistral_dataset[0], indent=2))
print("="*50)