#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 21:33:20 2025

@author: ajay
"""

import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

print("="*80)
print("Installation Verification")
print("="*80)

# Basic PyTorch info
print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    
    # Test tensor allocation
    tensor = torch.randn(3, 1024, 1024).to("cuda")
    print(f"\nTensor Allocation Test: Passed (Device: {tensor.device})")
else:
    print("\nCUDA NOT AVAILABLE!")

# Test bitsandbytes
try:
    import bitsandbytes as bnb
    print("\nBitsandbytes Test:")
    print(f"Version: {bnb.__version__}")
    
    # Correct FP4 quantization test
    x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    quantized, state = bnb.functional.quantize_fp4(x)
    print(f"Quantization Working: {quantized.dtype}")
except ImportError:
    print("\nBitsandbytes NOT INSTALLED!")
except Exception as e:
    print(f"\nBitsandbytes ERROR: {str(e)}")

# Test model loading
try:
    print("\nModel Loading Test:")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModel.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16
    )
    print(f"Model Type: {type(model).__name__}")
    print(f"Model Device: {model.device}")
    del model
    torch.cuda.empty_cache()
except Exception as e:
    print(f"Model Loading Failed: {str(e)}")

print("\n" + "="*80)
print("Verification Complete")
print("="*80)