#!/usr/bin/env python3
"""Debug script to inspect PaliGemma model structure."""

import os
import torch
from transformers import PaliGemmaForConditionalGeneration

# Set HF token from environment
token = os.environ.get("HF_TOKEN")

print("Loading PaliGemma...")
model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma-3b-mix-224",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=token,
)

print("\n=== Model Structure ===")
print(f"Type: {type(model)}")
print(f"\nTop-level attributes:")
for attr in dir(model):
    if not attr.startswith("_"):
        obj = getattr(model, attr, None)
        if hasattr(obj, "parameters"):
            print(f"  {attr}: {type(obj).__name__}")

print(f"\n=== Language Model Structure ===")
lm = model.language_model
print(f"language_model type: {type(lm)}")
for attr in dir(lm):
    if not attr.startswith("_"):
        obj = getattr(lm, attr, None)
        if hasattr(obj, "parameters"):
            print(f"  {attr}: {type(obj).__name__}")

print(f"\n=== Layers Path ===")
# Try different paths
paths_to_try = [
    "model.language_model.model.layers",
    "model.language_model.layers",
    "language_model.model.layers",
    "language_model.layers",
]

for path in paths_to_try:
    try:
        obj = model
        for part in path.split(".")[1:]:  # Skip 'model' prefix
            obj = getattr(obj, part)
        print(f"  {path}: Found! Length = {len(obj)}")
    except Exception as e:
        print(f"  {path}: {e}")

# Find layers directly
print("\n=== Finding layers ===")
for name, module in model.named_modules():
    if name.endswith(".layers"):
        print(f"  {name}: {type(module).__name__}, len={len(module)}")
    if name.endswith(".norm") and "layer" not in name:
        print(f"  {name}: {type(module).__name__}")

# Check attention config
print("\n=== Attention Config ===")
try:
    first_layer = model.language_model.layers[0]
    attn = first_layer.self_attn
    print(f"num_heads: {attn.num_heads}")
    print(f"num_key_value_heads: {attn.num_key_value_heads}")
    print(f"head_dim: {attn.head_dim}")
except Exception as e:
    print(f"Error accessing via language_model.layers: {e}")
    try:
        first_layer = model.language_model.model.layers[0]
        attn = first_layer.self_attn
        print(f"num_heads: {attn.num_heads}")
        print(f"num_key_value_heads: {attn.num_key_value_heads}")
        print(f"head_dim: {attn.head_dim}")
    except Exception as e2:
        print(f"Error accessing via language_model.model.layers: {e2}")
