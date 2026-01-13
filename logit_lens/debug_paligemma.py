#!/usr/bin/env python3
"""Debug PaliGemma model structure."""

import torch
from transformers import PaliGemmaForConditionalGeneration

print("Loading PaliGemma...")
model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma-3b-mix-224",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("\nTop-level children:")
for name, child in model.named_children():
    print(f"  {name}: {type(child)}")

print("\nlanguage_model type:", type(model.language_model))
print("\nlanguage_model children:")
for name, child in model.language_model.named_children():
    print(f"  {name}: {type(child)}")

# Check for lm_head
print("\n--- Checking lm_head locations ---")
print("model.lm_head exists?", hasattr(model, 'lm_head'))
print("model.language_model.lm_head exists?", hasattr(model.language_model, 'lm_head'))

# Check model structure
if hasattr(model.language_model, 'model'):
    print("\nmodel.language_model.model children:")
    for name, child in model.language_model.model.named_children():
        print(f"  {name}: {type(child)}")
