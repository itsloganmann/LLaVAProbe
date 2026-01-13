#!/usr/bin/env python3
"""Debug script for neuron activation collection."""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import requests
import json
import os

def main():
    # Load model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    model.eval()

    layers = model.model.language_model.layers
    print(f"Layers: {len(layers)}")

    # Test hooks
    neuron_activations = {}

    def hook_fn(layer_idx):
        def hook(module, input, output):
            print(f"Hook fired for layer {layer_idx}, shape: {output.shape}")
            neuron_activations[layer_idx] = output.detach()
        return hook

    # Register hooks on target layers
    target_layers = [18, 21, 24, 25, 26, 27]
    hooks = []
    for layer_idx in target_layers:
        hook = layers[layer_idx].mlp.register_forward_hook(hook_fn(layer_idx))
        hooks.append(hook)
    print(f"Hooks registered on layers: {target_layers}")

    # Load a test sample
    from qwen_vl_utils import process_vision_info
    
    url = "https://images.unsplash.com/photo-1533450718592-29d45635f0a9?w=200"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    question = "What animal is in the image?"
    ground_truth = "cat"

    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)

    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(**inputs)

    print(f"\nActivations collected: {list(neuron_activations.keys())}")
    
    for layer_idx in target_layers:
        if layer_idx in neuron_activations:
            act = neuron_activations[layer_idx]
            print(f"Layer {layer_idx}: shape={act.shape}")
            last_token_act = act[0, -1, :].cpu().numpy()
            print(f"  Last token shape: {last_token_act.shape}")
    
    # Get prediction
    logits = outputs.logits[0, -1, :]
    predicted_id = logits.argmax().item()
    predicted_text = processor.tokenizer.decode([predicted_id]).strip().lower()
    print(f"\nPredicted: {predicted_text}")
    print(f"Ground truth: {ground_truth}")
    print(f"Is correct: {ground_truth in predicted_text or predicted_text in ground_truth}")
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()

if __name__ == "__main__":
    main()
