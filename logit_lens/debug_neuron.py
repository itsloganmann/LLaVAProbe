#!/usr/bin/env python3
"""Debug script for neuron activation collection."""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import requests
import json
import os
import sys

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

    # Load samples
    records_path = "/home/ubuntu/LLaVAProbe/test_intervention_output/analysis_records.json"
    with open(records_path, 'r') as f:
        data = json.load(f)
    samples = data.get("records", [])[:10]
    print(f"Testing on {len(samples)} samples")

    # Test predictions
    from qwen_vl_utils import process_vision_info
    
    correct = 0
    total = 0
    
    for i, sample in enumerate(samples):
        url = sample['image_url']
        question = sample['question']
        ground_truth = sample['ground_truth']
        
        try:
            image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        except Exception as e:
            print(f"Sample {i}: Failed to load image: {e}")
            continue
            
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Get prediction
        logits = outputs.logits[0, -1, :]
        predicted_id = logits.argmax().item()
        predicted_text = processor.tokenizer.decode([predicted_id]).strip().lower()
        
        # More flexible correctness check
        gt_lower = ground_truth.lower().strip()
        pred_lower = predicted_text.lower().strip()
        
        is_correct = (
            gt_lower == pred_lower or
            gt_lower in pred_lower or 
            pred_lower in gt_lower or
            gt_lower.startswith(pred_lower) or
            pred_lower.startswith(gt_lower)
        )
        
        if is_correct:
            correct += 1
        total += 1
        
        print(f"Sample {i}: GT='{ground_truth}', Pred='{predicted_text}', Match={is_correct}")
    
    print(f"\nCorrect: {correct}/{total} = {correct/total*100:.1f}%")

if __name__ == "__main__":
    main()
