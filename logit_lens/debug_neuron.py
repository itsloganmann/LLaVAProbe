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

        # Get prediction logits
        logits = outputs.logits[0, -1, :]
        
        # Get ground truth token ID
        gt_tokens = processor.tokenizer.encode(ground_truth, add_special_tokens=False)
        gt_token_id = gt_tokens[0] if gt_tokens else 0
        gt_decoded = processor.tokenizer.decode([gt_token_id])
        
        # Check top-5, top-10, top-20
        top_5_ids = torch.topk(logits, k=5).indices.tolist()
        top_10_ids = torch.topk(logits, k=10).indices.tolist()
        top_20_ids = torch.topk(logits, k=20).indices.tolist()
        
        in_top5 = gt_token_id in top_5_ids
        in_top10 = gt_token_id in top_10_ids
        in_top20 = gt_token_id in top_20_ids
        
        # Decode top 5
        top_5_decoded = [processor.tokenizer.decode([tid]) for tid in top_5_ids]
        
        # Get GT logit rank
        gt_logit = logits[gt_token_id].item()
        rank = (logits > gt_logit).sum().item() + 1
        
        if in_top5:
            correct += 1
        total += 1
        
        print(f"Sample {i}: GT='{ground_truth}' (id={gt_token_id}, decoded='{gt_decoded}')")
        print(f"  Top-5: {top_5_decoded}")
        print(f"  In top-5: {in_top5}, top-10: {in_top10}, top-20: {in_top20}")
        print(f"  GT rank: {rank}")
        print()
    
    print(f"\nCorrect (top-5): {correct}/{total} = {correct/total*100:.1f}%")

if __name__ == "__main__":
    main()
