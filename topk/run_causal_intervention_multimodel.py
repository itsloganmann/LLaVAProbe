#!/usr/bin/env python3
"""
Multi-Model Causal Intervention Analysis

Tests whether attention is causally connected to task performance across VLMs.
Masks HIGH, LOW, and RANDOM attention patches and measures accuracy changes.

Supports: Qwen2-VL-7B, PaliGemma-3B

Usage:
    python run_causal_intervention_multimodel.py --model qwen2-vl --n_samples 200
    python run_causal_intervention_multimodel.py --model paligemma --n_samples 200
"""

import torch
import numpy as np
import pandas as pd
import argparse
import os
import json
import requests
from PIL import Image
from tqdm import tqdm
from scipy import stats

# Model imports
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

# ============================================================================
# Configuration
# ============================================================================

MODEL_CONFIGS = {
    "qwen2-vl": {
        "model_id": "Qwen/Qwen2-VL-7B-Instruct",
        "num_layers": 28,
        "num_heads": 28,
        "num_kv_heads": 4,
        "head_dim": 128,
        "hidden_dim": 3584,
        "patch_size": 14,
        "image_size": 224,  # Will be overridden by actual image processing
    },
    "paligemma": {
        "model_id": "google/paligemma-3b-mix-224",
        "num_layers": 18,
        "num_heads": 8,
        "num_kv_heads": 1,
        "head_dim": 256,
        "hidden_dim": 2048,
        "patch_size": 14,
        "image_size": 224,
    },
}

MASK_RATIO = 0.30  # Mask 30% of patches


# ============================================================================
# Image Masking Utilities
# ============================================================================

def get_patch_attention_scores(attentions, num_image_patches, model_type, image_token_start=0):
    """
    Extract attention scores for image patches from the last layer.
    Returns attention weights aggregated over heads for image patch positions.
    
    Args:
        attentions: Model attention outputs
        num_image_patches: Number of image patches expected
        model_type: "qwen2-vl" or "paligemma"
        image_token_start: Starting position of image tokens in sequence
    
    Returns:
        numpy array of attention scores for image patches
    """
    # Use last layer attention
    last_layer_attn = attentions[-1]  # [batch, heads, seq, seq]
    
    # Average over heads, take attention from last token to all positions
    # Shape: [seq_len]
    attn_weights = last_layer_attn[0].mean(dim=0)[-1, :].float().cpu().numpy()
    seq_len = len(attn_weights)
    
    # Extract image token attention based on model type
    if model_type == "paligemma":
        # PaliGemma: image tokens are at the BEGINNING (positions 0:256)
        end_idx = min(num_image_patches, seq_len)
        patch_attn = attn_weights[:end_idx]
    elif model_type == "qwen2-vl":
        # Qwen2-VL: image tokens come AFTER system prompt tokens
        # The exact position depends on the prompt, but typically after ~10-20 tokens
        # We need to find where the image tokens actually are
        
        # Heuristic: Image tokens typically have higher total attention
        # and are contiguous. For simplicity, use provided start position.
        end_idx = min(image_token_start + num_image_patches, seq_len)
        patch_attn = attn_weights[image_token_start:end_idx]
    else:
        patch_attn = attn_weights[:num_image_patches]
    
    # Ensure we return exactly num_image_patches values
    if len(patch_attn) < num_image_patches:
        # Pad with zeros
        padded = np.zeros(num_image_patches)
        padded[:len(patch_attn)] = patch_attn
        patch_attn = padded
    elif len(patch_attn) > num_image_patches:
        patch_attn = patch_attn[:num_image_patches]
    
    return patch_attn


def find_image_token_positions_qwen(inputs):
    """
    Find the range of image token positions in Qwen2-VL input sequence.
    
    Qwen2-VL uses <|image_pad|> tokens for image patches.
    Returns (start_idx, num_tokens).
    """
    if "image_grid_thw" in inputs:
        # image_grid_thw gives us [temporal, height, width] of the image grid
        grid = inputs["image_grid_thw"][0]  # First image
        num_image_tokens = int(grid[0] * grid[1] * grid[2])
    else:
        # Fallback estimate
        num_image_tokens = 576
    
    # Estimate start position (after system prompt tokens)
    # This is approximate - exact position depends on prompt template
    # Typical Qwen2-VL prompt: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n<image>...
    # Image tokens start around position 10-20
    start_idx = 10  # Conservative estimate
    
    return start_idx, num_image_tokens


def create_masked_image(image, mask_indices, patch_size=14, grid_size=16):
    """
    Create a masked version of the image by setting specified patches to gray.
    
    Args:
        image: PIL Image
        mask_indices: List of patch indices to mask
        patch_size: Size of each patch (14 for ViT-L)
        grid_size: Number of patches per side
    
    Returns:
        Masked PIL Image
    """
    img_array = np.array(image.copy())
    
    # Resize image to match expected grid
    expected_size = patch_size * grid_size
    if image.size[0] != expected_size or image.size[1] != expected_size:
        image_resized = image.resize((expected_size, expected_size), Image.BILINEAR)
        img_array = np.array(image_resized)
    
    # Mask specified patches with gray (128, 128, 128)
    for idx in mask_indices:
        row = idx // grid_size
        col = idx % grid_size
        y_start = row * patch_size
        y_end = (row + 1) * patch_size
        x_start = col * patch_size
        x_end = (col + 1) * patch_size
        img_array[y_start:y_end, x_start:x_end] = 128
    
    return Image.fromarray(img_array)


def get_mask_indices(attention_scores, num_patches, mask_ratio, mask_type):
    """
    Get indices of patches to mask based on attention scores.
    
    Args:
        attention_scores: Attention weights for each patch
        num_patches: Total number of patches
        mask_ratio: Fraction of patches to mask
        mask_type: 'high', 'low', or 'random'
    
    Returns:
        List of patch indices to mask
    """
    num_to_mask = int(num_patches * mask_ratio)
    
    if mask_type == "high":
        # Mask highest attention patches
        indices = np.argsort(attention_scores)[-num_to_mask:]
    elif mask_type == "low":
        # Mask lowest attention patches
        indices = np.argsort(attention_scores)[:num_to_mask]
    elif mask_type == "random":
        # Mask random patches
        indices = np.random.choice(num_patches, num_to_mask, replace=False)
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")
    
    return indices.tolist()


# ============================================================================
# Model Classes
# ============================================================================

class Qwen2VLCausalMechanism:
    """Qwen2-VL causal intervention mechanism."""
    
    def __init__(self, device="cuda"):
        self.config = MODEL_CONFIGS["qwen2-vl"]
        self.device = device
        self.model_type = "qwen2-vl"
        
        print(f"Loading Qwen2-VL from {self.config['model_id']}...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config["model_id"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
        self.processor = AutoProcessor.from_pretrained(self.config["model_id"])
        self.model.eval()
        
        # GQA config
        self.num_heads = self.config["num_heads"]
        self.num_kv_heads = self.config["num_kv_heads"]
        
        print(f"Qwen2-VL loaded successfully")
    
    def get_attention_and_answer(self, image, question):
        """Get attention weights and model answer for an image-question pair."""
        # Prepare input using Qwen2-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Cast pixel values to bfloat16
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        
        # Get attention weights
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                return_dict=True,
            )
        
        # Generate answer
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
            )
        
        # Decode answer
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True
        )[0].strip().lower()
        
        return outputs.attentions, answer
    
    def get_answer_for_masked_image(self, image, question):
        """Get model answer for a (potentially masked) image."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True
        )[0].strip().lower()
        
        return answer


class PaliGemmaCausalMechanism:
    """PaliGemma causal intervention mechanism."""
    
    def __init__(self, device="cuda"):
        self.config = MODEL_CONFIGS["paligemma"]
        self.device = device
        self.model_type = "paligemma"
        
        print(f"Loading PaliGemma from {self.config['model_id']}...")
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.config["model_id"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
        self.processor = AutoProcessor.from_pretrained(self.config["model_id"])
        self.model.eval()
        
        # GQA config
        config = self.model.language_model.config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        
        # PaliGemma uses 224x224 images with 14x14 patches = 16x16 = 256 patches
        self.num_image_patches = 256
        self.grid_size = 16
        
        print(f"PaliGemma loaded successfully")
    
    def get_attention_and_answer(self, image, question):
        """Get attention weights and model answer for an image-question pair."""
        # Prepare input
        prompt = f"<image>{question}"
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device)
        
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        
        # Get attention weights
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                return_dict=True,
            )
        
        # Generate answer
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
            )
        
        # Decode - for PaliGemma, strip the input portion
        answer = self.processor.decode(
            generated_ids[0], 
            skip_special_tokens=True
        )
        # Remove the prompt from the answer
        if question in answer:
            answer = answer.split(question)[-1].strip().lower()
        else:
            answer = answer.strip().lower()
        
        return outputs.attentions, answer
    
    def get_answer_for_masked_image(self, image, question):
        """Get model answer for a (potentially masked) image."""
        prompt = f"<image>{question}"
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device)
        
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
            )
        
        answer = self.processor.decode(
            generated_ids[0], 
            skip_special_tokens=True
        )
        if question in answer:
            answer = answer.split(question)[-1].strip().lower()
        else:
            answer = answer.strip().lower()
        
        return answer


# ============================================================================
# Evaluation Utilities
# ============================================================================

def check_answer_match(prediction, ground_truth):
    """Check if prediction matches ground truth (flexible matching)."""
    pred = prediction.lower().strip()
    gt = ground_truth.lower().strip()
    
    # Direct match
    if pred == gt:
        return True
    
    # Check if ground truth is contained in prediction
    if gt in pred:
        return True
    
    # Check if prediction starts with ground truth
    if pred.startswith(gt):
        return True
    
    # Handle yes/no questions
    if gt in ["yes", "no"]:
        if gt == "yes" and pred.startswith("yes"):
            return True
        if gt == "no" and pred.startswith("no"):
            return True
    
    # Handle numeric answers
    try:
        pred_num = float(pred.split()[0])
        gt_num = float(gt)
        if pred_num == gt_num:
            return True
    except:
        pass
    
    return False


def load_vqa_data(n_samples=200):
    """Load VQA data from the standard dataset location."""
    possible_paths = [
        "results.csv",
        "../results.csv",
        "/home/ubuntu/LLaVAProbe/results.csv",
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    
    if df is None:
        raise FileNotFoundError("Could not find results.csv")
    
    # Sample data
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    
    return df


def load_image(image_url):
    """Load image from URL."""
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(requests.get(image_url, stream=True, timeout=10).raw)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image {image_url}: {e}")
        return None


# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def run_causal_intervention(model_name, n_samples=200):
    """Run causal intervention analysis for a model."""
    
    print("=" * 80)
    print(f"Running Causal Intervention Analysis for {model_name.upper()}")
    print("=" * 80)
    
    # Initialize model
    if model_name == "qwen2-vl":
        mechanism = Qwen2VLCausalMechanism()
        num_patches = 576  # Qwen2-VL default for ~336x336 images (24x24)
        grid_size = 24
        image_token_start = 10  # Approximate start position after system prompt
    elif model_name == "paligemma":
        mechanism = PaliGemmaCausalMechanism()
        num_patches = 256  # 16x16 patches for 224x224 images
        grid_size = 16
        image_token_start = 0  # PaliGemma: image tokens at the start
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load data
    print("Loading VQA data...")
    df = load_vqa_data(n_samples)
    print(f"Loaded {len(df)} samples")
    
    # Results storage
    results = {
        "full": [],      # Baseline (no masking)
        "high": [],      # Mask high attention patches
        "low": [],       # Mask low attention patches
        "random": [],    # Mask random patches
    }
    
    detailed_results = []
    
    # Process samples
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        image_url = row["image_url"]
        question = row["question"]
        ground_truth = str(row["ground_truth"])
        
        # Load image
        image = load_image(image_url)
        if image is None:
            continue
        
        try:
            # 1. Get baseline attention and answer (FULL condition)
            attentions, full_answer = mechanism.get_attention_and_answer(image, question)
            full_correct = check_answer_match(full_answer, ground_truth)
            results["full"].append(full_correct)
            
            # Get patch attention scores with correct image token position
            patch_attn = get_patch_attention_scores(
                attentions, num_patches, mechanism.model_type, 
                image_token_start=image_token_start
            )
            
            # Ensure we have enough patches
            actual_patches = min(len(patch_attn), num_patches)
            
            # 2. Mask HIGH attention patches
            high_indices = get_mask_indices(patch_attn[:actual_patches], actual_patches, MASK_RATIO, "high")
            high_masked_image = create_masked_image(image, high_indices, patch_size=14, grid_size=grid_size)
            high_answer = mechanism.get_answer_for_masked_image(high_masked_image, question)
            high_correct = check_answer_match(high_answer, ground_truth)
            results["high"].append(high_correct)
            
            # 3. Mask LOW attention patches
            low_indices = get_mask_indices(patch_attn[:actual_patches], actual_patches, MASK_RATIO, "low")
            low_masked_image = create_masked_image(image, low_indices, patch_size=14, grid_size=grid_size)
            low_answer = mechanism.get_answer_for_masked_image(low_masked_image, question)
            low_correct = check_answer_match(low_answer, ground_truth)
            results["low"].append(low_correct)
            
            # 4. Mask RANDOM patches
            random_indices = get_mask_indices(patch_attn[:actual_patches], actual_patches, MASK_RATIO, "random")
            random_masked_image = create_masked_image(image, random_indices, patch_size=14, grid_size=grid_size)
            random_answer = mechanism.get_answer_for_masked_image(random_masked_image, question)
            random_correct = check_answer_match(random_answer, ground_truth)
            results["random"].append(random_correct)
            
            # Store detailed results
            detailed_results.append({
                "image_id": idx,
                "question_type": row.get("question_type", "unknown"),
                "question": question,
                "ground_truth": ground_truth,
                "full_answer": full_answer,
                "full_correct": full_correct,
                "high_answer": high_answer,
                "high_correct": high_correct,
                "low_answer": low_answer,
                "low_correct": low_correct,
                "random_answer": random_answer,
                "random_correct": random_correct,
            })
            
            # Progress update
            if (idx + 1) % 50 == 0:
                n = len(results["full"])
                print(f"\nProcessed {n} samples")
                print(f"  Full: {100*sum(results['full'])/n:.1f}%")
                print(f"  High masked: {100*sum(results['high'])/n:.1f}%")
                print(f"  Low masked: {100*sum(results['low'])/n:.1f}%")
                print(f"  Random masked: {100*sum(results['random'])/n:.1f}%")
                
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # Save detailed results
    results_df = pd.DataFrame(detailed_results)
    output_file = f"{model_name}_causal_intervention_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Compute statistics
    print("\n" + "=" * 80)
    print("CAUSAL INTERVENTION RESULTS")
    print("=" * 80)
    
    n = len(results["full"])
    full_acc = 100 * sum(results["full"]) / n
    high_acc = 100 * sum(results["high"]) / n
    low_acc = 100 * sum(results["low"]) / n
    random_acc = 100 * sum(results["random"]) / n
    
    print(f"\nTotal samples: {n}")
    print(f"\nAccuracy Results:")
    print(f"  Full (baseline):    {full_acc:.2f}%")
    print(f"  Mask HIGH 30%:      {high_acc:.2f}%  (Δ = {high_acc - full_acc:+.2f}%)")
    print(f"  Mask LOW 30%:       {low_acc:.2f}%  (Δ = {low_acc - full_acc:+.2f}%)")
    print(f"  Mask RANDOM 30%:    {random_acc:.2f}%  (Δ = {random_acc - full_acc:+.2f}%)")
    
    # Statistical tests
    print(f"\nStatistical Significance (McNemar's test approximation via paired t-test):")
    
    # Full vs High
    _, p_full_high = stats.ttest_rel(results["full"], results["high"])
    print(f"  Full vs Mask HIGH:  p = {p_full_high:.2e}")
    
    # Full vs Low
    _, p_full_low = stats.ttest_rel(results["full"], results["low"])
    print(f"  Full vs Mask LOW:   p = {p_full_low:.2e}")
    
    # High vs Low
    _, p_high_low = stats.ttest_rel(results["high"], results["low"])
    print(f"  Mask HIGH vs LOW:   p = {p_high_low:.2e}")
    
    # Random vs High
    _, p_random_high = stats.ttest_rel(results["random"], results["high"])
    print(f"  Mask RANDOM vs HIGH: p = {p_random_high:.2e}")
    
    # Causal effect size
    causal_effect = low_acc - high_acc
    print(f"\nCausal Effect Size (LOW - HIGH): {causal_effect:+.2f} percentage points")
    
    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    if high_acc < low_acc and p_high_low < 0.05:
        print("✓ SUPPORTS CAUSAL HYPOTHESIS")
        print("  Masking HIGH attention regions hurts more than LOW regions")
        print("  This indicates attention identifies task-relevant regions")
    else:
        print("✗ DOES NOT SUPPORT CAUSAL HYPOTHESIS")
        print("  No significant difference between HIGH and LOW masking")
    
    # Summary JSON
    summary = {
        "model": model_name,
        "n_samples": n,
        "mask_ratio": MASK_RATIO,
        "accuracy": {
            "full": full_acc,
            "mask_high": high_acc,
            "mask_low": low_acc,
            "mask_random": random_acc,
        },
        "delta_from_baseline": {
            "mask_high": high_acc - full_acc,
            "mask_low": low_acc - full_acc,
            "mask_random": random_acc - full_acc,
        },
        "p_values": {
            "full_vs_high": float(p_full_high),
            "full_vs_low": float(p_full_low),
            "high_vs_low": float(p_high_low),
            "random_vs_high": float(p_random_high),
        },
        "causal_effect_size": causal_effect,
        "supports_causal_hypothesis": high_acc < low_acc and p_high_low < 0.05,
    }
    
    summary_file = f"{model_name}_causal_intervention_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_file}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Causal Intervention Analysis")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["qwen2-vl", "paligemma"],
                        help="Model to analyze")
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of samples to process")
    
    args = parser.parse_args()
    run_causal_intervention(args.model, args.n_samples)


if __name__ == "__main__":
    main()
