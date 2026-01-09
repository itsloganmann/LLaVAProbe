#!/usr/bin/env python3
"""
Quick script to run logit lens analysis on LLaVA-1.5-13B.

This uses the existing LogitLensAnalyzer but with the 13B model.
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.logit_lens import LogitLensAnalyzer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_llava13b_analysis(n_samples: int = 1000):
    """Run logit lens analysis on LLaVA-1.5-13B."""
    
    print("=" * 70)
    print("LLAVA-1.5-13B LOGIT LENS ANALYSIS")
    print(f"Samples: {n_samples}")
    print("=" * 70)
    
    # Load analyzer with 13B model
    print("\nLoading LLaVA-1.5-13B model...")
    analyzer = LogitLensAnalyzer(
        model_id="llava-hf/llava-1.5-13b-hf",
        quantization="none"
    )
    
    # Load samples from the same source as the original scripts
    print("\nLoading samples...")
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(parent_dir, "test_intervention_output/analysis_records.json"), "r") as f:
        data = json.load(f)
    
    records = data.get("records", [])[:n_samples]
    print(f"Loaded {len(records)} samples")
    
    # Run analysis with real images
    print("\n" + "=" * 70)
    print("Running analysis WITH images...")
    print("=" * 70)
    
    results_with_image = []
    for i, record in enumerate(tqdm(records, desc="With image")):
        try:
            # Load image
            response = requests.get(record["image_url"], timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            # Run logit lens
            result = analyzer.analyze_sample(
                question=record["question"],
                image=image,
                ground_truth=record["ground_truth"]
            )
            results_with_image.append(result)
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Run analysis without images (blank)
    print("\n" + "=" * 70)
    print("Running analysis WITHOUT images (blank)...")
    print("=" * 70)
    
    blank_image = Image.new("RGB", (336, 336), color=(128, 128, 128))
    results_no_image = []
    for i, record in enumerate(tqdm(records, desc="Without image")):
        try:
            result = analyzer.analyze_sample(
                question=record["question"],
                image=blank_image,
                ground_truth=record["ground_truth"]
            )
            results_no_image.append(result)
        except Exception as e:
            continue
    
    # Compute metrics
    print("\n" + "=" * 70)
    print("Computing metrics...")
    print("=" * 70)
    
    n_correct_with = sum(1 for r in results_with_image if r.is_correct)
    n_correct_no = sum(1 for r in results_no_image if r.is_correct)
    
    acc_with_image = n_correct_with / len(results_with_image) if results_with_image else 0
    acc_no_image = n_correct_no / len(results_no_image) if results_no_image else 0
    
    # Compute average margins
    margins_with = np.stack([r.layer_margins for r in results_with_image])
    margins_no = np.stack([r.layer_margins for r in results_no_image])
    
    avg_margins_with = margins_with.mean(axis=0)
    avg_margins_no = margins_no.mean(axis=0)
    delta_margins = avg_margins_with - avg_margins_no
    
    # Compute MLP vs Attention contributions
    mlp_with = np.stack([r.mlp_contributions for r in results_with_image if r.mlp_contributions is not None])
    attn_with = np.stack([r.attn_contributions for r in results_with_image if r.attn_contributions is not None])
    
    avg_mlp = mlp_with.mean(axis=0) if len(mlp_with) > 0 else None
    avg_attn = attn_with.mean(axis=0) if len(attn_with) > 0 else None
    
    # Compile results
    results = {
        "model": "llava-hf/llava-1.5-13b-hf",
        "n_samples": len(results_with_image),
        "timestamp": datetime.now().isoformat(),
        "acc_with_image": acc_with_image,
        "acc_no_image": acc_no_image,
        "n_layers": 40,  # LLaVA-13B has 40 layers
        "avg_margins_with_image": avg_margins_with.tolist(),
        "avg_margins_no_image": avg_margins_no.tolist(),
        "delta_margins": delta_margins.tolist(),
        "avg_mlp_contributions": avg_mlp.tolist() if avg_mlp is not None else None,
        "avg_attn_contributions": avg_attn.tolist() if avg_attn is not None else None,
    }
    
    # Compute correct vs incorrect split
    correct_results = [r for r in results_with_image if r.is_correct]
    incorrect_results = [r for r in results_with_image if not r.is_correct]
    
    if correct_results and incorrect_results:
        correct_margins = np.stack([r.layer_margins for r in correct_results])
        incorrect_margins = np.stack([r.layer_margins for r in incorrect_results])
        results["avg_correct_margins"] = correct_margins.mean(axis=0).tolist()
        results["avg_incorrect_margins"] = incorrect_margins.mean(axis=0).tolist()
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Model: LLaVA-1.5-13B")
    print(f"Samples: {len(results_with_image)}")
    print(f"Accuracy WITH image: {acc_with_image:.1%}")
    print(f"Accuracy WITHOUT image: {acc_no_image:.1%}")
    print(f"Peak delta margin: Layer {np.argmax(delta_margins)}, value = {np.max(delta_margins):.2f}")
    
    # Find boosting vs suppression layers
    boosting_layers = [i for i, d in enumerate(delta_margins) if d > 0.5]
    suppression_layers = [i for i, d in enumerate(delta_margins) if d < -0.5]
    print(f"Boosting layers (Δmargin > 0.5): {boosting_layers}")
    print(f"Suppression layers (Δmargin < -0.5): {suppression_layers}")
    
    # Save results
    output_file = os.path.join(SCRIPT_DIR, "llava13b_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1000)
    args = parser.parse_args()
    
    run_llava13b_analysis(args.n_samples)
