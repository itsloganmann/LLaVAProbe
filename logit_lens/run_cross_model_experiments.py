#!/usr/bin/env python3
"""
Cross-Model Experiment Runner

Runs the same logit lens experiments across multiple models and datasets.

Usage:
    # Run all models on VQAv2
    python run_cross_model_experiments.py --models all --dataset vqav2 --n_samples 1000
    
    # Run specific models
    python run_cross_model_experiments.py --models llava-1.5-7b qwen2-vl-7b --dataset vqav2
    
    # Quick test
    python run_cross_model_experiments.py --models llava-1.5-7b --dataset vqav2 --n_samples 100
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict

import torch
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_configs import MODELS, DATASETS, get_model_config
from multi_model_analyzer import MultiModelLogitLensAnalyzer, load_dataset_samples


def run_single_experiment(
    model_name: str,
    dataset_name: str,
    n_samples: int,
    output_dir: str,
) -> Dict:
    """Run logit lens analysis for a single model-dataset combination."""

    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"DATASET: {dataset_name}")
    print(f"SAMPLES: {n_samples}")
    print(f"{'='*70}\n")

    results = {
        "model": model_name,
        "dataset": dataset_name,
        "n_samples": n_samples,
        "timestamp": datetime.now().isoformat(),
        "status": "pending",
    }

    try:
        # Load analyzer
        print("Loading model...")
        analyzer = MultiModelLogitLensAnalyzer(model_name=model_name)

        # Load samples
        print("Loading samples...")
        samples = load_dataset_samples(dataset_name, n_samples)
        print(f"Loaded {len(samples)} samples")

        # Run analysis
        print("Running logit lens analysis...")
        analysis_results = analyzer.analyze_batch(samples)

        # Summarize
        summary = analyzer.summarize_results(analysis_results)

        # Compute additional metrics
        correct_results = [r for r in analysis_results if r.is_correct]
        incorrect_results = [r for r in analysis_results if not r.is_correct]

        # Visual attribution (delta margins)
        if len(correct_results) > 0 and len(incorrect_results) > 0:
            correct_margins = np.stack(
                [r.layer_margins for r in correct_results])
            incorrect_margins = np.stack(
                [r.layer_margins for r in incorrect_results])

            avg_correct = correct_margins.mean(axis=0)
            avg_incorrect = incorrect_margins.mean(axis=0)

            summary["correct_vs_incorrect"] = {
                "avg_correct_margins": avg_correct.tolist(),
                "avg_incorrect_margins": avg_incorrect.tolist(),
                "margin_gap": (avg_correct - avg_incorrect).tolist(),
            }

        # Find peak layer
        if 'avg_margins' in summary:
            margins = np.array(summary['avg_margins'])
            summary["peak_margin_layer"] = int(np.argmax(margins))
            summary["peak_margin_value"] = float(np.max(margins))

        # MLP vs Attention by layer type
        if 'avg_mlp_contributions' in summary:
            mlp = np.array(summary['avg_mlp_contributions'])
            attn = np.array(summary['avg_attn_contributions'])

            # Last 2 layers (boosting)
            boosting_mlp = mlp[-2:].sum()
            boosting_attn = attn[-2:].sum()
            boosting_total = abs(boosting_mlp) + abs(boosting_attn)

            summary["boosting_layers"] = {
                "layers":
                list(range(analyzer.n_layers - 2, analyzer.n_layers)),
                "mlp_fraction":
                abs(boosting_mlp) /
                boosting_total if boosting_total > 0 else 0,
                "attn_fraction":
                abs(boosting_attn) /
                boosting_total if boosting_total > 0 else 0,
            }

            # Middle layers (suppression) - roughly layers 60-90% of the way through
            mid_start = int(analyzer.n_layers * 0.6)
            mid_end = int(analyzer.n_layers * 0.9)

            mid_mlp = mlp[mid_start:mid_end].sum()
            mid_attn = attn[mid_start:mid_end].sum()
            mid_total = abs(mid_mlp) + abs(mid_attn)

            summary["suppression_layers"] = {
                "layers": list(range(mid_start, mid_end)),
                "mlp_fraction":
                abs(mid_mlp) / mid_total if mid_total > 0 else 0,
                "attn_fraction":
                abs(mid_attn) / mid_total if mid_total > 0 else 0,
            }

        results["summary"] = summary
        results["status"] = "success"

        # Clean up
        del analyzer
        torch.cuda.empty_cache()

    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        print(f"ERROR: {e}")

    # Save individual result
    output_file = os.path.join(output_dir, f"{model_name}_{dataset_name}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {output_file}")

    return results


def compile_cross_model_results(results: List[Dict], output_dir: str):
    """Compile results from all models into a comparison table."""

    print(f"\n{'='*70}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'='*70}\n")

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "models": [],
    }

    for r in results:
        if r["status"] != "success":
            continue

        summary = r.get("summary", {})

        model_data = {
            "model": r["model"],
            "dataset": r["dataset"],
            "accuracy": summary.get("accuracy", 0),
            "n_layers": summary.get("n_layers", 0),
            "overall_mlp_fraction": summary.get("overall_mlp_fraction", 0),
            "overall_attn_fraction": summary.get("overall_attn_fraction", 0),
            "peak_margin_layer": summary.get("peak_margin_layer", -1),
            "peak_margin_value": summary.get("peak_margin_value", 0),
        }

        if "boosting_layers" in summary:
            model_data["boosting_mlp_fraction"] = summary["boosting_layers"][
                "mlp_fraction"]
            model_data["boosting_attn_fraction"] = summary["boosting_layers"][
                "attn_fraction"]

        if "suppression_layers" in summary:
            model_data["suppression_mlp_fraction"] = summary[
                "suppression_layers"]["mlp_fraction"]
            model_data["suppression_attn_fraction"] = summary[
                "suppression_layers"]["attn_fraction"]

        comparison["models"].append(model_data)

        print(
            f"{r['model']:20} | Acc: {model_data['accuracy']:.1%} | MLP: {model_data['overall_mlp_fraction']:.1%} | Peak L{model_data['peak_margin_layer']}"
        )

    # Save comparison
    comparison_file = os.path.join(output_dir, "cross_model_comparison.json")
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\nComparison saved to {comparison_file}")

    # Generate markdown table
    md_table = generate_markdown_table(comparison)
    md_file = os.path.join(output_dir, "cross_model_comparison.md")
    with open(md_file, 'w') as f:
        f.write(md_table)

    print(f"Markdown table saved to {md_file}")

    return comparison


def generate_markdown_table(comparison: Dict) -> str:
    """Generate a markdown comparison table."""

    md = "# Cross-Model Logit Lens Comparison\n\n"
    md += f"Generated: {comparison['timestamp']}\n\n"

    md += "## Summary Table\n\n"
    md += "| Model | Dataset | Accuracy | MLP % | Attn % | Peak Layer | Boosting MLP % | Suppression Attn % |\n"
    md += "|-------|---------|----------|-------|--------|------------|----------------|--------------------|\n"

    for m in comparison["models"]:
        md += f"| {m['model']} | {m['dataset']} | {m['accuracy']:.1%} | {m['overall_mlp_fraction']:.1%} | {m['overall_attn_fraction']:.1%} | L{m['peak_margin_layer']} | {m.get('boosting_mlp_fraction', 0):.1%} | {m.get('suppression_attn_fraction', 0):.1%} |\n"

    md += "\n## Key Questions\n\n"
    md += "1. **Does MLP/Attention division of labor hold across models?**\n"
    md += "   - Check if boosting MLP % is consistently high (>70%) across models\n"
    md += "   - Check if suppression Attn % is consistently high (>70%) across models\n\n"

    md += "2. **Does the peak layer scale with model depth?**\n"
    md += "   - Compare peak_layer / n_layers ratio across models\n\n"

    md += "3. **Does accuracy correlate with cleaner MLP/Attention separation?**\n"
    md += "   - Models with cleaner separation might be more interpretable\n"

    return md


def main():
    parser = argparse.ArgumentParser(
        description="Run cross-model logit lens experiments")
    parser.add_argument("--models",
                        nargs="+",
                        default=["llava-1.5-7b"],
                        help="Models to test (or 'all' for all models)")
    parser.add_argument("--dataset",
                        type=str,
                        default="vqav2",
                        help="Dataset to use")
    parser.add_argument("--n_samples",
                        type=int,
                        default=1000,
                        help="Number of samples per experiment")
    parser.add_argument("--output_dir",
                        type=str,
                        default="./cross_model_results",
                        help="Output directory")

    args = parser.parse_args()

    # Determine which models to run
    if "all" in args.models:
        model_list = list(MODELS.keys())
    else:
        model_list = args.models

    # Validate models
    for m in model_list:
        if m not in MODELS:
            print(f"Unknown model: {m}")
            print(f"Available models: {list(MODELS.keys())}")
            return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("CROSS-MODEL LOGIT LENS EXPERIMENTS")
    print(f"{'='*70}")
    print(f"Models: {model_list}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.n_samples}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*70}\n")

    # Run experiments
    all_results = []

    for model_name in model_list:
        result = run_single_experiment(
            model_name=model_name,
            dataset_name=args.dataset,
            n_samples=args.n_samples,
            output_dir=args.output_dir,
        )
        all_results.append(result)

    # Compile comparison
    compile_cross_model_results(all_results, args.output_dir)

    print(f"\n{'='*70}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
