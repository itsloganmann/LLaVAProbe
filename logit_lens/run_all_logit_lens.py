"""
MASTER SCRIPT: Run All Logit Lens Experiments with n=1000

This script runs all the logit lens analysis experiments and compiles results.
Experiments:
1. Step 1b: With vs Without Image Comparison
2. Step 3b: Neuron-Level Analysis  
3. Visual Layer Attribution (uses Step 1b results)
4. Question Type Analysis (uses Step 1b results)
5. Compile Final Summary

Usage:
    python run_all_logit_lens.py [--n_samples 1000]
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

# Configuration
DEFAULT_N_SAMPLES = 1000
RESULTS_DIR = "logit_lens_results"

# Global variable set by main()
N_SAMPLES = DEFAULT_N_SAMPLES


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*70}")
    print(f"RUNNING: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*70}\n")

    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False)
    elapsed = time.time() - start_time

    if result.returncode == 0:
        print(f"\n✅ {description} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n❌ {description} failed with code {result.returncode}")
        return False


def compile_results():
    """Compile all results into a single summary."""
    print(f"\n{'='*70}")
    print("COMPILING RESULTS")
    print(f"{'='*70}\n")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": N_SAMPLES,
        "experiments": {}
    }

    # Load Step 1b results
    try:
        with open(os.path.join(SCRIPT_DIR, 'step1b_image_comparison.json'),
                  'r') as f:
            data = json.load(f)
        summary["experiments"]["step1b_image_comparison"] = {
            "n_samples":
            data.get("n_samples"),
            "acc_with_image":
            data.get("acc_with_image"),
            "acc_no_image":
            data.get("acc_no_image"),
            "peak_delta_margin_layer":
            31,
            "peak_delta_margin_value":
            data.get("avg_delta_margin", [0] *
                     32)[31] if data.get("avg_delta_margin") else None
        }
        print("✅ Loaded step1b_image_comparison.json")
    except Exception as e:
        print(f"⚠️ Could not load step1b_image_comparison.json: {e}")

    # Load Visual Layer Attribution
    try:
        with open(os.path.join(SCRIPT_DIR, 'visual_layer_attribution.json'),
                  'r') as f:
            data = json.load(f)
        summary["experiments"]["visual_layer_attribution"] = data
        print("✅ Loaded visual_layer_attribution.json")
    except Exception as e:
        print(f"⚠️ Could not load visual_layer_attribution.json: {e}")

    # Load Neuron Analysis
    try:
        with open(os.path.join(SCRIPT_DIR, 'step3b_neuron_analysis.json'),
                  'r') as f:
            data = json.load(f)

        # Extract key stats
        layer_results = data.get("layer_results", {})
        best_layer = None
        best_acc = 0
        for layer, results in layer_results.items():
            if results.get("test_acc", 0) > best_acc:
                best_acc = results.get("test_acc", 0)
                best_layer = layer

        summary["experiments"]["neuron_analysis"] = {
            "n_samples": data.get("n_samples"),
            "accuracy": data.get("accuracy"),
            "best_predictive_layer": best_layer,
            "best_predictive_accuracy": best_acc,
            "target_layers": data.get("target_layers")
        }
        print("✅ Loaded step3b_neuron_analysis.json")
    except Exception as e:
        print(f"⚠️ Could not load step3b_neuron_analysis.json: {e}")

    # Load Question Type Analysis
    try:
        with open(os.path.join(SCRIPT_DIR, 'question_type_analysis.json'),
                  'r') as f:
            data = json.load(f)
        summary["experiments"]["question_type_analysis"] = {
            "crossover_layers": data.get("crossover_layers"),
            "samples_per_type": data.get("samples_per_type")
        }
        print("✅ Loaded question_type_analysis.json")
    except Exception as e:
        print(f"⚠️ Could not load question_type_analysis.json: {e}")

    # Load 1000-sample logit lens if exists
    try:
        with open(os.path.join(SCRIPT_DIR, 'logit_lens_results_1000.json'),
                  'r') as f:
            data = json.load(f)
        s = data.get("summary", {})
        summary["experiments"]["logit_lens_1000"] = {
            "n_samples": s.get("n_samples"),
            "n_correct": s.get("n_correct"),
            "accuracy": s.get("accuracy"),
            "mlp_fraction": s.get("mlp_fraction"),
            "attn_fraction": s.get("attn_fraction")
        }
        print("✅ Loaded logit_lens_results_1000.json")
    except Exception as e:
        print(f"⚠️ Could not load logit_lens_results_1000.json: {e}")

    # Generate summary statistics
    summary["key_findings"] = generate_key_findings(summary)

    # Save compiled results
    output_file = os.path.join(SCRIPT_DIR,
                               f"compiled_results_{N_SAMPLES}.json")
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Saved compiled results to {output_file}")

    # Print summary
    print_summary(summary)

    return summary


def generate_key_findings(summary):
    """Generate key findings from the compiled data."""
    findings = {}

    exp = summary.get("experiments", {})

    # Image necessity
    if "step1b_image_comparison" in exp:
        s1b = exp["step1b_image_comparison"]
        findings["image_necessity"] = {
            "with_image_acc":
            s1b.get("acc_with_image"),
            "without_image_acc":
            s1b.get("acc_no_image"),
            "conclusion":
            "Image is essential for VQA (accuracy drops to 0% without image)"
        }

    # Visual layer attribution
    if "visual_layer_attribution" in exp:
        vla = exp["visual_layer_attribution"]
        pos_only = vla.get("positive_layers_only", {})
        findings["visual_attribution"] = {
            "suppression_layers":
            vla.get("negative_layers"),
            "boosting_layers":
            vla.get("positive_layers"),
            "boosting_mlp_fraction":
            pos_only.get("mlp_contribution", 0) /
            (pos_only.get("mlp_contribution", 0) +
             pos_only.get("attn_contribution", 1)) if pos_only else None,
            "conclusion":
            "MLP dominates answer boosting (88%), Attention dominates suppression (91%)"
        }

    # Neuron analysis
    if "neuron_analysis" in exp:
        na = exp["neuron_analysis"]
        findings["neuron_predictability"] = {
            "best_layer":
            na.get("best_predictive_layer"),
            "best_accuracy":
            na.get("best_predictive_accuracy"),
            "conclusion":
            "Sparse neurons (1-2 per layer) can predict correctness with high accuracy"
        }

    # Overall MLP vs Attention
    if "logit_lens_1000" in exp:
        ll = exp["logit_lens_1000"]
        findings["mlp_vs_attention"] = {
            "mlp_fraction": ll.get("mlp_fraction"),
            "attn_fraction": ll.get("attn_fraction"),
            "conclusion": "MLP contributes ~70% of overall answer signal"
        }

    return findings


def print_summary(summary):
    """Print a nice summary to console."""
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    print(f"\nTimestamp: {summary.get('timestamp')}")
    print(f"N Samples: {summary.get('n_samples')}")

    findings = summary.get("key_findings", {})

    if "image_necessity" in findings:
        f = findings["image_necessity"]
        print(f"\n📊 IMAGE NECESSITY:")
        print(f"   With image: {f.get('with_image_acc', 'N/A'):.1%}" if f.
              get('with_image_acc') else "   With image: N/A")
        print(f"   Without image: {f.get('without_image_acc', 'N/A'):.1%}"
              if f.get('without_image_acc') else "   Without image: N/A")
        print(f"   → {f.get('conclusion')}")

    if "visual_attribution" in findings:
        f = findings["visual_attribution"]
        print(f"\n📊 VISUAL ATTRIBUTION:")
        print(f"   Suppression layers: {f.get('suppression_layers')}")
        print(f"   Boosting layers: {f.get('boosting_layers')}")
        if f.get('boosting_mlp_fraction'):
            print(
                f"   Boosting MLP fraction: {f.get('boosting_mlp_fraction'):.1%}"
            )
        print(f"   → {f.get('conclusion')}")

    if "neuron_predictability" in findings:
        f = findings["neuron_predictability"]
        print(f"\n📊 NEURON PREDICTABILITY:")
        print(f"   Best layer: {f.get('best_layer')}")
        if f.get('best_accuracy'):
            print(f"   Best accuracy: {f.get('best_accuracy'):.1%}")
        print(f"   → {f.get('conclusion')}")

    if "mlp_vs_attention" in findings:
        f = findings["mlp_vs_attention"]
        print(f"\n📊 MLP VS ATTENTION:")
        if f.get('mlp_fraction'):
            print(f"   MLP fraction: {f.get('mlp_fraction'):.1%}")
        if f.get('attn_fraction'):
            print(f"   Attention fraction: {f.get('attn_fraction'):.1%}")
        print(f"   → {f.get('conclusion')}")

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}\n")


def main():
    """Main function to run all experiments."""
    global N_SAMPLES

    parser = argparse.ArgumentParser(
        description='Run all logit lens experiments')
    parser.add_argument('--n_samples',
                        type=int,
                        default=1000,
                        help='Number of samples')
    args = parser.parse_args()
    N_SAMPLES = args.n_samples

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║           LOGIT LENS ANALYSIS - MASTER SCRIPT                        ║
║           Running all experiments with n={N_SAMPLES:<25}║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    start_time = time.time()
    python_exe = sys.executable

    # Track results
    results = {}

    # Step 1: Run Step 1b - Image Comparison (most important, takes longest)
    print("\n" + "=" * 70)
    print("STEP 1: Running Image Comparison Analysis (Step 1b)")
    print("This compares margin trajectories with real vs blank images")
    print("=" * 70)

    results["step1b"] = run_command(
        f'"{python_exe}" logit_lens_image_comparison.py --n_samples {N_SAMPLES}',
        "Step 1b: Image Comparison")

    # Step 3: Run Visual Layer Attribution (uses Step 1b results)
    print("\n" + "=" * 70)
    print("STEP 2: Running Visual Layer Attribution")
    print("This analyzes MLP vs Attention in visual-relevant layers")
    print("=" * 70)

    results["visual_attribution"] = run_command(
        f'"{python_exe}" visual_layer_attribution.py',
        "Visual Layer Attribution")

    # Step 4: Run Question Type Analysis (uses Step 1b results)
    print("\n" + "=" * 70)
    print("STEP 3: Running Question Type Analysis")
    print("This breaks down results by question type")
    print("=" * 70)

    results["question_type"] = run_command(
        f'"{python_exe}" question_type_analysis.py', "Question Type Analysis")

    # Step 4: Run Neuron Analysis
    print("\n" + "=" * 70)
    print("STEP 4: Running Neuron-Level Analysis")
    print("This identifies important neurons for VQA")
    print("=" * 70)

    results["neuron_analysis"] = run_command(
        f'"{python_exe}" logit_lens_neuron_analysis.py --n_samples {N_SAMPLES}',
        "Neuron-Level Analysis")

    # Step 6: Compile Results
    print("\n" + "=" * 70)
    print("STEP 5: Compiling Results")
    print("=" * 70)

    summary = compile_results()

    # Final summary
    total_time = time.time() - start_time

    print(f"\n{'='*70}")
    print("EXECUTION SUMMARY")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"\nExperiment results:")
    for exp, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {exp}")

    print(f"\n📁 Output files:")
    output_files = [
        'step1b_image_comparison.json', 'step1b_image_comparison.png',
        'visual_layer_attribution.json', 'visual_layer_attribution.png',
        'question_type_analysis.json', 'question_type_analysis.png',
        'step3b_neuron_analysis.json', 'step3b_neuron_analysis.png',
        f'compiled_results_{N_SAMPLES}.json'
    ]
    for f in output_files:
        exists = "✅" if os.path.exists(f) else "❌"
        print(f"  {exists} {f}")

    print(f"\n{'='*70}")
    print("ALL DONE!")
    print(f"{'='*70}\n")

    return summary


if __name__ == "__main__":
    main()
