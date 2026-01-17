"""
Step 1b: With vs Without Image Comparison
==========================================

Compare margin trajectory with real image vs blank image to see
which layers actually use visual information.
"""

import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import json
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

warnings.filterwarnings("ignore")

from analysis.logit_lens import LogitLensAnalyzer


def run_image_comparison(n_samples: int = 100):
    """
    Compare model behavior with real images vs blank images.
    
    This tells us which layers actually use visual information.
    """
    print("=" * 70)
    print("STEP 1b: WITH vs WITHOUT IMAGE COMPARISON")
    print("=" * 70)

    # Load analyzer
    print("\nLoading model...")
    analyzer = LogitLensAnalyzer(quantization="none")

    # Load samples
    print("\nLoading samples...")
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(
            os.path.join(parent_dir,
                         'test_intervention_output/analysis_records.json'),
            'r') as f:
        data = json.load(f)

    records = data.get('records', [])[:n_samples]
    print(f"Using {len(records)} samples")

    # Create blank image
    blank_image = Image.new('RGB', (336, 336), color=(128, 128, 128))  # Gray

    results_with_image = []
    results_no_image = []

    for i, record in enumerate(records):
        if i % 10 == 0:
            print(f"Processing sample {i+1}/{len(records)}...")

        try:
            # Load real image
            response = requests.get(record['image_url'], timeout=10)
            real_image = Image.open(BytesIO(response.content)).convert('RGB')

            # Analyze with real image
            result_real = analyzer.analyze_sample(
                image=real_image,
                question=record['question'],
                ground_truth=record['ground_truth'],
                decompose_components=True,
                question_type=record.get('question_type', 'unknown'))

            # Analyze with blank image
            result_blank = analyzer.analyze_sample(
                image=blank_image,
                question=record['question'],
                ground_truth=record['ground_truth'],
                decompose_components=True,
                question_type=record.get('question_type', 'unknown'))

            results_with_image.append({
                'question':
                record['question'],
                'ground_truth':
                record['ground_truth'],
                'predicted':
                result_real.predicted_token,
                'is_correct':
                result_real.is_correct,
                'margins':
                result_real.layer_margins.tolist(),
                'mlp':
                result_real.mlp_contributions.tolist()
                if result_real.mlp_contributions is not None else None,
                'attn':
                result_real.attn_contributions.tolist()
                if result_real.attn_contributions is not None else None,
            })

            results_no_image.append({
                'predicted':
                result_blank.predicted_token,
                'is_correct':
                result_blank.is_correct,
                'margins':
                result_blank.layer_margins.tolist(),
                'mlp':
                result_blank.mlp_contributions.tolist()
                if result_blank.mlp_contributions is not None else None,
                'attn':
                result_blank.attn_contributions.tolist()
                if result_blank.attn_contributions is not None else None,
            })

        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue

    print(f"\nSuccessfully processed {len(results_with_image)} samples")

    # Compute statistics
    n_layers = 32

    # Accuracy comparison
    acc_with_image = sum(r['is_correct']
                         for r in results_with_image) / len(results_with_image)
    acc_no_image = sum(r['is_correct']
                       for r in results_no_image) / len(results_no_image)

    print(f"\n📊 ACCURACY COMPARISON:")
    print(f"   With real image: {acc_with_image*100:.1f}%")
    print(f"   With blank image: {acc_no_image*100:.1f}%")
    print(f"   Difference: {(acc_with_image - acc_no_image)*100:+.1f}%")

    # Delta margins (image - no image)
    margins_with = np.array([r['margins'] for r in results_with_image])
    margins_no = np.array([r['margins'] for r in results_no_image])
    delta_margins = margins_with - margins_no

    avg_delta_margin = np.mean(delta_margins, axis=0)
    std_delta_margin = np.std(delta_margins, axis=0)

    print(f"\n📊 DELTA MARGIN (Image - No Image):")
    peak_layer = np.argmax(np.abs(avg_delta_margin))
    print(
        f"   Peak visual effect at Layer {peak_layer}: Δmargin = {avg_delta_margin[peak_layer]:.3f}"
    )

    # Which layers show significant difference?
    significant_layers = []
    for layer in range(n_layers):
        t_stat, p_val = stats.ttest_1samp(delta_margins[:, layer], 0)
        if p_val < 0.05:
            significant_layers.append(layer)
    print(f"   Significant layers (p<0.05): {significant_layers}")

    # Delta MLP vs Attention contributions
    mlp_with = np.array([r['mlp'] for r in results_with_image if r['mlp']])
    mlp_no = np.array([r['mlp'] for r in results_no_image if r['mlp']])
    attn_with = np.array([r['attn'] for r in results_with_image if r['attn']])
    attn_no = np.array([r['attn'] for r in results_no_image if r['attn']])

    if len(mlp_with) > 0 and len(mlp_no) > 0:
        delta_mlp = mlp_with - mlp_no
        delta_attn = attn_with - attn_no

        avg_delta_mlp = np.mean(delta_mlp, axis=0)
        avg_delta_attn = np.mean(delta_attn, axis=0)

        total_delta_mlp = np.sum(avg_delta_mlp)
        total_delta_attn = np.sum(avg_delta_attn)

        print(f"\n📊 VISUAL CONTRIBUTION BREAKDOWN:")
        print(f"   MLP contribution from image: {total_delta_mlp:.3f}")
        print(f"   Attention contribution from image: {total_delta_attn:.3f}")

        if abs(total_delta_mlp) + abs(total_delta_attn) > 0:
            mlp_frac = abs(total_delta_mlp) / (abs(total_delta_mlp) +
                                               abs(total_delta_attn))
            print(
                f"   MLP fraction of visual contribution: {mlp_frac*100:.1f}%")

        # Layer-by-layer visual contribution
        print(f"\n   Top 5 layers by MLP visual contribution:")
        mlp_sorted_idx = np.argsort(np.abs(avg_delta_mlp))[::-1][:5]
        for idx in mlp_sorted_idx:
            print(
                f"   Layer {idx}: ΔMLP={avg_delta_mlp[idx]:+.3f}, ΔAttn={avg_delta_attn[idx]:+.3f}"
            )

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Margin trajectory comparison
    ax1 = axes[0, 0]
    avg_margin_with = np.mean(margins_with, axis=0)
    avg_margin_no = np.mean(margins_no, axis=0)
    ax1.plot(range(n_layers),
             avg_margin_with,
             'b-',
             linewidth=2,
             label='With Image')
    ax1.plot(range(n_layers),
             avg_margin_no,
             'r--',
             linewidth=2,
             label='Blank Image')
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Margin')
    ax1.set_title('A) Margin Trajectory: With vs Without Image')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Delta margin
    ax2 = axes[0, 1]
    ax2.bar(range(n_layers),
            avg_delta_margin,
            color=['green' if d > 0 else 'red' for d in avg_delta_margin],
            alpha=0.7)
    ax2.axhline(y=0, color='k', linestyle='-')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Δ Margin (Image - No Image)')
    ax2.set_title('B) Visual Information Effect per Layer')
    ax2.grid(True, alpha=0.3)

    # 3. Delta MLP vs Attention
    if len(mlp_with) > 0:
        ax3 = axes[1, 0]
        x = np.arange(n_layers)
        width = 0.35
        ax3.bar(x - width / 2,
                avg_delta_mlp,
                width,
                label='ΔMLP',
                color='red',
                alpha=0.7)
        ax3.bar(x + width / 2,
                avg_delta_attn,
                width,
                label='ΔAttn',
                color='blue',
                alpha=0.7)
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Δ Contribution (Image - No Image)')
        ax3.set_title('C) Visual Info: MLP vs Attention per Layer')
        ax3.legend()

    # 4. Summary
    ax4 = axes[1, 1]
    if len(mlp_with) > 0:
        # Cumulative delta
        cum_delta_mlp = np.cumsum(avg_delta_mlp)
        cum_delta_attn = np.cumsum(avg_delta_attn)
        ax4.plot(range(n_layers),
                 cum_delta_mlp,
                 'r-',
                 linewidth=2,
                 label=f'MLP (total: {total_delta_mlp:.2f})')
        ax4.plot(range(n_layers),
                 cum_delta_attn,
                 'b-',
                 linewidth=2,
                 label=f'Attn (total: {total_delta_attn:.2f})')
        ax4.set_xlabel('Layer')
        ax4.set_ylabel('Cumulative Δ Contribution')
        ax4.set_title('D) Cumulative Visual Contribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'step1b_image_comparison.png'),
                dpi=150,
                bbox_inches='tight')
    print("\nSaved: step1b_image_comparison.png")

    # Save results
    output = {
        'n_samples': len(results_with_image),
        'acc_with_image': acc_with_image,
        'acc_no_image': acc_no_image,
        'avg_delta_margin': avg_delta_margin.tolist(),
        'avg_delta_mlp': avg_delta_mlp.tolist() if len(mlp_with) > 0 else None,
        'avg_delta_attn':
        avg_delta_attn.tolist() if len(attn_with) > 0 else None,
        'results_with_image': results_with_image,
        'results_no_image': results_no_image,
    }

    with open(os.path.join(SCRIPT_DIR, 'step1b_image_comparison.json'),
              'w') as f:
        json.dump(output, f, indent=2)
    print("Saved: step1b_image_comparison.json")

    print("\n" + "=" * 70)
    print("STEP 1b COMPLETE")
    print("=" * 70)

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples',
                        type=int,
                        default=100,
                        help='Number of samples')
    args = parser.parse_args()
    results = run_image_comparison(n_samples=args.n_samples)
