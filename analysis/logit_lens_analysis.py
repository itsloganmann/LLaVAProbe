"""
Visualization and analysis of logit lens results.

This script analyzes where in the language model the answer decision is made,
and compares the contributions of MLP vs attention components.
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def load_results(path: str = "logit_lens_results.json") -> Dict:
    """Load logit lens results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def plot_margin_trajectory(summary: Dict, save_path: Optional[str] = None):
    """
    Plot the margin trajectory across layers.
    Shows where in the model the answer decision is made.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Average margin across layers
    ax1 = axes[0]
    margins = np.array(summary['avg_layer_margins'])
    std = np.array(summary['std_layer_margins'])
    layers = np.arange(len(margins))

    ax1.plot(layers, margins, 'b-', linewidth=2, label='Average margin')
    ax1.fill_between(layers, margins - std, margins + std, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # Find key transition points
    margin_changes = np.diff(margins)
    peak_layer = np.argmax(margin_changes) + 1
    ax1.axvline(x=peak_layer,
                color='r',
                linestyle='--',
                alpha=0.7,
                label=f'Peak change: Layer {peak_layer}')

    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Margin (logit_correct - max_other)')
    ax1.set_title('Answer Margin Trajectory Across Layers\n(Logit Lens)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Correct vs Incorrect trajectories
    ax2 = axes[1]
    if 'correct_avg_margins' in summary and 'incorrect_avg_margins' in summary:
        correct_margins = np.array(summary['correct_avg_margins'])
        incorrect_margins = np.array(summary['incorrect_avg_margins'])

        ax2.plot(layers,
                 correct_margins,
                 'g-',
                 linewidth=2,
                 label='Correct answers')
        ax2.plot(layers,
                 incorrect_margins,
                 'r-',
                 linewidth=2,
                 label='Incorrect answers')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Margin')
        ax2.set_title('Margin Trajectory: Correct vs Incorrect')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def plot_component_contributions(summary: Dict,
                                 save_path: Optional[str] = None):
    """
    Plot MLP vs Attention contributions per layer.
    Tests whether MLP contributes more to answer logits.
    """
    if 'avg_attn_contributions' not in summary:
        print("No component decomposition data available.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    attn = np.array(summary['avg_attn_contributions'])
    mlp = np.array(summary['avg_mlp_contributions'])
    layers = np.arange(len(attn))

    # Plot 1: Stacked contributions
    ax1 = axes[0]

    # Separate positive and negative
    attn_pos = np.maximum(attn, 0)
    attn_neg = np.minimum(attn, 0)
    mlp_pos = np.maximum(mlp, 0)
    mlp_neg = np.minimum(mlp, 0)

    width = 0.8
    ax1.bar(layers,
            attn_pos,
            width,
            label='Attention (+)',
            color='blue',
            alpha=0.7)
    ax1.bar(layers,
            mlp_pos,
            width,
            bottom=attn_pos,
            label='MLP (+)',
            color='orange',
            alpha=0.7)
    ax1.bar(layers,
            attn_neg,
            width,
            label='Attention (-)',
            color='lightblue',
            alpha=0.7)
    ax1.bar(layers,
            mlp_neg,
            width,
            bottom=attn_neg,
            label='MLP (-)',
            color='lightyellow',
            alpha=0.7)

    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Contribution to Answer Logit')
    ax1.set_title('MLP vs Attention Contributions per Layer')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cumulative contributions
    ax2 = axes[1]

    cum_attn = np.cumsum(attn)
    cum_mlp = np.cumsum(mlp)

    ax2.plot(layers, cum_attn, 'b-', linewidth=2, label='Cumulative Attention')
    ax2.plot(layers, cum_mlp, 'orange', linewidth=2, label='Cumulative MLP')
    ax2.plot(layers, cum_attn + cum_mlp, 'g--', linewidth=2, label='Total')

    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Cumulative Contribution')
    ax2.set_title('Cumulative Contributions to Answer Logit')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Pie chart of total contributions
    ax3 = axes[2]

    total_attn = max(0, summary['total_attn_contribution'])
    total_mlp = max(0, summary['total_mlp_contribution'])

    if total_attn + total_mlp > 0:
        sizes = [total_attn, total_mlp]
        labels = [f'Attention\n({total_attn:.1f})', f'MLP\n({total_mlp:.1f})']
        colors = ['blue', 'orange']
        explode = (0, 0.05)  # Explode MLP slice

        ax3.pie(sizes,
                explode=explode,
                labels=labels,
                colors=colors,
                autopct='%1.1f%%',
                shadow=True,
                startangle=90)
        ax3.set_title(
            f'Total Contribution to Answer Logit\nMLP Fraction: {summary["mlp_fraction"]*100:.1f}%'
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def plot_correct_vs_incorrect_mlp(summary: Dict,
                                  save_path: Optional[str] = None):
    """
    Compare MLP contributions for correct vs incorrect predictions.
    """
    if 'correct_avg_mlp' not in summary:
        print("No correct/incorrect MLP comparison data available.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    correct_mlp = np.array(summary['correct_avg_mlp'])
    incorrect_mlp = np.array(summary['incorrect_avg_mlp'])
    layers = np.arange(len(correct_mlp))

    ax.plot(layers,
            correct_mlp,
            'g-',
            linewidth=2,
            label='Correct predictions')
    ax.plot(layers,
            incorrect_mlp,
            'r-',
            linewidth=2,
            label='Incorrect predictions')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # Highlight layers where correct >> incorrect
    diff = correct_mlp - incorrect_mlp
    significant_layers = np.where(diff > np.std(diff))[0]
    for layer in significant_layers:
        ax.axvspan(layer - 0.5, layer + 0.5, alpha=0.2, color='green')

    ax.set_xlabel('Layer Index')
    ax.set_ylabel('MLP Contribution to Answer Logit')
    ax.set_title(
        'MLP Contributions: Correct vs Incorrect Predictions\n(Green bands = layers where correct > incorrect)'
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def compute_mlp_reliability_correlation(results_data: Dict) -> Dict:
    """
    Compute correlation between MLP contributions and correctness.
    This tests whether MLP-derived scores predict reliability better than attention.
    """
    individual = results_data['individual_results']

    if not individual or individual[0]['mlp_contributions'] is None:
        return {}

    # Extract data
    correctness = np.array([1 if r['is_correct'] else 0 for r in individual])

    # Total MLP contribution per sample
    total_mlp = np.array([np.sum(r['mlp_contributions']) for r in individual])

    # Total attention contribution per sample
    total_attn = np.array(
        [np.sum(r['attn_contributions']) for r in individual])

    # Per-layer MLP correlations
    n_layers = len(individual[0]['mlp_contributions'])
    layer_mlp_corrs = []

    for layer in range(n_layers):
        layer_mlp = np.array(
            [r['mlp_contributions'][layer] for r in individual])
        if np.std(layer_mlp) > 0:
            corr, p = stats.pearsonr(layer_mlp, correctness)
            layer_mlp_corrs.append({
                'layer': layer,
                'corr': corr,
                'p': p,
                'r2': corr**2
            })

    # Summary statistics
    results = {
        'total_mlp_vs_correctness': {
            'corr': float(stats.pearsonr(total_mlp, correctness)[0]),
            'r2': float(stats.pearsonr(total_mlp, correctness)[0]**2),
            'p': float(stats.pearsonr(total_mlp, correctness)[1]),
        },
        'total_attn_vs_correctness': {
            'corr': float(stats.pearsonr(total_attn, correctness)[0]),
            'r2': float(stats.pearsonr(total_attn, correctness)[0]**2),
            'p': float(stats.pearsonr(total_attn, correctness)[1]),
        },
        'layer_mlp_correlations':
        layer_mlp_corrs,
        'best_mlp_layer':
        max(layer_mlp_corrs, key=lambda x: abs(x['corr']))
        if layer_mlp_corrs else None,
    }

    return results


def print_analysis_report(results_data: Dict):
    """Print a comprehensive analysis report."""
    summary = results_data['summary']

    print("=" * 70)
    print("LOGIT LENS ANALYSIS REPORT")
    print("=" * 70)

    print(f"\nDataset: {summary['n_samples']} samples")
    print(f"Accuracy: {summary['accuracy']*100:.1f}%")

    # Layer analysis
    print("\n" + "-" * 70)
    print("LAYER ANALYSIS: Where is the answer decided?")
    print("-" * 70)

    margins = np.array(summary['avg_layer_margins'])
    margin_changes = np.diff(margins)

    # Find key layers
    peak_layer = np.argmax(margin_changes) + 1
    early_margin = margins[:8].mean()
    middle_margin = margins[8:24].mean()
    late_margin = margins[24:].mean()

    print(f"\nMargin by layer phase:")
    print(f"  Early layers (0-7):   {early_margin:.3f}")
    print(f"  Middle layers (8-23): {middle_margin:.3f}")
    print(f"  Late layers (24-31):  {late_margin:.3f}")
    print(f"\nPeak margin increase at: Layer {peak_layer}")

    # Component analysis
    if 'mlp_fraction' in summary:
        print("\n" + "-" * 70)
        print("COMPONENT ANALYSIS: MLP vs Attention")
        print("-" * 70)

        print(f"\nTotal contributions to answer logit:")
        print(f"  Attention: {summary['total_attn_contribution']:.2f}")
        print(f"  MLP:       {summary['total_mlp_contribution']:.2f}")
        print(
            f"\n  >>> MLP contributes {summary['mlp_fraction']*100:.1f}% of the answer logit <<<"
        )

        # Find peak MLP layer
        mlp_contribs = np.array(summary['avg_mlp_contributions'])
        peak_mlp_layer = np.argmax(mlp_contribs)
        print(f"\nPeak MLP contribution at: Layer {peak_mlp_layer}")

    # Correct vs incorrect
    if 'correct_avg_margins' in summary:
        print("\n" + "-" * 70)
        print("CORRECT vs INCORRECT COMPARISON")
        print("-" * 70)

        correct_margins = np.array(summary['correct_avg_margins'])
        incorrect_margins = np.array(summary['incorrect_avg_margins'])

        print(f"\nFinal layer margin:")
        print(f"  Correct predictions:   {correct_margins[-1]:.3f}")
        print(f"  Incorrect predictions: {incorrect_margins[-1]:.3f}")

        # Where do they diverge?
        diff = correct_margins - incorrect_margins
        divergence_layer = np.argmax(diff > 0.1 * diff.max())
        print(f"\nTrajectories diverge at: Layer {divergence_layer}")

    # Reliability correlation
    reliability = compute_mlp_reliability_correlation(results_data)
    if reliability:
        print("\n" + "-" * 70)
        print("RELIABILITY CORRELATION: Does MLP predict correctness?")
        print("-" * 70)

        print(f"\nTotal MLP contribution vs correctness:")
        print(f"  R² = {reliability['total_mlp_vs_correctness']['r2']:.4f}")
        print(f"  p  = {reliability['total_mlp_vs_correctness']['p']:.4f}")

        print(f"\nTotal Attention contribution vs correctness:")
        print(f"  R² = {reliability['total_attn_vs_correctness']['r2']:.4f}")
        print(f"  p  = {reliability['total_attn_vs_correctness']['p']:.4f}")

        if reliability['best_mlp_layer']:
            best = reliability['best_mlp_layer']
            print(f"\nBest single MLP layer for predicting correctness:")
            print(
                f"  Layer {best['layer']}: R² = {best['r2']:.4f}, p = {best['p']:.4f}"
            )

    # Conclusions
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)

    if 'mlp_fraction' in summary:
        mlp_frac = summary['mlp_fraction']
        if mlp_frac > 0.6:
            print(
                f"\n✓ MLP dominates answer generation ({mlp_frac*100:.0f}% of logit)"
            )
            print("  This supports the hypothesis that language model MLPs,")
            print("  not visual attention, drive VQA answer decisions.")
        elif mlp_frac > 0.4:
            print(
                f"\n≈ MLP and Attention contribute roughly equally ({mlp_frac*100:.0f}% MLP)"
            )
        else:
            print(
                f"\n✗ Attention dominates ({(1-mlp_frac)*100:.0f}% of logit)")

    print()


def main():
    """Main analysis routine."""
    import sys

    # Try to load results
    results_path = sys.argv[1] if len(
        sys.argv) > 1 else "logit_lens_results.json"

    try:
        results = load_results(results_path)
    except FileNotFoundError:
        print(f"Results file not found: {results_path}")
        print("Run logit_lens.py first to generate results.")
        return

    # Print report
    print_analysis_report(results)

    # Generate plots
    print("\nGenerating visualizations...")

    plot_margin_trajectory(results['summary'],
                           save_path="logit_lens_margins.png")
    plot_component_contributions(results['summary'],
                                 save_path="logit_lens_components.png")

    if 'correct_avg_mlp' in results['summary']:
        plot_correct_vs_incorrect_mlp(
            results['summary'],
            save_path="logit_lens_correct_vs_incorrect.png")


if __name__ == "__main__":
    main()
