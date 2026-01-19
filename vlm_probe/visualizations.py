"""
Visualization utilities for VLM attention probing.

Creates publication-quality figures for attention analysis,
cross-model comparison, and novel insights visualization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image


# Custom colormap for attention visualization
ATTENTION_CMAP = LinearSegmentedColormap.from_list(
    "attention",
    ["#000033", "#0000FF", "#00FFFF", "#FFFF00", "#FF0000", "#FFFFFF"]
)


def plot_attention_heatmap(
    image: Image.Image,
    attention_map: np.ndarray,
    title: str = "Attention Map",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    alpha: float = 0.6,
) -> plt.Figure:
    """
    Plot attention heatmap overlaid on image.
    
    Args:
        image: Original PIL Image.
        attention_map: 2D attention array.
        title: Plot title.
        save_path: Path to save figure.
        figsize: Figure size.
        alpha: Overlay transparency.
        
    Returns:
        Matplotlib figure.
    """
    import cv2
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Resize attention to image size
    attn_resized = cv2.resize(
        attention_map.astype(np.float32),
        (w, h),
        interpolation=cv2.INTER_CUBIC
    )
    
    # Normalize
    attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
    
    # Original image
    axes[0].imshow(img_array)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Attention map alone
    im = axes[1].imshow(attention_map, cmap=ATTENTION_CMAP)
    axes[1].set_title("Attention Pattern")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(img_array)
    axes[2].imshow(attn_resized, cmap=ATTENTION_CMAP, alpha=alpha)
    axes[2].set_title("Attention Overlay")
    axes[2].axis("off")
    
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def plot_layer_attribution(
    layer_contributions: Dict[int, float],
    model_name: str = "Model",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot layer-wise contribution analysis.
    
    Args:
        layer_contributions: Dict mapping layer index to contribution score.
        model_name: Name for title.
        save_path: Path to save figure.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    layers = sorted(layer_contributions.keys())
    contributions = [layer_contributions[l] for l in layers]
    
    # Bar chart
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    axes[0].bar(layers, contributions, color=colors)
    axes[0].set_xlabel("Layer", fontsize=12)
    axes[0].set_ylabel("Contribution Score", fontsize=12)
    axes[0].set_title(f"{model_name} Layer Contributions", fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative plot
    cumulative = np.cumsum(contributions) / np.sum(contributions)
    axes[1].plot(layers, cumulative, 'b-o', linewidth=2, markersize=6)
    axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='50% threshold')
    axes[1].axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='90% threshold')
    axes[1].set_xlabel("Layer", fontsize=12)
    axes[1].set_ylabel("Cumulative Contribution", fontsize=12)
    axes[1].set_title(f"{model_name} Cumulative Attribution", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def plot_confidence_calibration(
    confidences: List[float],
    accuracies: List[bool],
    model_name: str = "Model",
    num_bins: int = 10,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot confidence calibration diagram.
    
    Args:
        confidences: List of model confidence scores.
        accuracies: List of whether predictions were correct.
        model_name: Name for title.
        num_bins: Number of calibration bins.
        save_path: Path to save figure.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    confidences = np.array(confidences)
    accuracies = np.array(accuracies, dtype=float)
    
    # Bin edges
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Compute per-bin accuracy and average confidence
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for i in range(num_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accs.append(accuracies[mask].mean())
            bin_confs.append(confidences[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accs.append(0)
            bin_confs.append(bin_centers[i])
            bin_counts.append(0)
    
    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)
    bin_counts = np.array(bin_counts)
    
    # Reliability diagram
    ax = axes[0, 0]
    ax.bar(bin_centers, bin_accs, width=1/num_bins * 0.8, alpha=0.7, label='Accuracy')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect calibration')
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"{model_name} Reliability Diagram", fontsize=14)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Confidence histogram
    ax = axes[0, 1]
    ax.hist(confidences, bins=num_bins, alpha=0.7, edgecolor='black')
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Confidence Distribution", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # ECE computation
    ece = np.sum(bin_counts * np.abs(bin_accs - bin_confs)) / np.sum(bin_counts)
    
    ax = axes[1, 0]
    gaps = bin_accs - bin_confs
    colors = ['green' if g >= 0 else 'red' for g in gaps]
    ax.bar(bin_centers, np.abs(gaps), width=1/num_bins * 0.8, color=colors, alpha=0.7)
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("|Accuracy - Confidence|", fontsize=12)
    ax.set_title(f"Calibration Gap (ECE = {ece:.4f})", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
    Calibration Summary
    ───────────────────
    
    Expected Calibration Error (ECE): {ece:.4f}
    
    Mean Confidence: {np.mean(confidences):.4f}
    Mean Accuracy: {np.mean(accuracies):.4f}
    
    Overconfidence Rate: {np.mean(confidences > accuracies):.2%}
    Underconfidence Rate: {np.mean(confidences < accuracies):.2%}
    
    Total Samples: {len(confidences)}
    """
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle(f"{model_name} Confidence Calibration Analysis", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def plot_cross_model_comparison(
    image: Image.Image,
    qwen_attention: np.ndarray,
    paligemma_attention: np.ndarray,
    qwen_answer: str,
    paligemma_answer: str,
    question: str,
    ground_truth: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
) -> plt.Figure:
    """
    Create comprehensive cross-model comparison visualization.
    
    Args:
        image: Original image.
        qwen_attention: Qwen attention map.
        paligemma_attention: PaliGemma attention map.
        qwen_answer: Qwen's answer.
        paligemma_answer: PaliGemma's answer.
        question: The question asked.
        ground_truth: Optional ground truth answer.
        save_path: Path to save figure.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    import cv2
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 0.3])
    
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Row 1: Original image and attention maps
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img_array)
    ax_img.set_title("Original Image", fontsize=12)
    ax_img.axis("off")
    
    ax_qwen = fig.add_subplot(gs[0, 1])
    im1 = ax_qwen.imshow(qwen_attention, cmap=ATTENTION_CMAP)
    ax_qwen.set_title("Qwen Attention", fontsize=12)
    ax_qwen.axis("off")
    plt.colorbar(im1, ax=ax_qwen, fraction=0.046, pad=0.04)
    
    ax_pali = fig.add_subplot(gs[0, 2])
    im2 = ax_pali.imshow(paligemma_attention, cmap=ATTENTION_CMAP)
    ax_pali.set_title("PaliGemma Attention", fontsize=12)
    ax_pali.axis("off")
    plt.colorbar(im2, ax=ax_pali, fraction=0.046, pad=0.04)
    
    # Attention difference
    # Resize to same size for comparison
    from scipy.ndimage import zoom
    target_size = max(qwen_attention.shape[0], paligemma_attention.shape[0])
    q_resized = zoom(qwen_attention, (target_size / qwen_attention.shape[0], 
                                       target_size / qwen_attention.shape[1]))
    p_resized = zoom(paligemma_attention, (target_size / paligemma_attention.shape[0], 
                                            target_size / paligemma_attention.shape[1]))
    diff = q_resized - p_resized
    
    ax_diff = fig.add_subplot(gs[0, 3])
    im3 = ax_diff.imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax_diff.set_title("Difference (Qwen - PaliGemma)", fontsize=12)
    ax_diff.axis("off")
    plt.colorbar(im3, ax=ax_diff, fraction=0.046, pad=0.04)
    
    # Row 2: Overlays
    qwen_overlay = cv2.resize(qwen_attention.astype(np.float32), (w, h), 
                               interpolation=cv2.INTER_CUBIC)
    pali_overlay = cv2.resize(paligemma_attention.astype(np.float32), (w, h),
                               interpolation=cv2.INTER_CUBIC)
    
    ax_qwen_ov = fig.add_subplot(gs[1, 0:2])
    ax_qwen_ov.imshow(img_array)
    ax_qwen_ov.imshow(qwen_overlay, cmap=ATTENTION_CMAP, alpha=0.5)
    ax_qwen_ov.set_title(f"Qwen Overlay → \"{qwen_answer}\"", fontsize=12)
    ax_qwen_ov.axis("off")
    
    ax_pali_ov = fig.add_subplot(gs[1, 2:4])
    ax_pali_ov.imshow(img_array)
    ax_pali_ov.imshow(pali_overlay, cmap=ATTENTION_CMAP, alpha=0.5)
    ax_pali_ov.set_title(f"PaliGemma Overlay → \"{paligemma_answer}\"", fontsize=12)
    ax_pali_ov.axis("off")
    
    # Row 3: Question and answers
    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis("off")
    
    agreement = "✓ AGREE" if qwen_answer.lower().strip() == paligemma_answer.lower().strip() else "✗ DISAGREE"
    gt_text = f"\nGround Truth: {ground_truth}" if ground_truth else ""
    
    # Correlation
    q_flat = q_resized.flatten()
    p_flat = p_resized.flatten()
    if np.std(q_flat) > 0 and np.std(p_flat) > 0:
        corr = np.corrcoef(q_flat, p_flat)[0, 1]
    else:
        corr = 0
    
    info_text = f"""Question: {question}
Qwen Answer: "{qwen_answer}"  |  PaliGemma Answer: "{paligemma_answer}"  |  {agreement}{gt_text}
Attention Correlation: {corr:.3f}"""
    
    ax_text.text(0.5, 0.5, info_text, transform=ax_text.transAxes, fontsize=11,
                 ha='center', va='center', 
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle("Cross-Model Visual Attention Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def plot_attention_entropy_analysis(
    qwen_entropies: List[float],
    paligemma_entropies: List[float],
    qwen_layer_entropies: Dict[int, List[float]],
    paligemma_layer_entropies: Dict[int, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Plot attention entropy analysis comparing models.
    
    Args:
        qwen_entropies: List of overall entropy values for Qwen.
        paligemma_entropies: List of overall entropy values for PaliGemma.
        qwen_layer_entropies: Per-layer entropy distributions for Qwen.
        paligemma_layer_entropies: Per-layer entropy distributions for PaliGemma.
        save_path: Path to save figure.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Overall entropy comparison
    ax = axes[0, 0]
    data = [qwen_entropies, paligemma_entropies]
    bp = ax.boxplot(data, labels=['Qwen', 'PaliGemma'], patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('coral')
    ax.set_ylabel("Attention Entropy", fontsize=12)
    ax.set_title("Overall Attention Entropy Distribution", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Statistical test
    from scipy import stats
    if len(qwen_entropies) > 2 and len(paligemma_entropies) > 2:
        t_stat, p_val = stats.ttest_ind(qwen_entropies, paligemma_entropies)
        ax.text(0.95, 0.95, f"t-test p={p_val:.4f}", transform=ax.transAxes,
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Entropy histograms
    ax = axes[0, 1]
    ax.hist(qwen_entropies, bins=20, alpha=0.6, label='Qwen', color='steelblue')
    ax.hist(paligemma_entropies, bins=20, alpha=0.6, label='PaliGemma', color='coral')
    ax.set_xlabel("Attention Entropy", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Entropy Distribution Comparison", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Layer-wise entropy for Qwen
    ax = axes[1, 0]
    if qwen_layer_entropies:
        layers = sorted(qwen_layer_entropies.keys())
        means = [np.mean(qwen_layer_entropies[l]) for l in layers]
        stds = [np.std(qwen_layer_entropies[l]) for l in layers]
        ax.errorbar(layers, means, yerr=stds, fmt='o-', capsize=3, 
                    color='steelblue', label='Qwen')
    if paligemma_layer_entropies:
        layers = sorted(paligemma_layer_entropies.keys())
        means = [np.mean(paligemma_layer_entropies[l]) for l in layers]
        stds = [np.std(paligemma_layer_entropies[l]) for l in layers]
        ax.errorbar(layers, means, yerr=stds, fmt='s-', capsize=3,
                    color='coral', label='PaliGemma')
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Mean Entropy", fontsize=12)
    ax.set_title("Layer-wise Attention Entropy", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    qwen_mean = np.mean(qwen_entropies) if qwen_entropies else 0
    qwen_std = np.std(qwen_entropies) if qwen_entropies else 0
    pali_mean = np.mean(paligemma_entropies) if paligemma_entropies else 0
    pali_std = np.std(paligemma_entropies) if paligemma_entropies else 0
    
    summary = f"""
    Attention Entropy Analysis Summary
    ══════════════════════════════════
    
    Qwen-VL:
      Mean Entropy: {qwen_mean:.4f} ± {qwen_std:.4f}
      Sample Size: {len(qwen_entropies)}
    
    PaliGemma:
      Mean Entropy: {pali_mean:.4f} ± {pali_std:.4f}
      Sample Size: {len(paligemma_entropies)}
    
    Difference: {qwen_mean - pali_mean:+.4f}
    
    Interpretation:
    {"Qwen shows MORE distributed attention" if qwen_mean > pali_mean else "PaliGemma shows MORE distributed attention"}
    (higher entropy = more spread out attention)
    """
    
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle("Cross-Model Attention Entropy Analysis", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def plot_logit_lens_comparison(
    qwen_logit_lens: Dict[str, Any],
    paligemma_logit_lens: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Plot logit lens analysis comparison between models.
    
    Shows how predictions evolve across layers for both models.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Qwen
    ax = axes[0]
    if qwen_logit_lens and 'layer_confidences' in qwen_logit_lens:
        layers = sorted(qwen_logit_lens['layer_confidences'].keys())
        confs = [qwen_logit_lens['layer_confidences'][l] for l in layers]
        ax.plot(layers, confs, 'o-', color='steelblue', linewidth=2, markersize=6)
        ax.fill_between(layers, 0, confs, alpha=0.3, color='steelblue')
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Top-1 Probability", fontsize=12)
    ax.set_title("Qwen-VL Logit Lens", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # PaliGemma
    ax = axes[1]
    if paligemma_logit_lens and 'layer_confidences' in paligemma_logit_lens:
        layers = sorted(paligemma_logit_lens['layer_confidences'].keys())
        confs = [paligemma_logit_lens['layer_confidences'][l] for l in layers]
        ax.plot(layers, confs, 's-', color='coral', linewidth=2, markersize=6)
        ax.fill_between(layers, 0, confs, alpha=0.3, color='coral')
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Top-1 Probability", fontsize=12)
    ax.set_title("PaliGemma Logit Lens", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.suptitle("Logit Lens: Prediction Evolution Across Layers", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def create_publication_figure(
    comparisons: List[Dict[str, Any]],
    insights: List[Dict[str, Any]],
    save_path: str,
    figsize: Tuple[int, int] = (20, 16),
) -> plt.Figure:
    """
    Create a comprehensive publication-ready figure summarizing all findings.
    
    Args:
        comparisons: List of comparison results.
        insights: List of novel insights.
        save_path: Path to save the figure.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Extract data for visualization
    qwen_confs = [c.get('qwen_confidence', 0) for c in comparisons if c]
    pali_confs = [c.get('paligemma_confidence', 0) for c in comparisons if c]
    attn_corrs = [c.get('attention_correlation', 0) for c in comparisons if c]
    agreements = [c.get('answer_agreement', False) for c in comparisons if c]
    
    # Panel A: Confidence comparison
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(qwen_confs, pali_confs, alpha=0.6, 
               c=['green' if a else 'red' for a in agreements])
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel("Qwen Confidence", fontsize=11)
    ax.set_ylabel("PaliGemma Confidence", fontsize=11)
    ax.set_title("A. Confidence Comparison", fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Panel B: Attention correlation distribution
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(attn_corrs, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(np.mean(attn_corrs), color='red', linestyle='--', 
               label=f'Mean: {np.mean(attn_corrs):.3f}')
    ax.set_xlabel("Attention Correlation", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("B. Cross-Model Attention Correlation", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel C: Agreement rate
    ax = fig.add_subplot(gs[0, 2])
    agree_rate = np.mean(agreements) if agreements else 0
    ax.bar(['Agree', 'Disagree'], [agree_rate, 1-agree_rate], 
           color=['green', 'red'], alpha=0.7)
    ax.set_ylabel("Proportion", fontsize=11)
    ax.set_title("C. Answer Agreement Rate", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    for i, v in enumerate([agree_rate, 1-agree_rate]):
        ax.text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=10)
    
    # Panel D-F: Key insights
    for i, insight in enumerate(insights[:3]):
        ax = fig.add_subplot(gs[1, i])
        ax.axis('off')
        
        text = f"""
{insight.get('title', 'Insight')}
────────────────────────────
{insight.get('description', '')}

Significance: {insight.get('significance', 0):.2f}
        """
        
        ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=10,
                ha='center', va='center', wrap=True,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Panel G: Summary statistics
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')
    
    summary = f"""
    ══════════════════════════════════════════════════════════════════════════════════════
                            COMPARATIVE VLM ATTENTION ANALYSIS SUMMARY
    ══════════════════════════════════════════════════════════════════════════════════════
    
    Total Samples Analyzed: {len(comparisons)}
    
    CONFIDENCE METRICS:
      • Qwen Mean Confidence: {np.mean(qwen_confs):.4f} (±{np.std(qwen_confs):.4f})
      • PaliGemma Mean Confidence: {np.mean(pali_confs):.4f} (±{np.std(pali_confs):.4f})
    
    ATTENTION AGREEMENT:
      • Mean Attention Correlation: {np.mean(attn_corrs):.4f}
      • Answer Agreement Rate: {np.mean(agreements):.2%}
    
    KEY FINDINGS:
      • {len(insights)} novel insights identified
      • Primary finding: {insights[0].get('title', 'N/A') if insights else 'N/A'}
    
    ══════════════════════════════════════════════════════════════════════════════════════
    """
    
    ax.text(0.5, 0.5, summary, transform=ax.transAxes, fontsize=11,
            ha='center', va='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
    
    plt.suptitle("Comparative Analysis of Visual Attention in VLMs: Qwen-VL vs PaliGemma",
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Publication figure saved: {save_path}")
    
    return fig
