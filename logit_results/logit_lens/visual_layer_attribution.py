"""
VISUAL-LAYER-ONLY ATTRIBUTION

This script computes MLP vs Attention attribution only for layers where 
the visual contribution (Δmargin with vs without image) is significant.

From Shikhar's plan:
"Same but only for layers where Δmargin(image vs no image) is non-zero: 
this answers 'where does visual information get turned into a higher answer 
logit and is that mostly via MLPs?'"
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def analyze_visual_layer_attribution():
    """Analyze MLP vs Attention attribution only for visual-relevant layers."""

    print("=" * 70)
    print("VISUAL-LAYER-ONLY ATTRIBUTION")
    print("=" * 70)

    # Load step1b data (has delta_mlp and delta_attn)
    with open(os.path.join(SCRIPT_DIR, 'step1b_image_comparison.json'),
              'r') as f:
        data = json.load(f)

    avg_delta_margin = np.array(data['avg_delta_margin'])
    avg_delta_mlp = np.array(data['avg_delta_mlp'])
    avg_delta_attn = np.array(data['avg_delta_attn'])

    n_layers = len(avg_delta_margin)
    print(f"\nNumber of layers: {n_layers}")

    # Identify "visual layers" where delta_margin is significant
    # Use threshold based on standard deviation or absolute value
    threshold = 0.5  # Absolute delta margin threshold

    visual_layers = []
    for i in range(n_layers):
        if abs(avg_delta_margin[i]) > threshold:
            visual_layers.append(i)

    print(f"\nVisual layers (|Δmargin| > {threshold}):")
    for layer in visual_layers:
        print(f"  Layer {layer}: Δmargin = {avg_delta_margin[layer]:.2f}")

    # Calculate attribution ONLY for visual layers
    visual_mlp_contribution = sum(avg_delta_mlp[i] for i in visual_layers)
    visual_attn_contribution = sum(avg_delta_attn[i] for i in visual_layers)
    visual_total = visual_mlp_contribution + visual_attn_contribution

    if visual_total != 0:
        visual_mlp_fraction = visual_mlp_contribution / abs(visual_total)
        visual_attn_fraction = visual_attn_contribution / abs(visual_total)
    else:
        visual_mlp_fraction = 0
        visual_attn_fraction = 0

    print(f"\n--- VISUAL LAYERS ONLY ---")
    print(
        f"MLP contribution: {visual_mlp_contribution:.2f} ({visual_mlp_fraction*100:.1f}%)"
    )
    print(
        f"Attention contribution: {visual_attn_contribution:.2f} ({visual_attn_fraction*100:.1f}%)"
    )

    # Compare to overall attribution
    total_mlp = sum(avg_delta_mlp)
    total_attn = sum(avg_delta_attn)
    total = total_mlp + total_attn

    if total != 0:
        overall_mlp_fraction = total_mlp / abs(total)
        overall_attn_fraction = total_attn / abs(total)
    else:
        overall_mlp_fraction = 0
        overall_attn_fraction = 0

    print(f"\n--- ALL LAYERS ---")
    print(
        f"MLP contribution: {total_mlp:.2f} ({overall_mlp_fraction*100:.1f}%)")
    print(
        f"Attention contribution: {total_attn:.2f} ({overall_attn_fraction*100:.1f}%)"
    )

    # Separate positive and negative contribution layers
    positive_layers = [
        i for i in range(n_layers) if avg_delta_margin[i] > threshold
    ]
    negative_layers = [
        i for i in range(n_layers) if avg_delta_margin[i] < -threshold
    ]

    print(f"\n--- POSITIVE VISUAL LAYERS (image helps) ---")
    pos_mlp = sum(avg_delta_mlp[i] for i in positive_layers)
    pos_attn = sum(avg_delta_attn[i] for i in positive_layers)
    pos_total = pos_mlp + pos_attn
    print(f"Layers: {positive_layers}")
    print(f"MLP: {pos_mlp:.2f}, Attention: {pos_attn:.2f}")
    if pos_total != 0:
        print(f"MLP fraction: {pos_mlp/abs(pos_total)*100:.1f}%")

    print(f"\n--- NEGATIVE VISUAL LAYERS (image suppresses) ---")
    neg_mlp = sum(avg_delta_mlp[i] for i in negative_layers)
    neg_attn = sum(avg_delta_attn[i] for i in negative_layers)
    neg_total = neg_mlp + neg_attn
    print(f"Layers: {negative_layers}")
    print(f"MLP: {neg_mlp:.2f}, Attention: {neg_attn:.2f}")
    if neg_total != 0:
        print(f"MLP fraction: {neg_mlp/abs(neg_total)*100:.1f}%")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Visual-Layer-Only Attribution Analysis',
                 fontsize=14,
                 fontweight='bold')

    # 1. Delta margin per layer with threshold
    ax1 = axes[0, 0]
    colors = [
        'green' if m > threshold else 'red' if m < -threshold else 'gray'
        for m in avg_delta_margin
    ]
    ax1.bar(range(n_layers), avg_delta_margin, color=colors)
    ax1.axhline(y=threshold,
                color='black',
                linestyle='--',
                alpha=0.5,
                label=f'Threshold: ±{threshold}')
    ax1.axhline(y=-threshold, color='black', linestyle='--', alpha=0.5)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Δmargin (with - without image)')
    ax1.set_title('Visual Impact per Layer')
    ax1.legend()

    # 2. MLP vs Attention for visual layers only
    ax2 = axes[0, 1]
    x = np.arange(len(visual_layers))
    width = 0.35

    mlp_vals = [avg_delta_mlp[i] for i in visual_layers]
    attn_vals = [avg_delta_attn[i] for i in visual_layers]

    ax2.bar(x - width / 2, mlp_vals, width, label='MLP', color='steelblue')
    ax2.bar(x + width / 2, attn_vals, width, label='Attention', color='coral')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Visual Layer')
    ax2.set_ylabel('Δlogit contribution')
    ax2.set_title('MLP vs Attention (Visual Layers Only)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'L{i}' for i in visual_layers])
    ax2.legend()

    # 3. Pie chart comparison
    ax3 = axes[1, 0]

    # Create side-by-side pie charts
    labels = ['MLP', 'Attention']

    # Visual layers
    visual_sizes = [
        abs(visual_mlp_contribution),
        abs(visual_attn_contribution)
    ]
    colors_pie = ['steelblue', 'coral']

    ax3.pie(visual_sizes,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors_pie,
            startangle=90)
    ax3.set_title(
        f'Attribution in Visual Layers Only\n(Layers {visual_layers})')

    # 4. Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    VISUAL-LAYER ATTRIBUTION SUMMARY
    
    Visual layers identified: {len(visual_layers)}
    (|Δmargin| > {threshold})
    
    === IN VISUAL LAYERS ONLY ===
    
    MLP contribution:      {visual_mlp_contribution:+.2f} ({abs(visual_mlp_fraction)*100:.1f}%)
    Attention contribution: {visual_attn_contribution:+.2f} ({abs(visual_attn_fraction)*100:.1f}%)
    
    === ACROSS ALL LAYERS ===
    
    MLP contribution:      {total_mlp:+.2f} ({abs(overall_mlp_fraction)*100:.1f}%)
    Attention contribution: {total_attn:+.2f} ({abs(overall_attn_fraction)*100:.1f}%)
    
    === KEY INSIGHT ===
    
    {"ATTENTION dominates visual processing!" if abs(visual_attn_fraction) > abs(visual_mlp_fraction) else "MLP dominates visual processing!"}
    
    In layers where the image matters:
    - Attention contributes {abs(visual_attn_fraction)*100:.1f}% of visual signal
    - MLP contributes {abs(visual_mlp_fraction)*100:.1f}% of visual signal
    
    This confirms that visual information flows
    primarily through {"ATTENTION" if abs(visual_attn_fraction) > 0.5 else "MLP"} mechanisms.
    """

    ax4.text(0.1,
             0.95,
             summary_text,
             transform=ax4.transAxes,
             fontsize=10,
             verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'visual_layer_attribution.png'),
                dpi=150,
                bbox_inches='tight')
    plt.savefig(os.path.join(SCRIPT_DIR, 'visual_layer_attribution.pdf'),
                bbox_inches='tight')
    print("\nSaved: visual_layer_attribution.png")

    # Save results
    results = {
        'threshold': threshold,
        'visual_layers': visual_layers,
        'positive_layers': positive_layers,
        'negative_layers': negative_layers,
        'visual_layers_only': {
            'mlp_contribution': float(visual_mlp_contribution),
            'attn_contribution': float(visual_attn_contribution),
            'mlp_fraction': float(visual_mlp_fraction),
            'attn_fraction': float(visual_attn_fraction)
        },
        'all_layers': {
            'mlp_contribution': float(total_mlp),
            'attn_contribution': float(total_attn),
            'mlp_fraction': float(overall_mlp_fraction),
            'attn_fraction': float(overall_attn_fraction)
        },
        'positive_layers_only': {
            'mlp_contribution': float(pos_mlp),
            'attn_contribution': float(pos_attn)
        },
        'negative_layers_only': {
            'mlp_contribution': float(neg_mlp),
            'attn_contribution': float(neg_attn)
        }
    }

    with open(os.path.join(SCRIPT_DIR, 'visual_layer_attribution.json'),
              'w') as f:
        json.dump(results, f, indent=2)
    print("Saved: visual_layer_attribution.json")

    return results


if __name__ == "__main__":
    results = analyze_visual_layer_attribution()
