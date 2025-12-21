"""
PER QUESTION-TYPE ANALYSIS

This script analyzes margin trajectories and neuron activations broken down by question type:
- counting
- color  
- yes/no
- object identification
- etc.

From Shikhar's plan:
"Plot margin vs layer index, averaged over many questions and also per question‑type (counting, color, etc.)"
"Mutual information between a_i and question type or concept label"
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
import warnings

warnings.filterwarnings('ignore')

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)


def load_existing_data():
    """Load data from existing experiments."""
    # Load 1000-sample logit lens results
    with open(os.path.join(SCRIPT_DIR, 'logit_lens_results_1000.json'),
              'r') as f:
        logit_lens_data = json.load(f)

    # Load step1b image comparison (has per-sample results)
    with open(os.path.join(SCRIPT_DIR, 'step1b_image_comparison.json'),
              'r') as f:
        image_comparison_data = json.load(f)

    # Load neuron analysis
    with open(os.path.join(SCRIPT_DIR, 'step3b_neuron_analysis.json'),
              'r') as f:
        neuron_data = json.load(f)

    # Load original CSV for question types
    df = pd.read_csv(os.path.join(PARENT_DIR, 'results.csv'))

    return logit_lens_data, image_comparison_data, neuron_data, df


def analyze_by_question_type():
    """Analyze margin trajectories per question type."""

    print("=" * 70)
    print("PER QUESTION-TYPE ANALYSIS")
    print("=" * 70)

    # Load data
    logit_lens_data, image_comparison_data, neuron_data, df = load_existing_data(
    )

    # Get question types
    question_types = df['question_type'].unique()
    print(f"\nQuestion types found: {list(question_types)}")

    # Count per type
    type_counts = df['question_type'].value_counts()
    print("\nSamples per type:")
    for qt, count in type_counts.items():
        print(f"  {qt}: {count}")

    # Analyze step1b results by question type
    # Match questions to their types
    results_with_image = image_comparison_data.get('results_with_image', [])

    # Create mapping from question to type
    q_to_type = dict(zip(df['question'], df['question_type']))

    # Group results by question type
    margins_by_type = {qt: [] for qt in question_types}
    correct_by_type = {
        qt: {
            'correct': [],
            'incorrect': []
        }
        for qt in question_types
    }

    for result in results_with_image:
        question = result['question']
        qt = q_to_type.get(question, 'unknown')
        if qt in margins_by_type:
            margins_by_type[qt].append(result['margins'])
            if result['is_correct']:
                correct_by_type[qt]['correct'].append(result['margins'])
            else:
                correct_by_type[qt]['incorrect'].append(result['margins'])

    # Calculate average margins per type
    avg_margins_by_type = {}
    for qt in question_types:
        if margins_by_type[qt]:
            avg_margins_by_type[qt] = np.mean(margins_by_type[qt], axis=0)
            print(f"\n{qt}: {len(margins_by_type[qt])} samples")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Per Question-Type Analysis', fontsize=14, fontweight='bold')

    # 1. Margin trajectory by question type
    ax1 = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(avg_margins_by_type)))

    for (qt, margins), color in zip(avg_margins_by_type.items(), colors):
        ax1.plot(margins, label=qt, color=color, linewidth=2)

    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Margin (logit_correct - logit_second)')
    ax1.set_title('Margin Trajectory by Question Type')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Final margin distribution by type
    ax2 = axes[0, 1]
    final_margins = {
        qt: [m[-1] for m in margins_by_type[qt]]
        for qt in question_types if margins_by_type[qt]
    }

    positions = range(len(final_margins))
    bp = ax2.boxplot([final_margins[qt] for qt in final_margins.keys()],
                     positions=positions,
                     patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_xticklabels(list(final_margins.keys()), rotation=45, ha='right')
    ax2.set_xlabel('Question Type')
    ax2.set_ylabel('Final Layer Margin')
    ax2.set_title('Final Margin Distribution by Question Type')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # 3. Correct vs Incorrect by type (for types with enough data)
    ax3 = axes[1, 0]

    # Pick types with at least 3 correct and 3 incorrect
    valid_types = [
        qt for qt in question_types if len(correct_by_type[qt]['correct']) >= 3
        and len(correct_by_type[qt]['incorrect']) >= 3
    ]

    if valid_types:
        # Just plot one type as example (or first valid)
        example_type = valid_types[0] if valid_types else list(
            question_types)[0]

        if correct_by_type[example_type]['correct']:
            correct_avg = np.mean(correct_by_type[example_type]['correct'],
                                  axis=0)
            ax3.plot(correct_avg, 'g-', linewidth=2, label='Correct')

        if correct_by_type[example_type]['incorrect']:
            incorrect_avg = np.mean(correct_by_type[example_type]['incorrect'],
                                    axis=0)
            ax3.plot(incorrect_avg, 'r-', linewidth=2, label='Incorrect')

        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Margin')
        ax3.set_title(f'Correct vs Incorrect: {example_type}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5,
                 0.5,
                 'Not enough data for\ncorrect/incorrect comparison',
                 ha='center',
                 va='center',
                 transform=ax3.transAxes)
        ax3.set_title('Correct vs Incorrect (Insufficient Data)')

    # 4. Layer where margin becomes positive (per type)
    ax4 = axes[1, 1]

    crossover_layers = {}
    for qt, margins in avg_margins_by_type.items():
        # Find first layer where margin > 0
        for i, m in enumerate(margins):
            if m > 0:
                crossover_layers[qt] = i
                break
        else:
            crossover_layers[qt] = len(margins)  # Never crossed

    types_sorted = sorted(crossover_layers.keys(),
                          key=lambda k: crossover_layers[k])
    layers = [crossover_layers[qt] for qt in types_sorted]

    bars = ax4.barh(range(len(types_sorted)), layers, color='steelblue')
    ax4.set_yticks(range(len(types_sorted)))
    ax4.set_yticklabels(types_sorted)
    ax4.set_xlabel('Layer Index')
    ax4.set_title('Layer Where Margin Becomes Positive')
    ax4.axvline(x=np.mean(layers),
                color='red',
                linestyle='--',
                label=f'Mean: {np.mean(layers):.1f}')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'question_type_analysis.png'),
                dpi=150,
                bbox_inches='tight')
    plt.savefig(os.path.join(SCRIPT_DIR, 'question_type_analysis.pdf'),
                bbox_inches='tight')
    print("\nSaved: question_type_analysis.png")

    # Save detailed results
    results = {
        'avg_margins_by_type': {
            qt: list(m)
            for qt, m in avg_margins_by_type.items()
        },
        'crossover_layers': crossover_layers,
        'samples_per_type': {
            qt: len(margins_by_type[qt])
            for qt in question_types
        },
        'final_margin_stats': {
            qt: {
                'mean':
                float(np.mean(final_margins[qt]))
                if qt in final_margins else None,
                'std':
                float(np.std(final_margins[qt]))
                if qt in final_margins else None,
                'median':
                float(np.median(final_margins[qt]))
                if qt in final_margins else None
            }
            for qt in question_types
        }
    }

    with open(os.path.join(SCRIPT_DIR, 'question_type_analysis.json'),
              'w') as f:
        json.dump(results, f, indent=2)
    print("Saved: question_type_analysis.json")

    # Print summary
    print("\n" + "=" * 70)
    print("QUESTION TYPE SUMMARY")
    print("=" * 70)

    print("\nCrossover layer (where margin becomes positive):")
    for qt in types_sorted:
        print(f"  {qt}: Layer {crossover_layers[qt]}")

    print("\nFinal margin statistics:")
    for qt in question_types:
        if qt in final_margins and final_margins[qt]:
            mean_m = np.mean(final_margins[qt])
            std_m = np.std(final_margins[qt])
            print(f"  {qt}: {mean_m:.2f} ± {std_m:.2f}")

    return results


def analyze_visual_contribution_by_type():
    """Analyze how much visual info contributes per question type."""

    print("\n" + "=" * 70)
    print("VISUAL CONTRIBUTION BY QUESTION TYPE")
    print("=" * 70)

    # Load data
    with open(os.path.join(SCRIPT_DIR, 'step1b_image_comparison.json'),
              'r') as f:
        data = json.load(f)

    df = pd.read_csv(os.path.join(PARENT_DIR, 'results.csv'))
    q_to_type = dict(zip(df['question'], df['question_type']))
    question_types = df['question_type'].unique()

    results_with = data.get('results_with_image', [])
    results_without = data.get('results_no_image', [])

    # Match results
    # Assuming same order
    delta_margins_by_type = {qt: [] for qt in question_types}

    for i, (r_with, r_without) in enumerate(zip(results_with,
                                                results_without)):
        question = r_with['question']
        qt = q_to_type.get(question, 'unknown')

        if qt in delta_margins_by_type:
            delta = np.array(r_with['margins']) - np.array(
                r_without['margins'])
            delta_margins_by_type[qt].append(delta)

    # Calculate average delta margins
    avg_delta_by_type = {}
    for qt in question_types:
        if delta_margins_by_type[qt]:
            avg_delta_by_type[qt] = np.mean(delta_margins_by_type[qt], axis=0)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Visual Contribution by Question Type',
                 fontsize=14,
                 fontweight='bold')

    # 1. Delta margin trajectory by type
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(avg_delta_by_type)))

    for (qt, delta), color in zip(avg_delta_by_type.items(), colors):
        ax1.plot(delta, label=qt, color=color, linewidth=2)

    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Δmargin (with image - without)')
    ax1.set_title('Visual Contribution per Layer by Question Type')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Total visual contribution (sum of positive deltas)
    ax2 = axes[1]

    total_visual = {}
    for qt, delta in avg_delta_by_type.items():
        # Sum only positive contributions (where image helps)
        total_visual[qt] = np.sum(np.maximum(delta, 0))

    types_sorted = sorted(total_visual.keys(),
                          key=lambda k: total_visual[k],
                          reverse=True)
    values = [total_visual[qt] for qt in types_sorted]

    bars = ax2.bar(range(len(types_sorted)), values, color='coral')
    ax2.set_xticks(range(len(types_sorted)))
    ax2.set_xticklabels(types_sorted, rotation=45, ha='right')
    ax2.set_xlabel('Question Type')
    ax2.set_ylabel('Total Positive Δmargin')
    ax2.set_title('Total Visual Contribution by Question Type')

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'visual_contribution_by_type.png'),
                dpi=150,
                bbox_inches='tight')
    print("\nSaved: visual_contribution_by_type.png")

    # Print findings
    print("\nTotal visual contribution (sum of positive Δmargin):")
    for qt in types_sorted:
        print(f"  {qt}: {total_visual[qt]:.2f}")

    return avg_delta_by_type, total_visual


if __name__ == "__main__":
    results = analyze_by_question_type()
    delta_results, total_visual = analyze_visual_contribution_by_type()
