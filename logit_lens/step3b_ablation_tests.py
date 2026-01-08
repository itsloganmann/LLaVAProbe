"""
STEP 3b: ABLATION TESTS

This script performs causal ablation tests on identified important neurons:
1. Zero out top neurons (e.g., Neuron 1512)
2. Measure accuracy drop overall and per question type
3. Show task-specificity: "ablating certain neurons tanks counting but leaves color intact"

From Shikhar's plan:
"To check causality: Zeroing test: for a small set of important neurons (e.g., top‑k by probe weight):
Set their activations to 0 (or to a baseline mean) at inference time
Re‑run the model and measure: Overall VQA accuracy, Accuracy per question type"
"""

import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)


class NeuronAblator:
    """Performs neuron ablation experiments."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.device = next(model.parameters()).device

        # Get model structure
        self.lm = model.language_model
        if hasattr(self.lm, 'layers'):
            self.layers = self.lm.layers
        elif hasattr(self.lm, 'model') and hasattr(self.lm.model, 'layers'):
            self.layers = self.lm.model.layers
        else:
            raise ValueError("Cannot find transformer layers")

        self.n_layers = len(self.layers)
        self.hooks = []
        self.ablation_config = {}  # {layer_idx: [neuron_indices]}

    def set_ablation(self, ablation_config):
        """Set which neurons to ablate.
        
        Args:
            ablation_config: dict mapping layer_idx -> list of neuron indices to zero
        """
        self.ablation_config = ablation_config

    def _create_ablation_hook(self, layer_idx):
        """Create a hook that zeros out specified neurons in MLP output."""

        def hook(module, input, output):
            if layer_idx in self.ablation_config:
                neurons_to_ablate = self.ablation_config[layer_idx]
                # output shape: (batch, seq, hidden_dim)
                # Zero out the specified neurons in-place to avoid tensor issues
                with torch.no_grad():
                    for neuron_idx in neurons_to_ablate:
                        if neuron_idx < output.shape[-1]:
                            output[:, :, neuron_idx] = 0
            return output

        return hook

    def setup_ablation_hooks(self):
        """Set up hooks for ablation."""
        self.clear_hooks()

        for layer_idx in self.ablation_config.keys():
            layer = self.layers[layer_idx]
            hook = layer.mlp.register_forward_hook(
                self._create_ablation_hook(layer_idx))
            self.hooks.append(hook)

        return len(self.hooks)

    def clear_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def run_inference(self, image, question):
        """Run inference with current ablation config."""
        prompt = f"USER: <image>\n{question}\nASSISTANT:"

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[0, -1, :]
        predicted_id = logits.argmax().item()
        predicted_text = self.processor.tokenizer.decode([predicted_id
                                                          ]).strip()

        return predicted_text

    def check_correct(self, predicted, ground_truth):
        """Check if prediction matches ground truth."""
        pred_lower = predicted.lower().strip()
        gt_lower = ground_truth.lower().strip()
        return gt_lower in pred_lower or pred_lower in gt_lower


def load_samples_with_types(n_samples=200):
    """Load samples with question type information."""
    df = pd.read_csv(os.path.join(PARENT_DIR, 'results.csv'))
    samples = []
    for _, row in df.iterrows():
        samples.append({
            'question': row['question'],
            'ground_truth': row['ground_truth'],
            'image_url': row['image_url'],
            'question_type': row['question_type']
        })
        if len(samples) >= n_samples:
            break
    return samples


def run_ablation_experiment(n_samples=200):
    """Run ablation experiments on identified important neurons."""

    print("=" * 70)
    print("STEP 3b: ABLATION TESTS")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model_name = "llava-hf/llava-1.5-7b-hf"

    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(model_name)

    ablator = NeuronAblator(model, processor)
    print(f"Model loaded. {ablator.n_layers} layers.")

    # Load samples
    print("\nLoading samples...")
    samples = load_samples_with_types(n_samples)
    print(f"Using {len(samples)} samples")

    # Count question types
    q_types = {}
    for s in samples:
        qt = s['question_type']
        q_types[qt] = q_types.get(qt, 0) + 1
    print(f"Question types: {q_types}")

    # Define ablation configurations to test
    # Based on our findings: Neuron 1512 is the "super-neuron" in Layer 31
    ablation_configs = {
        'baseline': {},  # No ablation
        'neuron_1512_L31': {
            31: [1512]
        },  # Ablate super-neuron
        'neuron_1512_all': {
            17: [1512],
            21: [1512],
            27: [1512],
            30: [1512],
            31: [1512]
        },  # Ablate across layers
        'top5_L31': {
            31: [1512, 1360, 1573, 3556, 2298]
        },  # Top 5 neurons from analysis
        'random_5_L31': {
            31: [100, 500, 1000, 2000, 3000]
        },  # Random 5 for comparison
    }

    results = {}

    for config_name, config in ablation_configs.items():
        print(f"\n{'=' * 50}")
        print(f"Testing: {config_name}")
        print(f"{'=' * 50}")

        ablator.set_ablation(config)
        if config:
            ablator.setup_ablation_hooks()

        # Track results per question type
        correct_by_type = {}
        total_by_type = {}
        all_correct = 0
        all_total = 0

        for i, sample in enumerate(tqdm(samples)):
            try:
                # Load image
                response = requests.get(sample['image_url'], timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')

                # Run inference
                predicted = ablator.run_inference(image, sample['question'])
                is_correct = ablator.check_correct(predicted,
                                                   sample['ground_truth'])

                # Track results
                qt = sample['question_type']
                if qt not in correct_by_type:
                    correct_by_type[qt] = 0
                    total_by_type[qt] = 0

                if is_correct:
                    correct_by_type[qt] += 1
                    all_correct += 1
                total_by_type[qt] += 1
                all_total += 1

            except Exception as e:
                print(f"Error on sample {i}: {e}")
                continue

        ablator.clear_hooks()

        # Calculate accuracies
        overall_acc = all_correct / all_total if all_total > 0 else 0
        acc_by_type = {
            qt: correct_by_type.get(qt, 0) / total_by_type.get(qt, 1)
            for qt in total_by_type.keys()
        }

        results[config_name] = {
            'overall_accuracy': overall_acc,
            'accuracy_by_type': acc_by_type,
            'n_correct': all_correct,
            'n_total': all_total,
            'config': {
                str(k): v
                for k, v in config.items()
            }  # Convert keys to strings for JSON
        }

        print(f"\nOverall accuracy: {overall_acc*100:.1f}%")
        print("Accuracy by question type:")
        for qt, acc in acc_by_type.items():
            print(f"  {qt}: {acc*100:.1f}%")

    # Calculate drops from baseline
    print("\n" + "=" * 70)
    print("ABLATION IMPACT ANALYSIS")
    print("=" * 70)

    baseline_acc = results['baseline']['overall_accuracy']
    baseline_by_type = results['baseline']['accuracy_by_type']

    impact_analysis = {}

    for config_name, res in results.items():
        if config_name == 'baseline':
            continue

        overall_drop = baseline_acc - res['overall_accuracy']
        drops_by_type = {}

        for qt in baseline_by_type.keys():
            if qt in res['accuracy_by_type']:
                drops_by_type[
                    qt] = baseline_by_type[qt] - res['accuracy_by_type'][qt]

        impact_analysis[config_name] = {
            'overall_drop': overall_drop,
            'drops_by_type': drops_by_type
        }

        print(f"\n{config_name}:")
        print(f"  Overall accuracy drop: {overall_drop*100:.1f}%")
        print("  Drops by question type:")
        for qt, drop in drops_by_type.items():
            marker = " ← SIGNIFICANT!" if abs(drop) > 0.05 else ""
            print(f"    {qt}: {drop*100:.1f}%{marker}")

    # Save results
    save_results = {
        'n_samples': n_samples,
        'ablation_results': results,
        'impact_analysis': impact_analysis
    }

    with open(os.path.join(SCRIPT_DIR, 'step3b_ablation_results.json'),
              'w') as f:
        json.dump(save_results, f, indent=2)
    print("\nSaved: step3b_ablation_results.json")

    # Create visualization
    create_ablation_visualization(results, impact_analysis)

    return results, impact_analysis


def create_ablation_visualization(results, impact_analysis):
    """Create visualization of ablation results."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Step 3b: Neuron Ablation Tests',
                 fontsize=14,
                 fontweight='bold')

    # 1. Overall accuracy comparison
    ax1 = axes[0, 0]
    configs = list(results.keys())
    overall_accs = [results[c]['overall_accuracy'] * 100 for c in configs]
    colors = ['green' if c == 'baseline' else 'coral' for c in configs]

    bars = ax1.bar(range(len(configs)), overall_accs, color=colors)
    ax1.set_xlabel('Ablation Configuration')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Overall Accuracy by Ablation')
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.axhline(y=overall_accs[0],
                color='green',
                linestyle='--',
                alpha=0.5,
                label='Baseline')

    # Add value labels
    for bar, acc in zip(bars, overall_accs):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f'{acc:.1f}%',
                 ha='center',
                 va='bottom',
                 fontsize=8)

    # 2. Accuracy drop by configuration
    ax2 = axes[0, 1]
    ablation_configs = [c for c in configs if c != 'baseline']
    drops = [
        impact_analysis[c]['overall_drop'] * 100 for c in ablation_configs
    ]

    bars = ax2.bar(range(len(ablation_configs)), drops, color='indianred')
    ax2.set_xlabel('Ablation Configuration')
    ax2.set_ylabel('Accuracy Drop (%)')
    ax2.set_title('Accuracy Drop from Baseline')
    ax2.set_xticks(range(len(ablation_configs)))
    ax2.set_xticklabels(ablation_configs, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # 3. Per question-type breakdown for key ablations
    ax3 = axes[1, 0]

    # Get question types
    q_types = list(results['baseline']['accuracy_by_type'].keys())
    x = np.arange(len(q_types))
    width = 0.15

    # Plot baseline and key ablations
    key_configs = ['baseline', 'neuron_1512_L31', 'top5_L31']
    colors_map = {
        'baseline': 'green',
        'neuron_1512_L31': 'coral',
        'top5_L31': 'purple'
    }

    for i, config in enumerate(key_configs):
        if config in results:
            accs = [
                results[config]['accuracy_by_type'].get(qt, 0) * 100
                for qt in q_types
            ]
            ax3.bar(x + i * width,
                    accs,
                    width,
                    label=config,
                    color=colors_map.get(config, 'gray'))

    ax3.set_xlabel('Question Type')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Accuracy by Question Type')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(q_types, rotation=45, ha='right')
    ax3.legend()

    # 4. Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Find most affected question type
    if 'neuron_1512_L31' in impact_analysis:
        drops_by_type = impact_analysis['neuron_1512_L31']['drops_by_type']
        if drops_by_type:
            max_drop_type = max(drops_by_type.keys(),
                                key=lambda k: drops_by_type[k])
            max_drop = drops_by_type[max_drop_type]
            min_drop_type = min(drops_by_type.keys(),
                                key=lambda k: drops_by_type[k])
            min_drop = drops_by_type[min_drop_type]
        else:
            max_drop_type, max_drop = "N/A", 0
            min_drop_type, min_drop = "N/A", 0
    else:
        max_drop_type, max_drop = "N/A", 0
        min_drop_type, min_drop = "N/A", 0

    neuron_drop = impact_analysis.get('neuron_1512_L31', {}).get(
        'overall_drop', 0) * 100
    top5_drop = impact_analysis.get('top5_L31', {}).get('overall_drop',
                                                        0) * 100
    random_drop = impact_analysis.get('random_5_L31', {}).get(
        'overall_drop', 0) * 100

    summary_text = f"""
    ABLATION TEST SUMMARY
    
    Baseline accuracy: {results['baseline']['overall_accuracy']*100:.1f}%
    
    Key findings:
    
    1. Neuron 1512 (Layer 31) ablation:
       Accuracy drop: {neuron_drop:.1f}%
       
    2. Top 5 neurons ablation:
       Accuracy drop: {top5_drop:.1f}%
       
    3. Random 5 neurons (control):
       Accuracy drop: {random_drop:.1f}%
    
    Task-specificity:
    - Most affected: {max_drop_type} ({max_drop*100:.1f}% drop)
    - Least affected: {min_drop_type} ({min_drop*100:.1f}% drop)
    
    Interpretation:
    {"Neuron 1512 is CAUSALLY important for VQA!" if neuron_drop > random_drop else "No significant causal effect detected."}
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
    plt.savefig(os.path.join(SCRIPT_DIR, 'step3b_ablation_tests.png'),
                dpi=150,
                bbox_inches='tight')
    plt.savefig(os.path.join(SCRIPT_DIR, 'step3b_ablation_tests.pdf'),
                bbox_inches='tight')
    print("Saved: step3b_ablation_tests.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=200)
    args = parser.parse_args()

    results, impact = run_ablation_experiment(n_samples=args.n_samples)
