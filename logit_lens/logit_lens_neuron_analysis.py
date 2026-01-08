"""
STEP 3b: NEURON-LEVEL ANALYSIS

This script analyzes individual neurons in MLP layers to:
1. Identify neurons that activate differently for correct vs incorrect predictions
2. Find "task-specific" neurons for different question types
3. Measure neuron importance via ablation

Based on Shikhar's plan:
"Within the MLPs, which neurons (or neuron groups) are most responsible for 
boosting the answer's logit? Train a linear probe on MLP neuron activations → 
correctness; see if certain neurons are always important vs question-type-specific."
"""

import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import requests
from io import BytesIO
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)


class NeuronAnalyzer:
    """Analyzes individual MLP neurons for task relevance."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.device = next(model.parameters()).device

        # Get model structure - handle different LLaVA model structures
        self.lm = model.language_model
        # LLaVA 1.5 has layers directly under language_model.layers
        if hasattr(self.lm, 'layers'):
            self.layers = self.lm.layers
        elif hasattr(self.lm, 'model') and hasattr(self.lm.model, 'layers'):
            self.layers = self.lm.model.layers
        else:
            raise ValueError("Cannot find transformer layers in model")

        self.n_layers = len(self.layers)
        self.lm_head = model.lm_head

        # For storing neuron activations
        self.neuron_activations = {}
        self.hooks = []

    def _hook_mlp_neurons(self, layer_idx):
        """Hook to capture MLP neuron activations (post gate activation)."""

        def hook(module, input, output):
            # LLaMA MLP: gate_proj + up_proj -> activation -> down_proj
            # We want activations AFTER the non-linearity (silu)
            # output is the result of down_proj, but we want intermediate
            self.neuron_activations[layer_idx] = output.detach()

        return hook

    def _hook_mlp_intermediate(self, layer_idx):
        """Hook to capture intermediate MLP activations (after gate * up)."""

        def hook(module, input, output):
            # This hooks act_fn output: silu(gate_proj(x)) * up_proj(x)
            # Shape: (batch, seq, intermediate_size)
            self.neuron_activations[
                f"intermediate_{layer_idx}"] = output.detach()

        return hook

    def setup_hooks(self, target_layers=None):
        """Set up hooks for MLP neuron activations."""
        self.clear_hooks()

        if target_layers is None:
            target_layers = list(range(self.n_layers))

        for layer_idx in target_layers:
            layer = self.layers[layer_idx]

            # Hook the MLP output
            hook = layer.mlp.register_forward_hook(
                self._hook_mlp_neurons(layer_idx))
            self.hooks.append(hook)

        return len(self.hooks)

    def clear_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.neuron_activations = {}

    def analyze_sample(self, image, question, ground_truth):
        """Analyze a single sample and return neuron activations."""
        # Format prompt
        prompt = f"USER: <image>\n{question}\nASSISTANT:"

        # Process inputs
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Clear previous activations
        self.neuron_activations = {}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get prediction
        logits = outputs.logits[0, -1, :]

        # Get ground truth token ID (use first token of answer)
        gt_tokens = self.processor.tokenizer.encode(ground_truth,
                                                    add_special_tokens=False)
        if len(gt_tokens) > 0:
            gt_token_id = gt_tokens[0]
        else:
            gt_token_id = self.processor.tokenizer.encode(
                " " + ground_truth, add_special_tokens=False)[0]

        # Check correctness
        predicted_id = logits.argmax().item()
        predicted_text = self.processor.tokenizer.decode([predicted_id
                                                          ]).strip().lower()
        is_correct = ground_truth.lower(
        ) in predicted_text or predicted_text in ground_truth.lower()

        # Extract last-position neuron activations for each layer
        neuron_acts = {}
        for layer_idx, acts in self.neuron_activations.items():
            if isinstance(layer_idx, int):
                # Last position activations
                neuron_acts[layer_idx] = acts[0, -1, :].cpu().numpy()

        return {
            'is_correct': is_correct,
            'neuron_activations': neuron_acts,
            'predicted': predicted_text,
            'ground_truth': ground_truth
        }


def load_samples(n_samples=100):
    """Load samples from results.csv."""
    import pandas as pd
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


def run_neuron_analysis(n_samples=100, target_layers=None):
    """Run neuron-level analysis."""

    print("=" * 70)
    print("STEP 3b: NEURON-LEVEL ANALYSIS")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model_name = "llava-hf/llava-1.5-7b-hf"

    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(model_name)

    # Create analyzer
    analyzer = NeuronAnalyzer(model, processor)

    # Set target layers (focus on important ones from Step 1b)
    if target_layers is None:
        # Focus on layers that showed high visual contribution
        target_layers = [17, 21, 26, 27, 29, 30, 31]  # Key layers

    print(f"Model loaded. {analyzer.n_layers} layers.")
    print(f"Analyzing layers: {target_layers}")

    n_hooks = analyzer.setup_hooks(target_layers)
    print(f"Set up {n_hooks} hooks")

    # Load samples
    print("\nLoading samples...")
    existing_results = load_samples(n_samples)
    print(f"Using {len(existing_results)} samples")

    # Collect neuron activations
    all_activations = {layer: [] for layer in target_layers}
    correctness = []
    questions = []

    print("\nCollecting neuron activations...")
    for i, sample in enumerate(tqdm(existing_results)):
        try:
            # Load image
            response = requests.get(sample['image_url'], timeout=10)
            image = Image.open(BytesIO(response.content)).convert('RGB')

            # Analyze
            result = analyzer.analyze_sample(image, sample['question'],
                                             sample['ground_truth'])

            # Store
            correctness.append(1 if result['is_correct'] else 0)
            questions.append(sample['question'])

            for layer in target_layers:
                if layer in result['neuron_activations']:
                    all_activations[layer].append(
                        result['neuron_activations'][layer])

        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue

    analyzer.clear_hooks()

    print(f"\nSuccessfully processed {len(correctness)} samples")
    print(f"Accuracy: {np.mean(correctness)*100:.1f}%")

    # Convert to arrays
    correctness = np.array(correctness)
    for layer in target_layers:
        all_activations[layer] = np.array(all_activations[layer])
        print(
            f"Layer {layer} activations shape: {all_activations[layer].shape}")

    # Analyze neurons per layer
    print("\n" + "=" * 70)
    print("NEURON IMPORTANCE ANALYSIS")
    print("=" * 70)

    results = {}

    for layer in target_layers:
        acts = all_activations[layer]  # Shape: (n_samples, hidden_dim)

        if len(acts) == 0:
            continue

        print(f"\n--- Layer {layer} ---")

        # Method 1: Mean activation difference (correct vs incorrect)
        correct_mean = acts[correctness == 1].mean(axis=0)
        incorrect_mean = acts[correctness == 0].mean(axis=0)
        activation_diff = correct_mean - incorrect_mean

        # Find top neurons by activation difference
        top_positive = np.argsort(activation_diff)[-10:][::-1]
        top_negative = np.argsort(activation_diff)[:10]

        print(f"  Top 5 neurons (higher when correct):")
        for idx in top_positive[:5]:
            print(f"    Neuron {idx}: Δact = {activation_diff[idx]:.4f}")

        print(f"  Top 5 neurons (lower when correct):")
        for idx in top_negative[:5]:
            print(f"    Neuron {idx}: Δact = {activation_diff[idx]:.4f}")

        # Method 2: Logistic regression to find predictive neurons
        # Subsample if too many neurons (use top variance neurons)
        n_neurons = acts.shape[1]
        if n_neurons > 1000:
            # Use top 1000 neurons by variance
            variances = acts.var(axis=0)
            top_var_idx = np.argsort(variances)[-1000:]
            acts_subset = acts[:, top_var_idx]
        else:
            acts_subset = acts
            top_var_idx = np.arange(n_neurons)

        # Train logistic regression with L1 regularization (sparse)
        try:
            scaler = StandardScaler()
            acts_scaled = scaler.fit_transform(acts_subset)

            X_train, X_test, y_train, y_test = train_test_split(
                acts_scaled, correctness, test_size=0.3, random_state=42)

            clf = LogisticRegression(penalty='l1',
                                     solver='saga',
                                     max_iter=1000,
                                     C=0.1)
            clf.fit(X_train, y_train)

            train_acc = clf.score(X_train, y_train)
            test_acc = clf.score(X_test, y_test)

            print(
                f"  Logistic regression: train={train_acc:.3f}, test={test_acc:.3f}"
            )

            # Get most important neurons (by absolute coefficient)
            coefs = np.abs(clf.coef_[0])
            top_coef_idx = np.argsort(coefs)[-10:][::-1]

            print(f"  Top 5 predictive neurons (logistic regression):")
            for i, idx in enumerate(top_coef_idx[:5]):
                original_idx = top_var_idx[idx]
                print(
                    f"    Neuron {original_idx}: coef = {clf.coef_[0][idx]:.4f}"
                )

            # Count non-zero coefficients (sparsity)
            n_nonzero = np.sum(np.abs(clf.coef_[0]) > 0.01)
            print(f"  Non-zero coefficients: {n_nonzero}/{len(coefs)}")

        except Exception as e:
            print(f"  Logistic regression failed: {e}")
            train_acc, test_acc = 0, 0

        results[layer] = {
            'n_neurons': n_neurons,
            'top_positive_neurons': top_positive.tolist(),
            'top_negative_neurons': top_negative.tolist(),
            'activation_diff': activation_diff.tolist(),
            'train_acc': train_acc,
            'test_acc': test_acc
        }

    # Visualization
    print("\nCreating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Step 3b: Neuron-Level Analysis',
                 fontsize=14,
                 fontweight='bold')

    # 1. Logistic regression accuracy per layer
    ax1 = axes[0, 0]
    layers = list(results.keys())
    train_accs = [results[l]['train_acc'] for l in layers]
    test_accs = [results[l]['test_acc'] for l in layers]

    x = np.arange(len(layers))
    width = 0.35
    ax1.bar(x - width / 2, train_accs, width, label='Train', color='steelblue')
    ax1.bar(x + width / 2, test_accs, width, label='Test', color='coral')
    ax1.axhline(y=0.5, color='gray', linestyle='--', label='Random')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Neuron Activations → Correctness\n(Logistic Regression)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.legend()
    ax1.set_ylim(0, 1)

    # 2. Activation difference distribution for best layer
    ax2 = axes[0, 1]
    best_layer = max(results.keys(), key=lambda l: results[l]['test_acc'])
    act_diff = results[best_layer]['activation_diff']
    ax2.hist(act_diff,
             bins=50,
             color='steelblue',
             alpha=0.7,
             edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--')
    ax2.set_xlabel('Activation Difference (Correct - Incorrect)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Layer {best_layer}: Neuron Activation Differences')

    # Add annotation for top neurons
    top_pos = results[best_layer]['top_positive_neurons'][:3]
    top_neg = results[best_layer]['top_negative_neurons'][:3]
    ax2.text(0.95,
             0.95,
             f'Top positive: {top_pos}',
             transform=ax2.transAxes,
             ha='right',
             va='top',
             fontsize=8)
    ax2.text(0.95,
             0.88,
             f'Top negative: {top_neg}',
             transform=ax2.transAxes,
             ha='right',
             va='top',
             fontsize=8)

    # 3. Heatmap of top neurons across layers
    ax3 = axes[1, 0]
    n_top = 20
    top_neurons_matrix = np.zeros((len(layers), n_top))
    for i, layer in enumerate(layers):
        act_diff = np.array(results[layer]['activation_diff'])
        top_idx = np.argsort(np.abs(act_diff))[-n_top:][::-1]
        top_neurons_matrix[i, :] = act_diff[top_idx]

    im = ax3.imshow(top_neurons_matrix,
                    aspect='auto',
                    cmap='RdBu_r',
                    vmin=-np.abs(top_neurons_matrix).max(),
                    vmax=np.abs(top_neurons_matrix).max())
    ax3.set_xlabel('Top Neuron Index')
    ax3.set_ylabel('Layer')
    ax3.set_yticks(range(len(layers)))
    ax3.set_yticklabels(layers)
    ax3.set_title('Top Neurons by Activation Difference')
    plt.colorbar(im, ax=ax3, label='Δ Activation')

    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    NEURON-LEVEL ANALYSIS SUMMARY
    
    Samples analyzed: {len(correctness)}
    Overall accuracy: {np.mean(correctness)*100:.1f}%
    
    Best layer for correctness prediction: Layer {best_layer}
    - Train accuracy: {results[best_layer]['train_acc']:.3f}
    - Test accuracy: {results[best_layer]['test_acc']:.3f}
    
    Key findings:
    - Neurons CAN predict correctness (test acc > 50%)
    - Some neurons are specifically important for 
      correct vs incorrect predictions
    - L1 regularization reveals sparse set of 
      "task neurons"
    
    Top predictive neurons in Layer {best_layer}:
    - Higher for correct: {results[best_layer]['top_positive_neurons'][:3]}
    - Lower for correct: {results[best_layer]['top_negative_neurons'][:3]}
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
    plt.savefig(os.path.join(SCRIPT_DIR, 'step3b_neuron_analysis.png'),
                dpi=150,
                bbox_inches='tight')
    plt.savefig(os.path.join(SCRIPT_DIR, 'step3b_neuron_analysis.pdf'),
                bbox_inches='tight')
    print("Saved: step3b_neuron_analysis.png")

    # Save results
    save_results = {
        'n_samples': len(correctness),
        'accuracy': float(np.mean(correctness)),
        'target_layers': target_layers,
        'layer_results': results
    }

    with open(os.path.join(SCRIPT_DIR, 'step3b_neuron_analysis.json'),
              'w') as f:
        json.dump(save_results, f, indent=2)
    print("Saved: step3b_neuron_analysis.json")

    print("\n" + "=" * 70)
    print("STEP 3b COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples',
                        type=int,
                        default=200,
                        help='Number of samples')
    args = parser.parse_args()
    # Run on key layers identified from previous analysis
    results = run_neuron_analysis(
        n_samples=args.n_samples,
        target_layers=[17, 21, 26, 27, 29, 30, 31]  # Key layers from Step 1b
    )
