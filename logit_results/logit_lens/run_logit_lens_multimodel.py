#!/usr/bin/env python3
"""
Multi-Model Logit Lens and Neuron Analysis

Runs comprehensive logit lens analysis on Qwen2-VL and PaliGemma to compare
with LLaVA-13B results. Tests:
1. Where in the model the answer decision is made
2. MLP vs Attention contribution
3. Neuron-level analysis
4. Correct vs Incorrect margin trajectories

Author: Yi Xia
Date: January 2026
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
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import argparse

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)


@dataclass
class LogitLensResult:
    """Results from logit lens analysis for a single sample."""
    question: str
    ground_truth: str
    predicted_token: str
    is_correct: bool
    layer_margins: np.ndarray  # Shape: (n_layers,)
    mlp_contributions: Optional[np.ndarray] = None
    attn_contributions: Optional[np.ndarray] = None
    question_type: Optional[str] = None


# ==============================================================================
# QWEN2-VL ANALYZER
# ==============================================================================


class Qwen2VLLogitLensAnalyzer:
    """Logit lens analyzer for Qwen2-VL model."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None
        self.n_layers = 28

        # Storage for hooked activations
        self.residual_streams = {}
        self.attn_outputs = {}
        self.mlp_outputs = {}
        self.hooks = []

    def load_model(self):
        """Load Qwen2-VL model."""
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        print("Loading Qwen2-VL-7B-Instruct...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)
        self.model.eval()

        # Get number of layers - Qwen2-VL has model.model.language_model.layers
        self.language_model = self.model.model.language_model
        self.n_layers = len(self.language_model.layers)
        print(f"Qwen2-VL loaded. {self.n_layers} layers.")

    def _get_lm_head(self):
        """Get the language model head for projecting hidden states to vocab."""
        return self.model.lm_head

    def _get_norm(self):
        """Get the final layer norm."""
        return self.language_model.norm

    def _hook_residual(self, layer_idx: int):
        """Hook to capture residual stream after a layer."""

        def hook(module, input, output):
            # output[0] is the hidden states
            self.residual_streams[layer_idx] = output[0].detach()

        return hook

    def _hook_attn(self, layer_idx: int):
        """Hook attention output."""

        def hook(module, input, output):
            self.attn_outputs[layer_idx] = output[0].detach()

        return hook

    def _hook_mlp(self, layer_idx: int):
        """Hook MLP output."""

        def hook(module, input, output):
            self.mlp_outputs[layer_idx] = output.detach()

        return hook

    def setup_hooks(self):
        """Set up hooks for all layers."""
        self.clear_hooks()

        for i in range(self.n_layers):
            layer = self.language_model.layers[i]

            # Hook after full layer (residual stream)
            h = layer.register_forward_hook(self._hook_residual(i))
            self.hooks.append(h)

            # Hook attention output
            h = layer.self_attn.register_forward_hook(self._hook_attn(i))
            self.hooks.append(h)

            # Hook MLP output
            h = layer.mlp.register_forward_hook(self._hook_mlp(i))
            self.hooks.append(h)

    def clear_hooks(self):
        """Remove all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.residual_streams = {}
        self.attn_outputs = {}
        self.mlp_outputs = {}

    def analyze_sample(self, image: Image.Image, question: str,
                       ground_truth: str) -> LogitLensResult:
        """Analyze a single sample with logit lens."""
        from qwen_vl_utils import process_vision_info

        # Format message
        messages = [{
            "role":
            "user",
            "content": [{
                "type": "image",
                "image": image
            }, {
                "type": "text",
                "text": question
            }]
        }]

        # Process
        text = self.processor.apply_chat_template(messages,
                                                  tokenize=False,
                                                  add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text],
                                images=image_inputs,
                                videos=video_inputs,
                                padding=True,
                                return_tensors="pt").to(self.device)

        # Clear and setup hooks
        self.residual_streams = {}
        self.attn_outputs = {}
        self.mlp_outputs = {}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Get prediction
        logits = outputs.logits[0, -1, :]
        predicted_id = logits.argmax().item()
        predicted_token = self.processor.tokenizer.decode([predicted_id
                                                           ]).strip().lower()

        # Check correctness
        gt_lower = ground_truth.lower().strip()
        is_correct = gt_lower in predicted_token or predicted_token in gt_lower

        # Compute layer-wise margins
        # Get ground truth token ID
        gt_tokens = self.processor.tokenizer.encode(ground_truth,
                                                    add_special_tokens=False)
        gt_token_id = gt_tokens[0] if gt_tokens else predicted_id

        # Get hidden states from outputs
        hidden_states = outputs.hidden_states  # Tuple of (n_layers + 1,) tensors

        layer_margins = []
        mlp_contributions = []
        attn_contributions = []

        lm_head = self._get_lm_head()
        norm = self._get_norm()

        for layer_idx in range(self.n_layers):
            # Get hidden state at this layer (after layer, before norm)
            h = hidden_states[layer_idx + 1][0, -1, :]  # Last position

            # Apply final norm and project to vocab
            h_normed = norm(h.unsqueeze(0).unsqueeze(0)).squeeze()
            layer_logits = lm_head(h_normed.unsqueeze(0)).squeeze()

            # Compute margin: logit(correct) - max(other logits)
            gt_logit = layer_logits[gt_token_id].item()
            other_logits = layer_logits.clone()
            other_logits[gt_token_id] = float('-inf')
            max_other = other_logits.max().item()
            margin = gt_logit - max_other
            layer_margins.append(margin)

            # MLP/Attention contributions (approximate via activation norms)
            if layer_idx in self.mlp_outputs and layer_idx in self.attn_outputs:
                mlp_out = self.mlp_outputs[layer_idx][0, -1, :]
                attn_out = self.attn_outputs[layer_idx][0, -1, :]
                mlp_contributions.append(mlp_out.norm().item())
                attn_contributions.append(attn_out.norm().item())
            else:
                mlp_contributions.append(0)
                attn_contributions.append(0)

        return LogitLensResult(question=question,
                               ground_truth=ground_truth,
                               predicted_token=predicted_token,
                               is_correct=is_correct,
                               layer_margins=np.array(layer_margins),
                               mlp_contributions=np.array(mlp_contributions),
                               attn_contributions=np.array(attn_contributions))


# ==============================================================================
# PALIGEMMA ANALYZER
# ==============================================================================


class PaliGemmaLogitLensAnalyzer:
    """Logit lens analyzer for PaliGemma model."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None
        self.n_layers = 18

        # Storage for hooked activations
        self.residual_streams = {}
        self.attn_outputs = {}
        self.mlp_outputs = {}
        self.hooks = []

    def load_model(self):
        """Load PaliGemma model."""
        from transformers import PaliGemmaForConditionalGeneration, AutoProcessor

        print("Loading PaliGemma-3B...")
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            "google/paligemma-3b-mix-224",
            torch_dtype=torch.bfloat16,
            device_map="auto")
        self.processor = AutoProcessor.from_pretrained(
            "google/paligemma-3b-mix-224")
        self.model.eval()

        # Get number of layers - PaliGemma uses language_model.model.layers
        self.n_layers = len(self.model.language_model.layers)
        print(f"PaliGemma loaded. {self.n_layers} layers.")

    def _get_lm_head(self):
        """Get the language model head."""
        # Note: PaliGemma has lm_head at model level, not language_model level
        return self.model.lm_head

    def _get_norm(self):
        """Get the final layer norm."""
        return self.model.language_model.norm

    def _hook_residual(self, layer_idx: int):

        def hook(module, input, output):
            self.residual_streams[layer_idx] = output[0].detach()

        return hook

    def _hook_attn(self, layer_idx: int):

        def hook(module, input, output):
            self.attn_outputs[layer_idx] = output[0].detach()

        return hook

    def _hook_mlp(self, layer_idx: int):

        def hook(module, input, output):
            self.mlp_outputs[layer_idx] = output.detach()

        return hook

    def setup_hooks(self):
        """Set up hooks for all layers."""
        self.clear_hooks()

        for i in range(self.n_layers):
            layer = self.model.language_model.layers[i]

            h = layer.register_forward_hook(self._hook_residual(i))
            self.hooks.append(h)

            h = layer.self_attn.register_forward_hook(self._hook_attn(i))
            self.hooks.append(h)

            h = layer.mlp.register_forward_hook(self._hook_mlp(i))
            self.hooks.append(h)

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.residual_streams = {}
        self.attn_outputs = {}
        self.mlp_outputs = {}

    def analyze_sample(self, image: Image.Image, question: str,
                       ground_truth: str) -> LogitLensResult:
        """Analyze a single sample with logit lens."""

        # Format prompt for PaliGemma
        prompt = question

        # Process inputs
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Clear activations
        self.residual_streams = {}
        self.attn_outputs = {}
        self.mlp_outputs = {}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Get prediction
        logits = outputs.logits[0, -1, :]
        predicted_id = logits.argmax().item()
        predicted_token = self.processor.tokenizer.decode([predicted_id
                                                           ]).strip().lower()

        # Check correctness
        gt_lower = ground_truth.lower().strip()
        is_correct = gt_lower in predicted_token or predicted_token in gt_lower

        # Get ground truth token ID
        gt_tokens = self.processor.tokenizer.encode(ground_truth,
                                                    add_special_tokens=False)
        gt_token_id = gt_tokens[0] if gt_tokens else predicted_id

        # Get hidden states
        hidden_states = outputs.hidden_states

        layer_margins = []
        mlp_contributions = []
        attn_contributions = []

        lm_head = self._get_lm_head()
        norm = self._get_norm()

        for layer_idx in range(self.n_layers):
            h = hidden_states[layer_idx + 1][0, -1, :]

            h_normed = norm(h.unsqueeze(0).unsqueeze(0)).squeeze()
            layer_logits = lm_head(h_normed.unsqueeze(0)).squeeze()

            gt_logit = layer_logits[gt_token_id].item()
            other_logits = layer_logits.clone()
            other_logits[gt_token_id] = float('-inf')
            max_other = other_logits.max().item()
            margin = gt_logit - max_other
            layer_margins.append(margin)

            if layer_idx in self.mlp_outputs and layer_idx in self.attn_outputs:
                mlp_out = self.mlp_outputs[layer_idx][0, -1, :]
                attn_out = self.attn_outputs[layer_idx][0, -1, :]
                mlp_contributions.append(mlp_out.norm().item())
                attn_contributions.append(attn_out.norm().item())
            else:
                mlp_contributions.append(0)
                attn_contributions.append(0)

        return LogitLensResult(question=question,
                               ground_truth=ground_truth,
                               predicted_token=predicted_token,
                               is_correct=is_correct,
                               layer_margins=np.array(layer_margins),
                               mlp_contributions=np.array(mlp_contributions),
                               attn_contributions=np.array(attn_contributions))


# ==============================================================================
# NEURON ANALYZER (Generic for both models)
# ==============================================================================


class NeuronAnalyzer:
    """Analyzes individual MLP neurons for task relevance."""

    def __init__(self, model, processor, model_type: str):
        self.model = model
        self.processor = processor
        self.model_type = model_type
        self.device = next(model.parameters()).device

        # Get layers based on model type
        if model_type == "qwen2-vl":
            self.layers = model.model.language_model.layers
        elif model_type == "paligemma":
            self.layers = model.language_model.layers

        self.n_layers = len(self.layers)
        self.neuron_activations = {}
        self.hooks = []

    def _hook_mlp_neurons(self, layer_idx: int):

        def hook(module, input, output):
            self.neuron_activations[layer_idx] = output.detach()

        return hook

    def setup_hooks(self, target_layers: List[int] = None):
        """Set up hooks for MLP neuron activations."""
        self.clear_hooks()

        if target_layers is None:
            target_layers = list(range(self.n_layers))

        for layer_idx in target_layers:
            layer = self.layers[layer_idx]
            hook = layer.mlp.register_forward_hook(
                self._hook_mlp_neurons(layer_idx))
            self.hooks.append(hook)

        return len(self.hooks)

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.neuron_activations = {}

    def analyze_sample(self, image: Image.Image, question: str,
                       ground_truth: str) -> Dict:
        """Analyze a single sample and return neuron activations."""

        # Clear previous activations
        self.neuron_activations = {}

        if self.model_type == "qwen2-vl":
            from qwen_vl_utils import process_vision_info
            messages = [{
                "role":
                "user",
                "content": [{
                    "type": "image",
                    "image": image
                }, {
                    "type": "text",
                    "text": question
                }]
            }]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(text=[text],
                                    images=image_inputs,
                                    videos=video_inputs,
                                    padding=True,
                                    return_tensors="pt").to(self.device)
        else:  # paligemma
            inputs = self.processor(text=question,
                                    images=image,
                                    return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get prediction logits
        logits = outputs.logits[0, -1, :]

        # Get ground truth token ID
        gt_tokens = self.processor.tokenizer.encode(ground_truth,
                                                    add_special_tokens=False)
        gt_token_id = gt_tokens[0] if gt_tokens else 0

        # Compute margin: GT logit - max other logit
        gt_logit = logits[gt_token_id].item()
        other_logits = logits.clone()
        other_logits[gt_token_id] = float('-inf')
        max_other = other_logits.max().item()
        margin = gt_logit - max_other

        predicted_id = logits.argmax().item()
        predicted_text = self.processor.tokenizer.decode([predicted_id
                                                          ]).strip().lower()

        # Extract neuron activations (convert bfloat16 to float32 for numpy)
        neuron_acts = {}
        for layer_idx, acts in self.neuron_activations.items():
            if isinstance(layer_idx, int):
                neuron_acts[layer_idx] = acts[0, -1, :].float().cpu().numpy()

        return {
            'margin': margin,
            'neuron_activations': neuron_acts,
            'predicted': predicted_text,
            'ground_truth': ground_truth
        }


# ==============================================================================
# DATA LOADING
# ==============================================================================


def load_vqa_samples(n_samples: int = 200) -> List[Dict]:
    """Load VQA samples."""

    # Try to load from analysis records
    records_path = os.path.join(
        PARENT_DIR, "test_intervention_output/analysis_records.json")
    if os.path.exists(records_path):
        with open(records_path, 'r') as f:
            data = json.load(f)
        records = data.get("records", [])[:n_samples]
        print(f"Loaded {len(records)} samples from analysis_records.json")
        return records

    # Fallback to results.csv
    csv_path = os.path.join(PARENT_DIR, "results.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        samples = []
        for _, row in df.iterrows():
            samples.append({
                'question': row['question'],
                'ground_truth': row['ground_truth'],
                'image_url': row['image_url'],
                'question_type': row.get('question_type', 'other')
            })
            if len(samples) >= n_samples:
                break
        print(f"Loaded {len(samples)} samples from results.csv")
        return samples

    raise FileNotFoundError("No sample data found!")


def load_image(url_or_path: str) -> Image.Image:
    """Load image from URL or path."""
    try:
        if url_or_path.startswith('http'):
            response = requests.get(url_or_path, timeout=10)
            return Image.open(BytesIO(response.content)).convert("RGB")
        else:
            return Image.open(url_or_path).convert("RGB")
    except Exception as e:
        # Return blank image on error
        return Image.new("RGB", (224, 224), color=(128, 128, 128))


# ==============================================================================
# MAIN ANALYSIS FUNCTIONS
# ==============================================================================


def run_logit_lens_analysis(model_type: str, n_samples: int = 200):
    """Run full logit lens analysis for a model."""

    print("=" * 70)
    print(f"LOGIT LENS ANALYSIS: {model_type.upper()}")
    print(f"Samples: {n_samples}")
    print("=" * 70)

    # Create analyzer
    if model_type == "qwen2-vl":
        analyzer = Qwen2VLLogitLensAnalyzer()
    elif model_type == "paligemma":
        analyzer = PaliGemmaLogitLensAnalyzer()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    analyzer.load_model()
    analyzer.setup_hooks()

    # Load samples
    samples = load_vqa_samples(n_samples)

    # Run analysis WITH images
    print("\n" + "=" * 70)
    print("Running analysis WITH images...")
    print("=" * 70)

    results_with_image = []
    for sample in tqdm(samples, desc="With image"):
        try:
            image = load_image(sample['image_url'])
            result = analyzer.analyze_sample(
                image=image,
                question=sample['question'],
                ground_truth=sample['ground_truth'])
            result.question_type = sample.get('question_type', 'other')
            results_with_image.append(result)
        except Exception as e:
            print(f"Error: {e}")
            continue

    # Run analysis WITHOUT images (blank)
    print("\n" + "=" * 70)
    print("Running analysis WITHOUT images (blank)...")
    print("=" * 70)

    blank_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    results_no_image = []
    for sample in tqdm(samples, desc="Without image"):
        try:
            result = analyzer.analyze_sample(
                image=blank_image,
                question=sample['question'],
                ground_truth=sample['ground_truth'])
            results_no_image.append(result)
        except Exception as e:
            continue

    analyzer.clear_hooks()

    # Compute metrics
    print("\n" + "=" * 70)
    print("Computing metrics...")
    print("=" * 70)

    n_correct_with = sum(1 for r in results_with_image if r.is_correct)
    n_correct_no = sum(1 for r in results_no_image if r.is_correct)

    acc_with = n_correct_with / len(
        results_with_image) if results_with_image else 0
    acc_no = n_correct_no / len(results_no_image) if results_no_image else 0

    # Average margins
    margins_with = np.stack([r.layer_margins for r in results_with_image])
    margins_no = np.stack(
        [r.layer_margins for r in results_no_image[:len(results_with_image)]])

    avg_margins_with = margins_with.mean(axis=0)
    avg_margins_no = margins_no.mean(axis=0)
    delta_margins = avg_margins_with - avg_margins_no

    # MLP vs Attention
    mlp_contribs = np.stack([r.mlp_contributions for r in results_with_image])
    attn_contribs = np.stack(
        [r.attn_contributions for r in results_with_image])

    avg_mlp = mlp_contribs.mean(axis=0)
    avg_attn = attn_contribs.mean(axis=0)

    # Correct vs Incorrect
    correct_results = [r for r in results_with_image if r.is_correct]
    incorrect_results = [r for r in results_with_image if not r.is_correct]

    correct_margins = np.stack([r.layer_margins for r in correct_results
                                ]) if correct_results else None
    incorrect_margins = np.stack([r.layer_margins for r in incorrect_results
                                  ]) if incorrect_results else None

    # Find peak visual effect layer
    peak_layer = np.argmax(delta_margins)
    peak_effect = delta_margins[peak_layer]

    # Compile results
    results = {
        "model":
        model_type,
        "n_samples":
        len(results_with_image),
        "n_layers":
        analyzer.n_layers,
        "timestamp":
        datetime.now().isoformat(),
        "acc_with_image":
        float(acc_with),
        "acc_no_image":
        float(acc_no),
        "avg_margins_with_image":
        avg_margins_with.tolist(),
        "avg_margins_no_image":
        avg_margins_no.tolist(),
        "delta_margins":
        delta_margins.tolist(),
        "peak_visual_layer":
        int(peak_layer),
        "peak_visual_effect":
        float(peak_effect),
        "avg_mlp_contributions":
        avg_mlp.tolist(),
        "avg_attn_contributions":
        avg_attn.tolist(),
        "total_mlp_contribution":
        float(avg_mlp.sum()),
        "total_attn_contribution":
        float(avg_attn.sum()),
        "mlp_percent":
        float(avg_mlp.sum() / (avg_mlp.sum() + avg_attn.sum()) * 100),
        "attn_percent":
        float(avg_attn.sum() / (avg_mlp.sum() + avg_attn.sum()) * 100),
    }

    if correct_margins is not None and incorrect_margins is not None:
        results["avg_correct_margins"] = correct_margins.mean(axis=0).tolist()
        results["avg_incorrect_margins"] = incorrect_margins.mean(
            axis=0).tolist()
        results["n_correct"] = len(correct_results)
        results["n_incorrect"] = len(incorrect_results)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Model: {model_type}")
    print(f"Layers: {analyzer.n_layers}")
    print(f"Samples: {len(results_with_image)}")
    print(f"\nAccuracy WITH image: {acc_with:.1%}")
    print(f"Accuracy WITHOUT image: {acc_no:.1%}")
    print(
        f"\nPeak visual effect: Layer {peak_layer} (Δmargin = {peak_effect:.2f})"
    )
    print(f"\nMLP contribution: {results['mlp_percent']:.1f}%")
    print(f"Attention contribution: {results['attn_percent']:.1f}%")

    # Save results
    output_path = os.path.join(SCRIPT_DIR,
                               f"{model_type}_logit_lens_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


def run_neuron_analysis(model_type: str, n_samples: int = 200):
    """Run neuron-level analysis for a model."""

    print("\n" + "=" * 70)
    print(f"NEURON ANALYSIS: {model_type.upper()}")
    print("=" * 70)

    # Load model
    if model_type == "qwen2-vl":
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True)
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct",
                                                  trust_remote_code=True)
        n_layers = len(model.model.language_model.layers)
        # Target key layers (similar to LLaVA analysis)
        target_layers = [
            n_layers - 10, n_layers - 7, n_layers - 4, n_layers - 3,
            n_layers - 2, n_layers - 1
        ]
        target_layers = [l for l in target_layers if l >= 0]
    else:  # paligemma
        from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            "google/paligemma-3b-mix-224",
            torch_dtype=torch.bfloat16,
            device_map="auto")
        processor = AutoProcessor.from_pretrained(
            "google/paligemma-3b-mix-224")
        n_layers = len(model.language_model.layers)
        # For PaliGemma with 18 layers
        target_layers = [
            n_layers - 8, n_layers - 5, n_layers - 3, n_layers - 2,
            n_layers - 1
        ]
        target_layers = [l for l in target_layers if l >= 0]

    model.eval()

    # Create neuron analyzer
    analyzer = NeuronAnalyzer(model, processor, model_type)
    analyzer.setup_hooks(target_layers)

    print(f"Model: {model_type}, {n_layers} layers")
    print(f"Target layers: {target_layers}")

    # Load samples
    samples = load_vqa_samples(n_samples)

    # Collect neuron activations
    print("\nCollecting neuron activations...")

    all_activations = {l: [] for l in target_layers}
    margins = []  # Use margins instead of binary correctness

    for sample in tqdm(samples, desc="Processing"):
        try:
            image = load_image(sample['image_url'])
            result = analyzer.analyze_sample(image, sample['question'],
                                             sample['ground_truth'])

            margins.append(result['margin'])
            for layer_idx, acts in result['neuron_activations'].items():
                if layer_idx in all_activations:
                    all_activations[layer_idx].append(acts)
        except Exception as e:
            continue

    analyzer.clear_hooks()

    margins = np.array(margins)

    # Convert margins to binary correctness using median split
    # High margin = more "correct" behavior, low margin = less "correct"
    median_margin = np.median(margins)
    correctness = margins > median_margin
    n_correct = correctness.sum()
    n_incorrect = len(correctness) - n_correct

    print(f"\nProcessed {len(margins)} samples")
    print(
        f"Margin range: [{margins.min():.2f}, {margins.max():.2f}], median: {median_margin:.2f}"
    )
    print(
        f"High-margin samples: {n_correct}, Low-margin samples: {n_incorrect}")

    # Analyze neurons at each layer
    print("\n" + "=" * 70)
    print("NEURON ANALYSIS RESULTS")
    print("=" * 70)

    neuron_results = {}

    for layer_idx in target_layers:
        if layer_idx not in all_activations or len(
                all_activations[layer_idx]) == 0:
            continue

        X = np.stack(all_activations[layer_idx])
        y = correctness[:len(X)]

        if len(X) < 50:
            continue

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train sparse probe
        X_train, X_test, y_train, y_test = train_test_split(X_scaled,
                                                            y,
                                                            test_size=0.3,
                                                            random_state=42)

        probe = LogisticRegression(penalty='l1',
                                   solver='saga',
                                   max_iter=1000,
                                   C=0.1)
        probe.fit(X_train, y_train)

        # Evaluate
        train_acc = probe.score(X_train, y_train)
        test_acc = probe.score(X_test, y_test)

        # Find important neurons
        weights = probe.coef_[0]
        n_nonzero = np.sum(np.abs(weights) > 0.01)

        # Top success neurons (positive weight)
        top_success = np.argsort(weights)[-5:][::-1]

        # Top failure neurons (negative weight)
        top_failure = np.argsort(weights)[:5]

        # Compute activation differences
        correct_mask = y == 1
        incorrect_mask = y == 0

        if correct_mask.sum() > 0 and incorrect_mask.sum() > 0:
            correct_mean = X[correct_mask].mean(axis=0)
            incorrect_mean = X[incorrect_mask].mean(axis=0)
            diff = correct_mean - incorrect_mean

            # Top differentiating neurons
            top_diff = np.argsort(np.abs(diff))[-10:][::-1]
        else:
            diff = None
            top_diff = []

        neuron_results[layer_idx] = {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "n_nonzero_neurons": int(n_nonzero),
            "total_neurons": X.shape[1],
            "sparsity_percent": float(n_nonzero / X.shape[1] * 100),
            "top_success_neurons": top_success.tolist(),
            "top_failure_neurons": top_failure.tolist(),
            "top_diff_neurons": top_diff.tolist() if len(top_diff) > 0 else [],
        }

        print(f"\nLayer {layer_idx}:")
        print(f"  Test Accuracy: {test_acc:.1%}")
        print(
            f"  Non-zero neurons: {n_nonzero} / {X.shape[1]} ({n_nonzero/X.shape[1]*100:.1f}%)"
        )
        print(f"  Top high-margin neurons: {top_success[:3]}")
        print(f"  Top low-margin neurons: {top_failure[:3]}")

    # Compile final results
    final_results = {
        "model": model_type,
        "n_samples": len(margins),
        "n_high_margin": int(n_correct),
        "n_low_margin": int(n_incorrect),
        "margin_median": float(median_margin),
        "margin_min": float(margins.min()),
        "margin_max": float(margins.max()),
        "target_layers": target_layers,
        "layer_results": neuron_results,
        "timestamp": datetime.now().isoformat()
    }

    # Save
    output_path = os.path.join(SCRIPT_DIR,
                               f"{model_type}_neuron_analysis_results.json")
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Multi-model logit lens and neuron analysis")
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        choices=["qwen2-vl", "paligemma"],
                        help="Model to analyze")
    parser.add_argument("--analysis",
                        type=str,
                        default="both",
                        choices=["logit_lens", "neuron", "both"],
                        help="Which analysis to run")
    parser.add_argument("--n_samples",
                        type=int,
                        default=200,
                        help="Number of samples")

    args = parser.parse_args()

    if args.analysis in ["logit_lens", "both"]:
        run_logit_lens_analysis(args.model, args.n_samples)

    if args.analysis in ["neuron", "both"]:
        run_neuron_analysis(args.model, args.n_samples)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
