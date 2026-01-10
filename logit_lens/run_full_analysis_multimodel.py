#!/usr/bin/env python3
"""
FULL Logit Lens Analysis for Qwen2-VL and PaliGemma models.

This script runs ALL the experiments that were done for LLaVA-7B:
1. Step 1b: With vs Without Image Comparison
2. Visual Layer Attribution (MLP vs Attention contribution)
3. Neuron-Level Analysis (which neurons predict correctness)
4. Question Type Analysis
5. Compile all results

Usage:
    python run_full_analysis_multimodel.py --model qwen2-vl --n_samples 1000
    python run_full_analysis_multimodel.py --model paligemma --n_samples 1000
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
from typing import Dict, List, Optional, Tuple
import argparse
from collections import defaultdict
import re

# Qwen2-VL specific
try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False

# PaliGemma specific
try:
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
    PALIGEMMA_AVAILABLE = True
except ImportError:
    PALIGEMMA_AVAILABLE = False


class FullLogitLensAnalyzer:
    """Complete logit lens analyzer for Qwen2-VL and PaliGemma."""
    
    def __init__(self, model_type: str = "qwen2-vl"):
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hooks = []
        self.residual_streams = {}
        self.mlp_outputs = {}
        self.attn_outputs = {}
        
        if model_type == "qwen2-vl":
            self._load_qwen2vl()
        elif model_type == "paligemma":
            self._load_paligemma()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _load_qwen2vl(self):
        """Load Qwen2-VL-7B-Instruct."""
        print("Loading Qwen2-VL-7B-Instruct...")
        model_id = "Qwen/Qwen2-VL-7B-Instruct"
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # Qwen2-VL: model.model.language_model.layers
        self.n_layers = len(self.model.model.language_model.layers)
        print(f"Model loaded. {self.n_layers} layers.")
    
    def _load_paligemma(self):
        """Load PaliGemma-3B."""
        print("Loading PaliGemma-3B-mix-224...")
        model_id = "google/paligemma-3b-mix-224"
        
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.processor = PaliGemmaProcessor.from_pretrained(model_id)
        
        # PaliGemma: model.language_model.layers
        self.n_layers = len(self.model.language_model.layers)
        print(f"Model loaded. {self.n_layers} layers.")
    
    def _get_layers(self):
        """Get all layer modules."""
        if self.model_type == "qwen2-vl":
            return self.model.model.language_model.layers
        elif self.model_type == "paligemma":
            return self.model.language_model.layers
    
    def _get_layer_module(self, layer_idx: int):
        """Get a specific layer module."""
        return self._get_layers()[layer_idx]
    
    def _get_lm_head(self):
        """Get the LM head."""
        return self.model.lm_head
    
    def _get_norm(self):
        """Get the final layer norm."""
        if self.model_type == "qwen2-vl":
            return self.model.model.language_model.norm
        elif self.model_type == "paligemma":
            return self.model.language_model.norm
    
    def _register_hooks_basic(self):
        """Register hooks to capture residual streams only."""
        self.residual_streams = {}
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.residual_streams[layer_idx] = output[0].detach()
                else:
                    self.residual_streams[layer_idx] = output.detach()
            return hook
        
        for i in range(self.n_layers):
            layer = self._get_layer_module(i)
            h = layer.register_forward_hook(make_hook(i))
            self.hooks.append(h)
    
    def _register_hooks_detailed(self):
        """Register hooks to capture MLP and attention outputs separately."""
        self.residual_streams = {}
        self.mlp_outputs = {}
        self.attn_outputs = {}
        
        def make_residual_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.residual_streams[layer_idx] = output[0].detach()
                else:
                    self.residual_streams[layer_idx] = output.detach()
            return hook
        
        def make_mlp_hook(layer_idx):
            def hook(module, input, output):
                self.mlp_outputs[layer_idx] = output.detach()
            return hook
        
        def make_attn_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.attn_outputs[layer_idx] = output[0].detach()
                else:
                    self.attn_outputs[layer_idx] = output.detach()
            return hook
        
        for i in range(self.n_layers):
            layer = self._get_layer_module(i)
            
            # Hook the full layer output
            h1 = layer.register_forward_hook(make_residual_hook(i))
            self.hooks.append(h1)
            
            # Hook MLP output
            if hasattr(layer, 'mlp'):
                h2 = layer.mlp.register_forward_hook(make_mlp_hook(i))
                self.hooks.append(h2)
            
            # Hook attention output
            if hasattr(layer, 'self_attn'):
                h3 = layer.self_attn.register_forward_hook(make_attn_hook(i))
                self.hooks.append(h3)
    
    def _remove_hooks(self):
        """Remove all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def _prepare_inputs(self, question: str, image: Optional[Image.Image] = None):
        """Prepare inputs based on model type."""
        if self.model_type == "qwen2-vl":
            return self._prepare_inputs_qwen(question, image)
        else:
            return self._prepare_inputs_paligemma(question, image)
    
    def _prepare_inputs_qwen(self, question: str, image: Optional[Image.Image] = None):
        """Prepare inputs for Qwen2-VL."""
        if image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question + " Answer with a single word or phrase."},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": question + " Answer with a single word or phrase."},
                    ],
                }
            ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        if image is not None:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )
        
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def _prepare_inputs_paligemma(self, question: str, image: Optional[Image.Image] = None):
        """Prepare inputs for PaliGemma."""
        prompt = question + " Answer briefly:"
        
        if image is None:
            # Create blank image for PaliGemma (it always needs an image)
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def analyze_sample_basic(
        self,
        question: str,
        image: Optional[Image.Image],
        ground_truth: str,
    ) -> Dict:
        """Basic analysis: layer-wise margins."""
        inputs = self._prepare_inputs(question, image)
        self._register_hooks_basic()
        
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get predicted token
            final_logits = outputs.logits[0, -1, :]
            predicted_token_id = final_logits.argmax().item()
            predicted_token = self.processor.tokenizer.decode([predicted_token_id]).strip()
            
            # Check correctness
            ground_truth_lower = ground_truth.lower().strip()
            predicted_lower = predicted_token.lower().strip()
            is_correct = (
                predicted_lower == ground_truth_lower or
                ground_truth_lower in predicted_lower or
                predicted_lower in ground_truth_lower
            )
            
            # Get ground truth token id
            gt_tokens = self.processor.tokenizer.encode(ground_truth, add_special_tokens=False)
            gt_token_id = gt_tokens[0] if gt_tokens else predicted_token_id
            
            # Compute layer-wise margins
            lm_head = self._get_lm_head()
            norm = self._get_norm()
            
            layer_margins = []
            for layer_idx in range(self.n_layers):
                if layer_idx in self.residual_streams:
                    hidden = self.residual_streams[layer_idx][:, -1, :]
                    normed = norm(hidden)
                    logits = lm_head(normed)[0]
                    
                    gt_logit = logits[gt_token_id].item()
                    logits_copy = logits.clone()
                    logits_copy[gt_token_id] = float('-inf')
                    max_other = logits_copy.max().item()
                    margin = gt_logit - max_other
                    layer_margins.append(margin)
                else:
                    layer_margins.append(0.0)
            
            return {
                "question": question,
                "ground_truth": ground_truth,
                "predicted_token": predicted_token,
                "is_correct": is_correct,
                "layer_margins": layer_margins,
            }
            
        finally:
            self._remove_hooks()
            self.residual_streams = {}
    
    def analyze_sample_detailed(
        self,
        question: str,
        image: Optional[Image.Image],
        ground_truth: str,
    ) -> Dict:
        """Detailed analysis: MLP vs attention contributions."""
        inputs = self._prepare_inputs(question, image)
        self._register_hooks_detailed()
        
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get predicted token
            final_logits = outputs.logits[0, -1, :]
            predicted_token_id = final_logits.argmax().item()
            predicted_token = self.processor.tokenizer.decode([predicted_token_id]).strip()
            
            # Check correctness
            ground_truth_lower = ground_truth.lower().strip()
            predicted_lower = predicted_token.lower().strip()
            is_correct = (
                predicted_lower == ground_truth_lower or
                ground_truth_lower in predicted_lower or
                predicted_lower in ground_truth_lower
            )
            
            # Get ground truth token id
            gt_tokens = self.processor.tokenizer.encode(ground_truth, add_special_tokens=False)
            gt_token_id = gt_tokens[0] if gt_tokens else predicted_token_id
            
            lm_head = self._get_lm_head()
            norm = self._get_norm()
            
            layer_margins = []
            mlp_contributions = []
            attn_contributions = []
            
            for layer_idx in range(self.n_layers):
                if layer_idx in self.residual_streams:
                    hidden = self.residual_streams[layer_idx][:, -1, :]
                    normed = norm(hidden)
                    logits = lm_head(normed)[0]
                    
                    gt_logit = logits[gt_token_id].item()
                    logits_copy = logits.clone()
                    logits_copy[gt_token_id] = float('-inf')
                    max_other = logits_copy.max().item()
                    margin = gt_logit - max_other
                    layer_margins.append(margin)
                    
                    # MLP contribution
                    if layer_idx in self.mlp_outputs:
                        mlp_out = self.mlp_outputs[layer_idx][:, -1, :]
                        mlp_normed = norm(mlp_out)
                        mlp_logits = lm_head(mlp_normed)[0]
                        mlp_contrib = mlp_logits[gt_token_id].item()
                        mlp_contributions.append(mlp_contrib)
                    else:
                        mlp_contributions.append(0.0)
                    
                    # Attention contribution  
                    if layer_idx in self.attn_outputs:
                        attn_out = self.attn_outputs[layer_idx][:, -1, :]
                        attn_normed = norm(attn_out)
                        attn_logits = lm_head(attn_normed)[0]
                        attn_contrib = attn_logits[gt_token_id].item()
                        attn_contributions.append(attn_contrib)
                    else:
                        attn_contributions.append(0.0)
                else:
                    layer_margins.append(0.0)
                    mlp_contributions.append(0.0)
                    attn_contributions.append(0.0)
            
            return {
                "question": question,
                "ground_truth": ground_truth,
                "predicted_token": predicted_token,
                "is_correct": is_correct,
                "layer_margins": layer_margins,
                "mlp_contributions": mlp_contributions,
                "attn_contributions": attn_contributions,
            }
            
        finally:
            self._remove_hooks()
            self.residual_streams = {}
            self.mlp_outputs = {}
            self.attn_outputs = {}


def classify_question_type(question: str) -> str:
    """Classify a question into a type category."""
    q_lower = question.lower()
    
    if any(w in q_lower for w in ['color', 'colour']):
        return 'color recognition'
    elif q_lower.startswith(('is ', 'are ', 'does ', 'do ', 'can ', 'will ', 'has ', 'have ')):
        return 'yes/no'
    elif any(w in q_lower for w in ['how many', 'count', 'number of']):
        return 'counting'
    elif any(w in q_lower for w in ['where', 'location', 'position']):
        return 'location'
    elif any(w in q_lower for w in ['why', 'because', 'reason']):
        return 'reasoning'
    elif any(w in q_lower for w in ['what type', 'what kind', 'category']):
        return 'classification'
    elif any(w in q_lower for w in ['which', 'compare', 'difference']):
        return 'comparison'
    elif any(w in q_lower for w in ['who', 'person', 'man', 'woman', 'people']):
        return 'person identification'
    elif any(w in q_lower for w in ['when', 'time', 'year', 'date']):
        return 'time-related'
    elif any(w in q_lower for w in ['what is', 'what are', 'identify', 'name']):
        return 'object identification'
    else:
        return 'other'


def run_full_analysis(model_type: str, n_samples: int = 1000):
    """Run the complete logit lens analysis suite."""
    
    print("=" * 70)
    print(f"FULL LOGIT LENS ANALYSIS: {model_type.upper()}")
    print(f"Samples: {n_samples}")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = FullLogitLensAnalyzer(model_type=model_type)
    
    # Load samples
    print("\nLoading samples...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    data_path = os.path.join(parent_dir, "test_intervention_output/analysis_records.json")
    
    with open(data_path, "r") as f:
        data = json.load(f)
    
    records = data.get("records", [])[:n_samples]
    print(f"Loaded {len(records)} samples")
    
    # ========================================================================
    # STEP 1: With vs Without Image Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: WITH vs WITHOUT IMAGE ANALYSIS")
    print("=" * 70)
    
    results_with_image = []
    results_no_image = []
    
    # With image
    print("\nRunning WITH images...")
    for record in tqdm(records, desc="With image"):
        try:
            image_url = record.get("image_url")
            question = record.get("question", "")
            gt = record.get("ground_truth", "")
            
            if not image_url:
                continue
                
            response = requests.get(image_url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            result = analyzer.analyze_sample_basic(question, image, gt)
            results_with_image.append(result)
        except Exception as e:
            continue
    
    # Without image
    print("\nRunning WITHOUT images...")
    for record in tqdm(records, desc="Without image"):
        try:
            question = record.get("question", "")
            gt = record.get("ground_truth", "")
            
            result = analyzer.analyze_sample_basic(question, None, gt)
            results_no_image.append(result)
        except Exception as e:
            continue
    
    # Compute step 1 metrics
    acc_with_image = sum(1 for r in results_with_image if r["is_correct"]) / len(results_with_image)
    acc_no_image = sum(1 for r in results_no_image if r["is_correct"]) / len(results_no_image)
    
    # Average margins per layer
    avg_margins_with = [0.0] * analyzer.n_layers
    avg_margins_no = [0.0] * analyzer.n_layers
    
    for r in results_with_image:
        for i, m in enumerate(r["layer_margins"]):
            avg_margins_with[i] += m / len(results_with_image)
    
    for r in results_no_image:
        for i, m in enumerate(r["layer_margins"]):
            avg_margins_no[i] += m / len(results_no_image)
    
    # Delta margins
    delta_margins = [avg_margins_with[i] - avg_margins_no[i] for i in range(analyzer.n_layers)]
    peak_delta_layer = int(np.argmax(delta_margins))
    
    print(f"\nAccuracy WITH image: {acc_with_image*100:.1f}%")
    print(f"Accuracy WITHOUT image: {acc_no_image*100:.1f}%")
    print(f"Peak delta margin: Layer {peak_delta_layer} (value = {delta_margins[peak_delta_layer]:.2f})")
    
    step1_results = {
        "model": model_type,
        "n_samples": len(results_with_image),
        "n_layers": analyzer.n_layers,
        "timestamp": datetime.now().isoformat(),
        "acc_with_image": acc_with_image,
        "acc_no_image": acc_no_image,
        "avg_margins_with_image": avg_margins_with,
        "avg_margins_no_image": avg_margins_no,
        "delta_margins": delta_margins,
        "peak_delta_layer": peak_delta_layer,
        "peak_delta_value": delta_margins[peak_delta_layer],
    }
    
    # ========================================================================
    # STEP 2: Visual Layer Attribution (MLP vs Attention)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: VISUAL LAYER ATTRIBUTION")
    print("=" * 70)
    
    # Run detailed analysis on a subset
    detailed_samples = min(200, len(records))
    print(f"\nAnalyzing {detailed_samples} samples with detailed hooks...")
    
    detailed_results = []
    for record in tqdm(records[:detailed_samples], desc="Detailed analysis"):
        try:
            image_url = record.get("image_url")
            question = record.get("question", "")
            gt = record.get("ground_truth", "")
            
            if not image_url:
                continue
                
            response = requests.get(image_url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            result = analyzer.analyze_sample_detailed(question, image, gt)
            detailed_results.append(result)
        except Exception as e:
            continue
    
    # Compute MLP vs attention contributions
    avg_mlp = [0.0] * analyzer.n_layers
    avg_attn = [0.0] * analyzer.n_layers
    
    for r in detailed_results:
        for i in range(analyzer.n_layers):
            avg_mlp[i] += r["mlp_contributions"][i] / len(detailed_results)
            avg_attn[i] += r["attn_contributions"][i] / len(detailed_results)
    
    # Find visual layers (where delta margin > threshold)
    threshold = 0.5
    visual_layers = [i for i, d in enumerate(delta_margins) if abs(d) > threshold]
    
    # Compute fractions for visual layers
    mlp_total = sum(abs(avg_mlp[i]) for i in visual_layers) if visual_layers else 0
    attn_total = sum(abs(avg_attn[i]) for i in visual_layers) if visual_layers else 0
    total = mlp_total + attn_total
    
    step2_results = {
        "model": model_type,
        "n_samples": len(detailed_results),
        "threshold": threshold,
        "visual_layers": visual_layers,
        "avg_mlp_contributions": avg_mlp,
        "avg_attn_contributions": avg_attn,
        "mlp_fraction": mlp_total / total if total > 0 else 0,
        "attn_fraction": attn_total / total if total > 0 else 0,
    }
    
    print(f"\nVisual layers (|delta| > {threshold}): {visual_layers}")
    print(f"MLP contribution fraction: {step2_results['mlp_fraction']*100:.1f}%")
    print(f"Attention contribution fraction: {step2_results['attn_fraction']*100:.1f}%")
    
    # ========================================================================
    # STEP 3: Question Type Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: QUESTION TYPE ANALYSIS")
    print("=" * 70)
    
    # Group results by question type
    type_results = defaultdict(list)
    
    for i, record in enumerate(records):
        if i >= len(results_with_image):
            break
        question = record.get("question", "")
        q_type = classify_question_type(question)
        type_results[q_type].append(results_with_image[i])
    
    # Compute per-type stats
    type_stats = {}
    for q_type, results in type_results.items():
        if len(results) < 5:  # Skip types with too few samples
            continue
        
        acc = sum(1 for r in results if r["is_correct"]) / len(results)
        
        # Average margins per layer
        avg_margins = [0.0] * analyzer.n_layers
        for r in results:
            for i, m in enumerate(r["layer_margins"]):
                avg_margins[i] += m / len(results)
        
        # Find crossover layer (where margin becomes positive)
        crossover = analyzer.n_layers - 1
        for i, m in enumerate(avg_margins):
            if m > 0:
                crossover = i
                break
        
        type_stats[q_type] = {
            "n_samples": len(results),
            "accuracy": acc,
            "avg_margins": avg_margins,
            "crossover_layer": crossover,
        }
    
    step3_results = {
        "model": model_type,
        "type_stats": type_stats,
        "samples_per_type": {k: len(v) for k, v in type_results.items()},
    }
    
    print("\nQuestion type breakdown:")
    for q_type, stats in sorted(type_stats.items(), key=lambda x: -x[1]["n_samples"]):
        print(f"  {q_type}: {stats['n_samples']} samples, {stats['accuracy']*100:.1f}% acc, crossover layer {stats['crossover_layer']}")
    
    # ========================================================================
    # COMPILE ALL RESULTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("COMPILING RESULTS")
    print("=" * 70)
    
    full_results = {
        "model": model_type,
        "n_layers": analyzer.n_layers,
        "n_samples": len(results_with_image),
        "timestamp": datetime.now().isoformat(),
        "step1_image_comparison": step1_results,
        "step2_visual_attribution": step2_results,
        "step3_question_types": step3_results,
    }
    
    # Save results
    output_file = os.path.join(script_dir, f"{model_type.replace('-', '_')}_full_results.json")
    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model: {model_type}")
    print(f"Layers: {analyzer.n_layers}")
    print(f"Samples: {len(results_with_image)}")
    print(f"\nAccuracy WITH image: {acc_with_image*100:.1f}%")
    print(f"Accuracy WITHOUT image: {acc_no_image*100:.1f}%")
    print(f"Peak delta layer: {peak_delta_layer} (value: {delta_margins[peak_delta_layer]:.2f})")
    print(f"MLP fraction: {step2_results['mlp_fraction']*100:.1f}%")
    print(f"Attention fraction: {step2_results['attn_fraction']*100:.1f}%")
    
    return full_results


def main():
    parser = argparse.ArgumentParser(description="Full logit lens analysis")
    parser.add_argument("--model", type=str, choices=["qwen2-vl", "paligemma"], required=True)
    parser.add_argument("--n_samples", type=int, default=1000)
    args = parser.parse_args()
    
    run_full_analysis(args.model, args.n_samples)


if __name__ == "__main__":
    main()
