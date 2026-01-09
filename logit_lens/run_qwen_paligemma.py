#!/usr/bin/env python3
"""
Logit Lens Analysis for Qwen2-VL and PaliGemma models.

These models have different architectures from LLaVA, requiring custom handling.
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


class MultiModelLogitLens:
    """Logit lens analyzer that works with Qwen2-VL and PaliGemma."""
    
    def __init__(self, model_type: str = "qwen2-vl"):
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hooks = []
        self.residual_streams = {}
        
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
        
        # Get number of layers - Qwen2-VL uses model.language_model.layers
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
        
        # Get number of layers - PaliGemma uses language_model.layers (GemmaModel)
        self.n_layers = len(self.model.language_model.layers)
        print(f"Model loaded. {self.n_layers} layers.")
    
    def _get_layer_module(self, layer_idx: int):
        """Get the layer module for the current model type."""
        if self.model_type == "qwen2-vl":
            return self.model.model.language_model.layers[layer_idx]
        elif self.model_type == "paligemma":
            return self.model.language_model.layers[layer_idx]
    
    def _get_lm_head(self):
        """Get the LM head for the current model type."""
        if self.model_type == "qwen2-vl":
            return self.model.lm_head
        elif self.model_type == "paligemma":
            return self.model.lm_head  # lm_head is on main model, not language_model
    
    def _get_norm(self):
        """Get the final layer norm."""
        if self.model_type == "qwen2-vl":
            return self.model.model.language_model.norm
        elif self.model_type == "paligemma":
            return self.model.language_model.norm
    
    def _register_hooks(self):
        """Register hooks to capture residual streams."""
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
    
    def _remove_hooks(self):
        """Remove all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
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
    
    def analyze_sample(
        self,
        question: str,
        image: Optional[Image.Image],
        ground_truth: str,
    ) -> Dict:
        """Analyze a single sample with logit lens."""
        
        # Prepare inputs based on model type
        if self.model_type == "qwen2-vl":
            inputs = self._prepare_inputs_qwen(question, image)
        else:
            inputs = self._prepare_inputs_paligemma(question, image)
        
        # Register hooks
        self._register_hooks()
        
        try:
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get the predicted token
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
            
            # Compute layer-wise logits using logit lens
            lm_head = self._get_lm_head()
            norm = self._get_norm()
            
            layer_logits = []
            layer_margins = []
            
            for layer_idx in range(self.n_layers):
                if layer_idx in self.residual_streams:
                    hidden = self.residual_streams[layer_idx][:, -1, :]  # Last token
                    
                    # Apply layer norm and project to vocab
                    normed = norm(hidden)
                    logits = lm_head(normed)[0]
                    
                    # Get logit for ground truth token
                    gt_logit = logits[gt_token_id].item()
                    
                    # Compute margin (gt_logit - max_other)
                    logits_copy = logits.clone()
                    logits_copy[gt_token_id] = float('-inf')
                    max_other = logits_copy.max().item()
                    margin = gt_logit - max_other
                    
                    layer_logits.append(gt_logit)
                    layer_margins.append(margin)
                else:
                    layer_logits.append(0.0)
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


def run_analysis(model_type: str, n_samples: int = 1000):
    """Run logit lens analysis for the specified model."""
    
    print("=" * 70)
    print(f"{model_type.upper()} LOGIT LENS ANALYSIS")
    print(f"Samples: {n_samples}")
    print("=" * 70)
    
    # Load analyzer
    analyzer = MultiModelLogitLens(model_type=model_type)
    
    # Load samples
    print("\nLoading samples...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    data_path = os.path.join(parent_dir, "test_intervention_output/analysis_records.json")
    
    with open(data_path, "r") as f:
        data = json.load(f)
    
    records = data.get("records", [])[:n_samples]
    print(f"Loaded {len(records)} samples")
    
    # Run with images
    print("\n" + "=" * 70)
    print("Running analysis WITH images...")
    print("=" * 70)
    
    results_with_image = []
    for record in tqdm(records, desc="With image"):
        try:
            response = requests.get(record["image_url"], timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            result = analyzer.analyze_sample(
                question=record["question"],
                image=image,
                ground_truth=record["ground_truth"]
            )
            results_with_image.append(result)
        except Exception as e:
            continue
    
    # Run without images
    print("\n" + "=" * 70)
    print("Running analysis WITHOUT images...")
    print("=" * 70)
    
    results_no_image = []
    for record in tqdm(records, desc="Without image"):
        try:
            result = analyzer.analyze_sample(
                question=record["question"],
                image=None,
                ground_truth=record["ground_truth"]
            )
            results_no_image.append(result)
        except Exception as e:
            continue
    
    # Compute metrics
    print("\n" + "=" * 70)
    print("Computing metrics...")
    print("=" * 70)
    
    acc_with_image = np.mean([r["is_correct"] for r in results_with_image])
    acc_no_image = np.mean([r["is_correct"] for r in results_no_image])
    
    # Average margins per layer
    n_layers = analyzer.n_layers
    avg_margins_with = np.zeros(n_layers)
    avg_margins_no = np.zeros(n_layers)
    
    for r in results_with_image:
        avg_margins_with += np.array(r["layer_margins"])
    avg_margins_with /= len(results_with_image)
    
    for r in results_no_image:
        avg_margins_no += np.array(r["layer_margins"])
    avg_margins_no /= len(results_no_image)
    
    # Delta margins (with - without)
    delta_margins = avg_margins_with - avg_margins_no
    
    print(f"\nAccuracy WITH image: {acc_with_image:.1%}")
    print(f"Accuracy WITHOUT image: {acc_no_image:.1%}")
    print(f"\nPeak delta margin: Layer {np.argmax(delta_margins)} (value = {delta_margins.max():.2f})")
    
    # Save results
    output = {
        "model": model_type,
        "n_samples": n_samples,
        "n_layers": n_layers,
        "timestamp": datetime.now().isoformat(),
        "acc_with_image": float(acc_with_image),
        "acc_no_image": float(acc_no_image),
        "avg_margins_with_image": avg_margins_with.tolist(),
        "avg_margins_no_image": avg_margins_no.tolist(),
        "delta_margins": delta_margins.tolist(),
    }
    
    output_file = os.path.join(script_dir, f"{model_type.replace('-', '_')}_results.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2-vl",
                       choices=["qwen2-vl", "paligemma"],
                       help="Model to analyze")
    parser.add_argument("--n_samples", type=int, default=1000,
                       help="Number of samples to analyze")
    args = parser.parse_args()
    
    run_analysis(args.model, args.n_samples)
