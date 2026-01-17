"""
Multi-Model Logit Lens Analyzer

A model-agnostic version of the logit lens analyzer that works with:
- LLaVA 1.5 (7B, 13B)
- Qwen2-VL
- PaliGemma 2

Usage:
    python multi_model_analyzer.py --model llava-1.5-7b --dataset vqav2 --n_samples 1000
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from io import BytesIO

import torch
import numpy as np
from PIL import Image
import requests
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_configs import get_model_config, get_dataset_config, get_hook_config, MODELS


@dataclass
class LogitLensResult:
    """Result from logit lens analysis on a single sample."""
    question: str
    ground_truth: str
    predicted_token: str
    predicted_token_id: int
    is_correct: bool
    layer_logits: np.ndarray
    layer_margins: np.ndarray
    attn_contributions: Optional[np.ndarray] = None
    mlp_contributions: Optional[np.ndarray] = None
    question_type: Optional[str] = None


class MultiModelLogitLensAnalyzer:
    """
    Model-agnostic logit lens analyzer for vision-language models.
    
    Supports multiple architectures by using configuration-based hook registration.
    """

    def __init__(
        self,
        model_name: str = "llava-1.5-7b",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype

        # Get model configuration
        self.config = get_model_config(model_name)
        self.hook_config = get_hook_config(self.config["architecture"])

        # Storage for hooked activations
        self.residual_streams: Dict[int, torch.Tensor] = {}
        self.attn_outputs: Dict[int, torch.Tensor] = {}
        self.mlp_outputs: Dict[int, torch.Tensor] = {}
        self.hooks: List[Any] = []

        # Load model
        self._load_model()

    def _load_model(self):
        """Load the model and processor based on configuration."""
        print(f"Loading {self.config['model_id']}...")

        model_id = self.config["model_id"]
        architecture = self.config["architecture"]

        if architecture == "llava":
            from transformers import LlavaForConditionalGeneration, AutoProcessor
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=self.dtype,
            )
            self.processor = AutoProcessor.from_pretrained(model_id)

        elif architecture == "qwen2_vl":
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=self.dtype,
            )
            self.processor = AutoProcessor.from_pretrained(model_id)

        elif architecture == "paligemma":
            from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=self.dtype,
            )
            self.processor = AutoProcessor.from_pretrained(model_id)

        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        self.model.eval()

        # Get LM head
        self.lm_head = self._get_lm_head()
        self.n_layers = self.config["n_layers"]

        print(
            f"Model loaded. {self.n_layers} layers, architecture: {architecture}"
        )

    def _get_lm_head(self) -> torch.nn.Module:
        """Get the LM head (unembedding matrix) based on architecture."""
        architecture = self.config["architecture"]

        if architecture == "llava":
            return self.model.language_model.lm_head
        elif architecture == "qwen2_vl":
            return self.model.lm_head
        elif architecture == "paligemma":
            # PaliGemma: lm_head is at model level, not language_model
            return self.model.lm_head
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def _get_layer_module(self, layer_idx: int) -> torch.nn.Module:
        """Get the transformer layer module based on architecture."""
        architecture = self.config["architecture"]

        if architecture == "llava":
            return self.model.language_model.model.layers[layer_idx]
        elif architecture == "qwen2_vl":
            # Qwen2-VL has layers at model.model.layers
            return self.model.model.layers[layer_idx]
        elif architecture == "paligemma":
            # PaliGemma: layers at model.language_model.layers (no .model)
            return self.model.language_model.layers[layer_idx]
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def _register_hooks(self):
        """Register forward hooks to capture residual stream, attention, and MLP outputs."""
        self.residual_streams.clear()
        self.attn_outputs.clear()
        self.mlp_outputs.clear()

        for layer_idx in range(self.n_layers):
            layer = self._get_layer_module(layer_idx)

            # Hook for residual stream after attention + layernorm
            def make_residual_hook(idx):

                def hook(module, input, output):
                    if isinstance(output, tuple):
                        self.residual_streams[idx] = output[0].detach()
                    else:
                        self.residual_streams[idx] = output.detach()

                return hook

            h_res = layer.post_attention_layernorm.register_forward_hook(
                make_residual_hook(layer_idx))
            self.hooks.append(h_res)

            # Hook for attention output
            def make_attn_hook(idx):

                def hook(module, input, output):
                    if isinstance(output, tuple):
                        self.attn_outputs[idx] = output[0].detach()
                    else:
                        self.attn_outputs[idx] = output.detach()

                return hook

            h_attn = layer.self_attn.register_forward_hook(
                make_attn_hook(layer_idx))
            self.hooks.append(h_attn)

            # Hook for MLP output
            def make_mlp_hook(idx):

                def hook(module, input, output):
                    self.mlp_outputs[idx] = output.detach()

                return hook

            h_mlp = layer.mlp.register_forward_hook(make_mlp_hook(layer_idx))
            self.hooks.append(h_mlp)

    def _clear_hooks(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def _apply_lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the LM head to get logits from hidden states."""
        hidden = hidden_states.to(self.lm_head.weight.device).to(
            self.lm_head.weight.dtype)
        return self.lm_head(hidden)

    def _get_margin(self,
                    logits: torch.Tensor,
                    correct_token_id: int,
                    position: int = -1) -> float:
        """Get the margin: logit(correct) - max(logit(other))."""
        correct_logit = logits[0, position, correct_token_id].item()

        all_logits = logits[0, position, :].clone()
        all_logits[correct_token_id] = float('-inf')
        max_other = all_logits.max().item()

        return correct_logit - max_other

    def _format_prompt(self, question: str) -> str:
        """Format the prompt based on architecture."""
        architecture = self.config["architecture"]

        if architecture == "llava":
            return f"USER: <image>\n{question}\nAnswer with a single word or phrase.\nASSISTANT:"
        elif architecture == "qwen2_vl":
            return f"<|im_start|>user\n<image>\n{question}<|im_end|>\n<|im_start|>assistant\n"
        elif architecture == "paligemma":
            return f"{question}"
        else:
            return question

    def _check_correctness(self, predicted: str, ground_truth: str) -> bool:
        """Check if prediction matches ground truth (lenient matching)."""
        pred_lower = predicted.lower().strip()
        gt_lower = ground_truth.lower().strip()

        return (pred_lower == gt_lower or gt_lower.startswith(pred_lower)
                or pred_lower.startswith(gt_lower) or gt_lower in pred_lower
                or pred_lower in gt_lower)

    def analyze_sample(
        self,
        image: Image.Image,
        question: str,
        ground_truth: str,
        decompose_components: bool = True,
        question_type: Optional[str] = None,
    ) -> LogitLensResult:
        """Run logit lens analysis on a single VQA sample."""

        prompt = self._format_prompt(question)

        # Process inputs
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        self._register_hooks()

        try:
            with torch.no_grad():
                # Generate one token
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    output_hidden_states=False,
                    return_dict_in_generate=True,
                )

                generated_ids = outputs.sequences[0]
                input_len = inputs['input_ids'].shape[1]
                predicted_token_id = generated_ids[input_len].item()
                predicted_token = self.processor.tokenizer.decode(
                    [predicted_token_id]).strip()

                is_correct = self._check_correctness(predicted_token,
                                                     ground_truth)

                # Run forward pass to collect activations
                self.residual_streams.clear()
                self.attn_outputs.clear()
                self.mlp_outputs.clear()

                _ = self.model(**inputs, output_hidden_states=True)

                # Collect per-layer margins
                layer_logits = np.zeros(self.n_layers)
                layer_margins = np.zeros(self.n_layers)

                for layer_idx in range(self.n_layers):
                    if layer_idx in self.residual_streams:
                        residual = self.residual_streams[layer_idx]
                        logits = self._apply_lm_head(residual)

                        layer_logits[layer_idx] = logits[
                            0, -1, predicted_token_id].item()
                        layer_margins[layer_idx] = self._get_margin(
                            logits, predicted_token_id)

                # Component decomposition
                attn_contributions = None
                mlp_contributions = None

                if decompose_components:
                    attn_contributions = np.zeros(self.n_layers)
                    mlp_contributions = np.zeros(self.n_layers)

                    unembed_vec = self.lm_head.weight[
                        predicted_token_id, :].detach()
                    unembed_device = unembed_vec.device

                    for layer_idx in range(self.n_layers):
                        if layer_idx in self.attn_outputs:
                            attn_out = self.attn_outputs[layer_idx][0, -1, :]
                            attn_out = attn_out.to(unembed_device).to(
                                unembed_vec.dtype)
                            attn_contributions[layer_idx] = torch.dot(
                                attn_out, unembed_vec).item()

                        if layer_idx in self.mlp_outputs:
                            mlp_out = self.mlp_outputs[layer_idx][0, -1, :]
                            mlp_out = mlp_out.to(unembed_device).to(
                                unembed_vec.dtype)
                            mlp_contributions[layer_idx] = torch.dot(
                                mlp_out, unembed_vec).item()

                return LogitLensResult(
                    question=question,
                    ground_truth=ground_truth,
                    predicted_token=predicted_token,
                    predicted_token_id=predicted_token_id,
                    is_correct=is_correct,
                    layer_logits=layer_logits,
                    layer_margins=layer_margins,
                    attn_contributions=attn_contributions,
                    mlp_contributions=mlp_contributions,
                    question_type=question_type,
                )

        finally:
            self._clear_hooks()
            torch.cuda.empty_cache()

    def analyze_batch(
        self,
        samples: List[Dict],
        decompose_components: bool = True,
    ) -> List[LogitLensResult]:
        """Analyze multiple samples."""
        results = []

        for i, sample in enumerate(tqdm(samples, desc="Analyzing samples")):
            try:
                if isinstance(sample['image'], str):
                    if sample['image'].startswith('http'):
                        response = requests.get(sample['image'])
                        image = Image.open(BytesIO(
                            response.content)).convert('RGB')
                    else:
                        image = Image.open(sample['image']).convert('RGB')
                else:
                    image = sample['image']

                result = self.analyze_sample(
                    image=image,
                    question=sample['question'],
                    ground_truth=sample['ground_truth'],
                    decompose_components=decompose_components,
                    question_type=sample.get('question_type'),
                )
                results.append(result)

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

        return results

    def summarize_results(self, results: List[LogitLensResult]) -> Dict:
        """Summarize logit lens results."""
        if not results:
            return {}

        all_margins = np.stack([r.layer_margins for r in results])

        summary = {
            'model': self.model_name,
            'architecture': self.config['architecture'],
            'n_layers': self.n_layers,
            'n_samples': len(results),
            'n_correct': sum(r.is_correct for r in results),
            'accuracy': sum(r.is_correct for r in results) / len(results),
            'avg_margins': all_margins.mean(axis=0).tolist(),
            'std_margins': all_margins.std(axis=0).tolist(),
        }

        # MLP vs Attention breakdown
        if results[0].attn_contributions is not None:
            all_attn = np.stack([r.attn_contributions for r in results])
            all_mlp = np.stack([r.mlp_contributions for r in results])

            summary['avg_attn_contributions'] = all_attn.mean(axis=0).tolist()
            summary['avg_mlp_contributions'] = all_mlp.mean(axis=0).tolist()

            total_attn = all_attn.sum()
            total_mlp = all_mlp.sum()
            total = abs(total_attn) + abs(total_mlp)

            summary['overall_attn_fraction'] = abs(
                total_attn) / total if total > 0 else 0
            summary['overall_mlp_fraction'] = abs(
                total_mlp) / total if total > 0 else 0

        return summary


def load_dataset_samples(dataset_name: str,
                         n_samples: int = 1000) -> List[Dict]:
    """Load samples from a dataset."""
    from datasets import load_dataset

    config = get_dataset_config(dataset_name)

    print(f"Loading {config['name']}...")

    if dataset_name == "vqav2":
        # Load from existing analysis records if available
        records_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'test_intervention_output/analysis_records.json')

        if os.path.exists(records_path):
            with open(records_path, 'r') as f:
                data = json.load(f)
            records = data.get('records', [])[:n_samples]

            samples = []
            for r in records:
                samples.append({
                    'image':
                    r['image_url'],
                    'question':
                    r['question'],
                    'ground_truth':
                    r['ground_truth'],
                    'question_type':
                    r.get('question_type', 'other'),
                })
            return samples
        else:
            # Load from HuggingFace
            ds = load_dataset(config['hf_path'], split=config['split'])
            ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

            samples = []
            for item in ds:
                samples.append({
                    'image':
                    item[config['image_key']],
                    'question':
                    item[config['question_key']],
                    'ground_truth':
                    item[config['answer_key']][0] if isinstance(
                        item[config['answer_key']], list) else
                    item[config['answer_key']],
                })
            return samples

    elif dataset_name == "pope":
        ds = load_dataset(config['hf_path'], split=config['split'])
        ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

        samples = []
        for item in ds:
            samples.append({
                'image': item[config['image_key']],
                'question': item[config['question_key']],
                'ground_truth': item[config['answer_key']],
                'question_type': 'hallucination',
            })
        return samples

    elif dataset_name == "coco_captions":
        ds = load_dataset(config['hf_path'], split=config['split'])
        ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

        samples = []
        for item in ds:
            # For captioning, use "Describe this image" as question
            samples.append({
                'image':
                item[config['image_key']],
                'question':
                'Describe this image briefly.',
                'ground_truth':
                item[config['caption_key']][0]['raw'] if isinstance(
                    item[config['caption_key']], list) else
                item[config['caption_key']],
                'question_type':
                'captioning',
            })
        return samples

    else:
        raise ValueError(
            f"Dataset loading not implemented for: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-model logit lens analysis")
    parser.add_argument("--model",
                        type=str,
                        default="llava-1.5-7b",
                        choices=list(MODELS.keys()),
                        help="Model to analyze")
    parser.add_argument("--dataset",
                        type=str,
                        default="vqav2",
                        help="Dataset to use")
    parser.add_argument("--n_samples",
                        type=int,
                        default=1000,
                        help="Number of samples")
    parser.add_argument("--output_dir",
                        type=str,
                        default="./results",
                        help="Output directory")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    analyzer = MultiModelLogitLensAnalyzer(model_name=args.model)

    # Load samples
    samples = load_dataset_samples(args.dataset, args.n_samples)
    print(f"Loaded {len(samples)} samples from {args.dataset}")

    # Run analysis
    results = analyzer.analyze_batch(samples)

    # Summarize
    summary = analyzer.summarize_results(results)

    # Save results
    output_file = os.path.join(args.output_dir,
                               f"{args.model}_{args.dataset}_results.json")
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print(f"\nSummary:")
    print(f"  Model: {summary['model']}")
    print(f"  Accuracy: {summary['accuracy']:.1%}")
    print(f"  MLP fraction: {summary.get('overall_mlp_fraction', 0):.1%}")
    print(
        f"  Attention fraction: {summary.get('overall_attn_fraction', 0):.1%}")


if __name__ == "__main__":
    main()
