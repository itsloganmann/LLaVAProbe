"""
Logit Lens Analysis for LLaVA

This implements the logit lens technique to analyze where in the language model
the answer decision is made, and whether MLPs contribute more than attention.

Based on: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

Key hypotheses to test:
1. Layer location: Answer logit grows mainly in middle/late LM layers
2. Component type: MLP sublayers contribute more than attention sublayers
3. (Future) Neuron level: Specific MLP neurons are task-specific
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import requests
from io import BytesIO
import json
import warnings

warnings.filterwarnings("ignore")


@dataclass
class LogitLensResult:
    """Results from logit lens analysis for a single sample."""
    question: str
    ground_truth: str
    predicted_token: str
    predicted_token_id: int
    is_correct: bool

    # Per-layer logits for the predicted/ground-truth token
    layer_logits: np.ndarray  # Shape: (n_layers,)
    layer_margins: np.ndarray  # Shape: (n_layers,) - logit_correct - max_other

    # Per-layer component contributions (if decomposition enabled)
    attn_contributions: Optional[np.ndarray] = None  # Shape: (n_layers,)
    mlp_contributions: Optional[np.ndarray] = None  # Shape: (n_layers,)

    # Metadata
    question_type: Optional[str] = None


class LogitLensAnalyzer:
    """
    Analyzes LLaVA using the logit lens technique to understand
    where answer decisions are made in the model.
    """

    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        device: str = "cuda",
        quantization: str = "4bit",
    ):
        self.model_id = model_id
        self.device = device
        self.quantization = quantization

        # Storage for hooked activations
        self.residual_streams: Dict[int, torch.Tensor] = {}
        self.attn_outputs: Dict[int, torch.Tensor] = {}
        self.mlp_outputs: Dict[int, torch.Tensor] = {}

        # Hooks
        self.hooks: List[Any] = []

        # Load model
        self._load_model()

    def _load_model(self):
        """Load the LLaVA model with quantization."""
        print(f"Loading {self.model_id}...")

        if self.quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id,
                quantization_config=quant_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.float16,
            )

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model.eval()

        # Get the LM head (unembedding matrix) - it's on the main model, not language_model
        self.lm_head = self.model.lm_head

        # Get number of layers
        self.n_layers = self.model.config.text_config.num_hidden_layers
        print(f"Model loaded. {self.n_layers} layers.")

    def _register_hooks(self):
        """Register forward hooks to capture residual streams and component outputs."""
        self._clear_hooks()
        self.residual_streams.clear()
        self.attn_outputs.clear()
        self.mlp_outputs.clear()

        # Get the decoder layers (language_model.layers, not language_model.model.layers)
        layers = self.model.language_model.layers

        for layer_idx, layer in enumerate(layers):
            # Hook after the full layer (residual stream)
            def make_residual_hook(idx):

                def hook(module, input, output):
                    # output is a tuple, first element is hidden states
                    hidden = output[0] if isinstance(output, tuple) else output
                    self.residual_streams[idx] = hidden.detach()

                return hook

            h = layer.register_forward_hook(make_residual_hook(layer_idx))
            self.hooks.append(h)

            # Hook for attention output (before residual add)
            def make_attn_hook(idx):

                def hook(module, input, output):
                    # Self-attention output
                    attn_out = output[0] if isinstance(output,
                                                       tuple) else output
                    self.attn_outputs[idx] = attn_out.detach()

                return hook

            h_attn = layer.self_attn.register_forward_hook(
                make_attn_hook(layer_idx))
            self.hooks.append(h_attn)

            # Hook for MLP output (before residual add)
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
        # hidden_states: (batch, seq, hidden_dim)
        # Returns: (batch, seq, vocab_size)
        # Move to same device and dtype as lm_head
        hidden = hidden_states.to(self.lm_head.weight.device).to(
            self.lm_head.weight.dtype)
        return self.lm_head(hidden)

    def _get_token_logit(self,
                         logits: torch.Tensor,
                         token_id: int,
                         position: int = -1) -> float:
        """Get the logit for a specific token at a specific position."""
        return logits[0, position, token_id].item()

    def _get_margin(self,
                    logits: torch.Tensor,
                    correct_token_id: int,
                    position: int = -1) -> float:
        """Get the margin: logit(correct) - max(logit(other))."""
        correct_logit = logits[0, position, correct_token_id].item()

        # Get max of other tokens
        all_logits = logits[0, position, :].clone()
        all_logits[correct_token_id] = float('-inf')
        max_other = all_logits.max().item()

        return correct_logit - max_other

    def analyze_sample(
        self,
        image: Image.Image,
        question: str,
        ground_truth: str,
        decompose_components: bool = True,
        question_type: Optional[str] = None,
    ) -> LogitLensResult:
        """
        Run logit lens analysis on a single VQA sample.
        
        Args:
            image: PIL Image
            question: The question text
            ground_truth: The ground truth answer
            decompose_components: Whether to decompose attention vs MLP contributions
            question_type: Optional question category
            
        Returns:
            LogitLensResult with per-layer logits and contributions
        """
        # Prepare prompt
        prompt = f"USER: <image>\n{question}\nAnswer with a single word or phrase.\nASSISTANT:"

        # Process inputs
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Register hooks
        self._register_hooks()

        try:
            # Run forward pass to get the first generated token
            with torch.no_grad():
                # First, run to get the prediction
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    output_hidden_states=False,
                    return_dict_in_generate=True,
                )

                # Get the predicted token
                generated_ids = outputs.sequences[0]
                input_len = inputs['input_ids'].shape[1]
                predicted_token_id = generated_ids[input_len].item()
                predicted_token = self.processor.tokenizer.decode(
                    [predicted_token_id]).strip()

                # Check correctness
                is_correct = predicted_token.lower() == ground_truth.lower(
                ).strip()

                # Now run forward pass again to collect residual streams
                # We need to run with the input that would produce this token
                self.residual_streams.clear()
                self.attn_outputs.clear()
                self.mlp_outputs.clear()

                _ = self.model(**inputs, output_hidden_states=True)

                # Collect per-layer logits for the predicted token
                layer_logits = np.zeros(self.n_layers)
                layer_margins = np.zeros(self.n_layers)

                for layer_idx in range(self.n_layers):
                    if layer_idx in self.residual_streams:
                        residual = self.residual_streams[layer_idx]
                        logits = self._apply_lm_head(residual)

                        layer_logits[layer_idx] = self._get_token_logit(
                            logits, predicted_token_id, position=-1)
                        layer_margins[layer_idx] = self._get_margin(
                            logits, predicted_token_id, position=-1)

                # Component decomposition
                attn_contributions = None
                mlp_contributions = None

                if decompose_components:
                    attn_contributions = np.zeros(self.n_layers)
                    mlp_contributions = np.zeros(self.n_layers)

                    # Get the unembedding vector for the predicted token
                    # W_U[:, token_id] gives us the direction for this token
                    unembed_vec = self.lm_head.weight[
                        predicted_token_id, :].detach()
                    unembed_device = unembed_vec.device

                    for layer_idx in range(self.n_layers):
                        if layer_idx in self.attn_outputs:
                            # Attention contribution to the logit
                            attn_out = self.attn_outputs[layer_idx][
                                0, -1, :]  # Last position
                            attn_out = attn_out.to(unembed_device).to(
                                unembed_vec.dtype)
                            attn_contributions[layer_idx] = torch.dot(
                                attn_out, unembed_vec).item()

                        if layer_idx in self.mlp_outputs:
                            # MLP contribution to the logit
                            mlp_out = self.mlp_outputs[layer_idx][
                                0, -1, :]  # Last position
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
        """
        Analyze multiple samples.
        
        Args:
            samples: List of dicts with 'image', 'question', 'ground_truth', 'question_type'
            decompose_components: Whether to decompose attention vs MLP contributions
            
        Returns:
            List of LogitLensResult
        """
        results = []

        for i, sample in enumerate(samples):
            print(f"Processing sample {i+1}/{len(samples)}...")

            try:
                # Load image if it's a URL
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
        """
        Summarize logit lens results across all samples.
        
        Returns dict with:
        - avg_layer_margins: Average margin trajectory
        - avg_attn_contributions: Average attention contribution per layer
        - avg_mlp_contributions: Average MLP contribution per layer
        - correct_vs_incorrect: Comparison of trajectories
        """
        if not results:
            return {}

        # Aggregate
        all_margins = np.stack([r.layer_margins for r in results])

        summary = {
            'n_samples': len(results),
            'n_correct': sum(r.is_correct for r in results),
            'accuracy': sum(r.is_correct for r in results) / len(results),
            'avg_layer_margins': all_margins.mean(axis=0).tolist(),
            'std_layer_margins': all_margins.std(axis=0).tolist(),
        }

        # Component contributions
        if results[0].attn_contributions is not None:
            all_attn = np.stack([r.attn_contributions for r in results])
            all_mlp = np.stack([r.mlp_contributions for r in results])

            summary['avg_attn_contributions'] = all_attn.mean(axis=0).tolist()
            summary['avg_mlp_contributions'] = all_mlp.mean(axis=0).tolist()

            # Total contributions
            total_attn = all_attn.sum(axis=1).mean()
            total_mlp = all_mlp.sum(axis=1).mean()

            summary['total_attn_contribution'] = float(total_attn)
            summary['total_mlp_contribution'] = float(total_mlp)
            summary['mlp_fraction'] = float(total_mlp /
                                            (total_attn + total_mlp + 1e-10))

        # Correct vs incorrect comparison
        correct_results = [r for r in results if r.is_correct]
        incorrect_results = [r for r in results if not r.is_correct]

        if correct_results and incorrect_results:
            correct_margins = np.stack(
                [r.layer_margins for r in correct_results])
            incorrect_margins = np.stack(
                [r.layer_margins for r in incorrect_results])

            summary['correct_avg_margins'] = correct_margins.mean(
                axis=0).tolist()
            summary['incorrect_avg_margins'] = incorrect_margins.mean(
                axis=0).tolist()

            if results[0].mlp_contributions is not None:
                correct_mlp = np.stack(
                    [r.mlp_contributions for r in correct_results])
                incorrect_mlp = np.stack(
                    [r.mlp_contributions for r in incorrect_results])

                summary['correct_avg_mlp'] = correct_mlp.mean(axis=0).tolist()
                summary['incorrect_avg_mlp'] = incorrect_mlp.mean(
                    axis=0).tolist()

        return summary


def run_logit_lens_experiment(
    n_samples: int = 50,
    output_path: str = "logit_lens_results.json",
):
    """
    Run the full logit lens experiment.
    """
    print("=" * 70)
    print("LOGIT LENS EXPERIMENT")
    print("=" * 70)

    # Initialize analyzer
    analyzer = LogitLensAnalyzer(quantization="4bit")

    # Load VQA samples from existing analysis records
    print("\nLoading VQA samples...")

    samples = []

    # Try to load from existing analysis_records.json
    try:
        with open('test_intervention_output/analysis_records.json', 'r') as f:
            data = json.load(f)

        records = data.get('records', [])
        print(f"Found {len(records)} records in analysis_records.json")

        # Convert to samples format
        for record in records[:n_samples]:
            samples.append({
                'image':
                record['image_url'],
                'question':
                record['question'],
                'ground_truth':
                record['ground_truth'],
                'question_type':
                record.get('question_type', 'unknown'),
            })

        print(f"Loaded {len(samples)} VQA samples")
    except FileNotFoundError:
        print("analysis_records.json not found, using example samples...")
        samples = [
            {
                'image': 'https://llava-vl.github.io/static/images/view.jpg',
                'question': 'What is in this image?',
                'ground_truth': 'lake',
                'question_type': 'what',
            },
        ]
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using example samples...")
        samples = [
            {
                'image': 'https://llava-vl.github.io/static/images/view.jpg',
                'question': 'What is in this image?',
                'ground_truth': 'lake',
                'question_type': 'what',
            },
        ]

    # Run analysis
    print(f"\nAnalyzing {len(samples)} samples...")
    results = analyzer.analyze_batch(samples[:n_samples],
                                     decompose_components=True)

    # Summarize
    print("\nSummarizing results...")
    summary = analyzer.summarize_results(results)

    # Print key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    print(f"\nSamples analyzed: {summary['n_samples']}")
    print(f"Accuracy: {summary['accuracy']*100:.1f}%")

    if 'mlp_fraction' in summary:
        print(f"\nComponent contributions to answer logit:")
        print(
            f"  Total Attention contribution: {summary['total_attn_contribution']:.2f}"
        )
        print(
            f"  Total MLP contribution: {summary['total_mlp_contribution']:.2f}"
        )
        print(f"  MLP fraction: {summary['mlp_fraction']*100:.1f}%")

    # Find layer with biggest margin increase
    margins = np.array(summary['avg_layer_margins'])
    margin_changes = np.diff(margins)
    peak_layer = np.argmax(margin_changes) + 1
    print(f"\nLayer with largest margin increase: Layer {peak_layer}")

    if 'avg_mlp_contributions' in summary:
        mlp_contribs = np.array(summary['avg_mlp_contributions'])
        peak_mlp_layer = np.argmax(mlp_contribs)
        print(f"Layer with largest MLP contribution: Layer {peak_mlp_layer}")

    # Save results
    print(f"\nSaving results to {output_path}...")

    # Convert results to serializable format
    results_data = {
        'summary':
        summary,
        'individual_results': [{
            'question':
            r.question,
            'ground_truth':
            r.ground_truth,
            'predicted':
            r.predicted_token,
            'is_correct':
            r.is_correct,
            'question_type':
            r.question_type,
            'layer_margins':
            r.layer_margins.tolist(),
            'attn_contributions':
            r.attn_contributions.tolist()
            if r.attn_contributions is not None else None,
            'mlp_contributions':
            r.mlp_contributions.tolist()
            if r.mlp_contributions is not None else None,
        } for r in results]
    }

    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    print("Done!")

    return results_data


if __name__ == "__main__":
    results = run_logit_lens_experiment(n_samples=50)
