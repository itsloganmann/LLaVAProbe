"""PaliGemma 2 Runner for Layer Evolution Analysis"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from .config import AnalysisConfig
from .metrics import ConfidenceMetrics, compute_confidence_metrics
from .ablations import AblationResult


class RunnerMode(str, enum.Enum):
    FULL = "full"
    LANGUAGE_ONLY = "language_only"
    VISUAL_DROPOUT = "visual_dropout"


@dataclass
class RunnerOutput:
    """Outputs produced for a single prompt/image pair."""

    prompt: str
    prefix: str
    predicted_answer: str
    predicted_token_ids: Sequence[int]
    token_strings: Sequence[str]
    token_probabilities: torch.Tensor
    token_logits: torch.Tensor
    attention_map: np.ndarray
    token_confidence: float
    confidence_metrics: ConfidenceMetrics
    head_delta: float
    ground_truth_delta: Optional[float]
    ground_truth_tokens: Optional[Sequence[str]]
    ablations: List[AblationResult]
    layer_evolution: Optional[dict] = None


class PaliGemmaRunner:
    """High-level helper for PaliGemma 2 execution and layer evolution tracking."""

    def __init__(
        self,
        model_id: str = "google/paligemma2-3b-pt-224",
        device: str = "cuda",
        quantization: Optional[str] = None,
        config: Optional[AnalysisConfig] = None,
    ):
        self.model_id = model_id
        self.device = device
        self._config = config or AnalysisConfig()

        print(f"Loading PaliGemma 2 model: {model_id}")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # Force eager attention to enable attention output
        import os
        os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"
        
        # Load model with optional quantization
        if quantization == "4bit":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                attn_implementation="eager"
            )
        elif quantization == "8bit":
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id,
                load_in_8bit=True,
                device_map="auto",
                attn_implementation="eager"
            )
        else:
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                attn_implementation="eager"
            ).to(device)
        
        self.model.eval()
        print(f"✅ PaliGemma 2 model loaded on {device}")

    @torch.inference_mode()
    def run(
        self,
        image: Image.Image,
        prompt: str,
        prefix: str = "",
        *,
        mode: RunnerMode = RunnerMode.FULL,
        dropout_rate: float = 0.0,
        ground_truth: Optional[str] = None,
    ) -> RunnerOutput:
        """Run inference with layer evolution tracking."""
        
        # Prepare input with image token (PaliGemma expects <image> token)
        full_prompt = f"{prefix} {prompt}".strip() if prefix else prompt
        # Add image token at the beginning if not present
        if "<image>" not in full_prompt:
            full_prompt = f"<image>{full_prompt}"
        
        inputs = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Generate with attention output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
        
        # Get generated tokens
        generated_ids = outputs.sequences[0]
        input_len = inputs.input_ids.shape[1]
        predicted_token_ids = generated_ids[input_len:].tolist()
        
        # Decode answer
        predicted_answer = self.processor.decode(
            predicted_token_ids,
            skip_special_tokens=True
        ).strip()
        
        # Get token strings
        token_strings = [
            self.processor.tokenizer.decode([tid]).strip()
            for tid in predicted_token_ids
        ]
        
        # Compute token probabilities and logits
        if hasattr(outputs, 'scores') and outputs.scores:
            # Stack scores across generation steps
            token_logits = torch.stack([s[0] for s in outputs.scores], dim=0)
            token_probabilities = torch.softmax(token_logits, dim=-1)
            
            # Get probabilities for predicted tokens
            token_confidence = float(torch.mean(torch.tensor([
                token_probabilities[i, tid].item()
                for i, tid in enumerate(predicted_token_ids)
            ])))
        else:
            # Fallback if scores not available
            token_logits = torch.zeros(len(predicted_token_ids), self.model.config.vocab_size)
            token_probabilities = torch.ones(len(predicted_token_ids), self.model.config.vocab_size) / self.model.config.vocab_size
            token_confidence = 0.5
        
        # Get ground truth info if provided
        ground_truth_token_ids: Optional[Sequence[int]] = None
        ground_truth_tokens: Optional[List[str]] = None
        if ground_truth:
            ground_truth_token_ids = self.processor.tokenizer.encode(
                ground_truth,
                add_special_tokens=False
            )
            if ground_truth_token_ids:
                ground_truth_tokens = [
                    self.processor.tokenizer.decode([tid]).strip()
                    for tid in ground_truth_token_ids
                ]
        
        # Compute confidence metrics
        confidence_metrics = compute_confidence_metrics(
            logits=token_logits.cpu(),
            probabilities=token_probabilities.cpu(),
            tokenizer=self.processor.tokenizer,
            predicted_token_ids=predicted_token_ids,
            ground_truth_token_ids=ground_truth_token_ids,
            entropy_config=self._config.entropy,
        )
        
        # Run forward pass with attention tracking
        forward_outputs = self.model(
            **inputs,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        
        # Extract layer attentions
        if hasattr(forward_outputs, 'attentions') and forward_outputs.attentions:
            layer_attentions = [
                attn.detach().cpu().numpy()
                for attn in forward_outputs.attentions
            ]
            
            # Track attention evolution
            from .clustering import track_attention_evolution, print_layer_evolution
            layer_evolution = track_attention_evolution(layer_attentions)
            
            print("\n==== LAYER EVOLUTION ====\n")
            print_layer_evolution(layer_evolution)
        else:
            layer_evolution = None
            print("⚠️  Warning: Attention outputs not available")
        
        # Extract attention map and compute head delta
        attention_map, head_delta, gt_delta = self._extract_attention_map(
            forward_outputs,
            predicted_token_ids,
            ground_truth_token_ids,
        )
        
        # Placeholder for ablations (can be implemented later)
        ablations = []
        
        return RunnerOutput(
            prompt=prompt,
            prefix=prefix,
            predicted_answer=predicted_answer,
            predicted_token_ids=predicted_token_ids,
            token_strings=token_strings,
            token_probabilities=token_probabilities,
            token_logits=token_logits,
            attention_map=attention_map,
            token_confidence=token_confidence,
            confidence_metrics=confidence_metrics,
            head_delta=head_delta,
            ground_truth_delta=gt_delta,
            ground_truth_tokens=ground_truth_tokens,
            ablations=ablations,
            layer_evolution=layer_evolution,
        )

    def _extract_attention_map(
        self,
        outputs,
        predicted_token_ids: Sequence[int],
        ground_truth_token_ids: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, float, Optional[float]]:
        """Extract attention map from model outputs."""
        
        attentions = outputs.attentions if hasattr(outputs, 'attentions') else None
        
        if attentions is None or len(attentions) == 0:
            # Return dummy values if attentions not available
            return np.ones((24, 24)) / (24 * 24), 0.0, None
        
        # Find most focused head across all layers
        batch_size, num_heads, seq_len, _ = attentions[0].shape
        
        best_entropy = float('inf')
        best_layer = 0
        best_head = 0
        
        for layer_idx, attn in enumerate(attentions):
            # Use last token's attention
            last_token_attn = attn[0, :, -1, :]  # [num_heads, seq_len]
            
            for head_idx in range(num_heads):
                head_attn = last_token_attn[head_idx]
                # Compute entropy
                head_attn_normalized = head_attn / (head_attn.sum() + 1e-9)
                entropy = -torch.sum(
                    head_attn_normalized * torch.log(head_attn_normalized + 1e-9)
                )
                
                if entropy < best_entropy:
                    best_entropy = entropy
                    best_layer = layer_idx
                    best_head = head_idx
        
        # Extract attention map from best head
        best_attn = attentions[best_layer][0, best_head, -1, :]
        best_attn_np = best_attn.detach().cpu().numpy()
        
        # Reshape to square (or pad)
        map_size = int(np.sqrt(len(best_attn_np)))
        if map_size * map_size != len(best_attn_np):
            # Pad to nearest square
            map_size = 24  # Default size
            if len(best_attn_np) < map_size * map_size:
                padded = np.zeros(map_size * map_size)
                padded[:len(best_attn_np)] = best_attn_np
                best_attn_np = padded
            else:
                best_attn_np = best_attn_np[:map_size * map_size]
        
        attention_map = best_attn_np.reshape(map_size, map_size)
        
        # Compute head delta (difference from average entropy in that layer)
        layer_entropies = []
        layer_attn = attentions[best_layer][0, :, -1, :]
        for head_idx in range(num_heads):
            head_attn = layer_attn[head_idx]
            head_attn_normalized = head_attn / (head_attn.sum() + 1e-9)
            entropy = -torch.sum(
                head_attn_normalized * torch.log(head_attn_normalized + 1e-9)
            )
            layer_entropies.append(entropy.item())
        
        avg_entropy = np.mean(layer_entropies)
        head_delta = float(avg_entropy - best_entropy.item())
        
        # Ground truth delta (placeholder)
        gt_delta = None
        if ground_truth_token_ids:
            gt_delta = 0.0  # Can be implemented if needed
        
        return attention_map, head_delta, gt_delta

