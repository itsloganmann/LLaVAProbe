"""Wrapper around LLaVA for reproducible attention and confidence analysis."""

from __future__ import annotations

import enum
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.generation.utils import GenerateDecoderOnlyOutput
try:  # pragma: no cover - optional dependency
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover - optional dependency
    BitsAndBytesConfig = None

from .ablations import AblationResult, AblationExperiment, visual_dropout_mask
from .config import AnalysisConfig, EntropyConfig
from .metrics import ConfidenceMetrics, compute_confidence_metrics


class RunnerMode(str, enum.Enum):
    """Operating modes for the runner."""

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


class LlavaRunner:
    """High-level helper encapsulating LLaVA execution and intermediate extraction."""

    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        device: Optional[str] = None,
        config: Optional[AnalysisConfig] = None,
        quantization: Optional[str] = None,
    ) -> None:
        self._config = config or AnalysisConfig()
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)
        quant_cfg = None
        if quantization in {"4bit", "8bit"} and BitsAndBytesConfig is not None:
            if quantization == "4bit":
                quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)
            else:
                quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            revision="a272c74",
            quantization_config=quant_cfg,
            torch_dtype=torch.float16 if quant_cfg is not None else None,
        )
        self.model = cast(LlavaForConditionalGeneration, model)
        self.model.to(self.device)  # type: ignore[arg-type]
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id, revision="a272c74")
        self.patch_size = getattr(self.processor, "patch_size", 14)

        llama_config = self.model.language_model.model.config
        self.num_layers = llama_config.num_hidden_layers
        self.num_heads = llama_config.num_attention_heads
        self.head_dim = llama_config.hidden_size // self.num_heads

    def set_vision_cutoff(self, mode: str, cutoff_layer: int) -> None:
        """Configure vision cutoff ablation for the model.
        
        Args:
            mode: "early_cut" (disable layers > cutoff_layer) or "late_only" (disable layers < cutoff_layer)
            cutoff_layer: Layer index for cutoff boundary
        """
        state = self.model.language_model.vision_cutoff
        state["enabled"] = True
        state["mode"] = mode
        state["cutoff_layer"] = cutoff_layer
        
        if mode == "early_cut":
            # disable layers > cutoff_layer
            state["disabled_layers"] = set(range(cutoff_layer + 1, self.num_layers))
        elif mode == "late_only":
            # disable layers < cutoff_layer
            state["disabled_layers"] = set(range(0, cutoff_layer))
        
        # Update module-level registry for layer access
        try:
            import sys
            if 'modeling_llama' in sys.modules:
                sys.modules['modeling_llama']._VISION_CUTOFF_STATE = state
        except:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def run(
        self,
        image: Image.Image,
        prompt: str,
        prefix: str,
        *,
        mode: RunnerMode = RunnerMode.FULL,
        dropout_rate: float = 0.0,
        ground_truth: Optional[str] = None,
    ) -> RunnerOutput:
        full_prompt = f"USER: <image>\n{prompt}\nASSISTANT: {prefix}"
        inputs = self.processor(text=full_prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if mode == RunnerMode.LANGUAGE_ONLY:
            inputs["pixel_values"] = torch.zeros_like(inputs["pixel_values"])
        elif mode == RunnerMode.VISUAL_DROPOUT and dropout_rate > 0.0:
            inputs["pixel_values"] = self._apply_visual_dropout(inputs["pixel_values"], dropout_rate)

        generation = cast(
            GenerateDecoderOnlyOutput,
            self.model.generate(
                **inputs,
                max_new_tokens=32,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
            ),
        )
        if generation.scores is None:
            raise RuntimeError("Expected generation scores to be available; ensure output_scores=True")
        generated_ids = generation.sequences[0, inputs["input_ids"].shape[1]:]
        token_logits = torch.stack(generation.scores).to(torch.float32)
        token_probabilities = torch.stack(
            [torch.nn.functional.softmax(score, dim=-1) for score in generation.scores]
        )

        predicted_answer = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        predicted_token_ids = generated_ids.tolist()
        token_strings = [self.processor.tokenizer.decode([tid]).strip() for tid in predicted_token_ids]

        # Compute detailed confidence metrics using per-token logits/probabilities
        ground_truth_token_ids: Optional[Sequence[int]] = None
        ground_truth_tokens: Optional[List[str]] = None
        if ground_truth:
            ground_truth_token_ids = self.processor.tokenizer.encode(ground_truth, add_special_tokens=False)
            if ground_truth_token_ids:
                ground_truth_tokens = [
                    self.processor.tokenizer.decode([tid]).strip() for tid in ground_truth_token_ids
                ]

        if token_logits.ndim == 3:
            token_logits = token_logits.squeeze(1)

        if token_probabilities.ndim == 3:
            token_probabilities = token_probabilities.squeeze(1)

        confidence_metrics = compute_confidence_metrics(
            logits=token_logits,
            probabilities=token_probabilities,
            tokenizer=self.processor.tokenizer,
            predicted_token_ids=predicted_token_ids,
            ground_truth_token_ids=ground_truth_token_ids,
            entropy_config=self._config.entropy,
        )
        token_confidence = confidence_metrics.top1_probability

        # Extract attention map and head deltas from a focused forward pass on final token
        forward_outputs = self.model(
            **inputs,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=True,
            return_dict=True,
        )

        layer_attentions = [
            attn.detach().cpu().numpy()
            for attn in forward_outputs.attentions
        ]

        from .clustering import track_attention_evolution, print_layer_evolution
        layer_evolution = track_attention_evolution(layer_attentions)

        print("\n==== LAYER EVOLUTION ====\n")
        print_layer_evolution(layer_evolution)

        attention_map, head_delta, gt_delta = self._extract_attention_map(
            forward_outputs,
            predicted_token_ids,
            ground_truth_token_ids,
        )

        ablations = self._run_head_ablation(
            forward_outputs,
            predicted_token_ids,
            baseline_prob=token_confidence,
        )

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
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_visual_dropout(self, pixel_values: torch.Tensor, dropout_rate: float) -> torch.Tensor:
        batch, channels, height, width = pixel_values.shape
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        mask = visual_dropout_mask(num_patches_h * num_patches_w, dropout_rate)
        mask = mask.reshape(num_patches_h, num_patches_w)

        pixel_values = pixel_values.clone()
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                if mask[i, j] == 0:
                    h_start = i * self.patch_size
                    w_start = j * self.patch_size
                    pixel_values[:, :, h_start:h_start + self.patch_size, w_start:w_start + self.patch_size] = 0.0
        return pixel_values

    def _extract_attention_map(
        self,
        outputs,
        predicted_token_ids: Sequence[int],
        ground_truth_token_ids: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, float, Optional[float]]:
        """
        Option B: work purely in attention space.

        - Use per-layer attention for the *last generated token*.
        - Score heads by how focused they are (low entropy).
        - Pick the single best (layer, head) across all layers.
        - Build a 24x24 heatmap from that head's attention over (typically) vision tokens.
        - Define head_delta as how much more focused this head is vs the average head in that layer.
        """
        attentions = outputs.attentions  # tuple of length num_layers
        if attentions is None or len(attentions) == 0:
            raise RuntimeError("Model outputs do not contain attentions. Ensure output_attentions=True.")

        # We assume batch_size == 1 for this analysis pipeline
        batch_size, num_heads, seq_len, _ = attentions[0].shape
        assert batch_size == 1, "LlavaRunner._extract_attention_map currently assumes batch_size = 1"

        # ------------------------------------------------------------------
        # 1) Find the most "focused" head across all layers (lowest entropy)
        # ------------------------------------------------------------------
        best_global_score: Optional[torch.Tensor] = None
        best_layer: int = -1
        best_head: int = -1
        best_head_entropy: float = 0.0

        for layer_idx, attn in enumerate(attentions):
            # attn: [batch, heads, seq, seq]
            attn_layer = attn[0]  # [heads, seq, seq]
            # Take attention for the *last token* as query: [:, -1, :]
            final_attn = attn_layer[:, -1, :]  # [heads, seq]

            # Entropy per head: H(p) = -sum p log p
            # Clamp to avoid log(0)
            p = torch.clamp(final_attn, min=1e-8)
            ent = -(p * p.log()).sum(dim=-1)  # [heads]
            # Define focus score as negative entropy (higher => more focused)
            score = -ent

            max_score, head_idx = torch.max(score, dim=0)
            if best_global_score is None or max_score > best_global_score:
                best_global_score = max_score
                best_layer = layer_idx
                best_head = int(head_idx.item())
                best_head_entropy = float(ent[head_idx].item())

        # ------------------------------------------------------------------
        # 2) Compute a simple "head_delta" = average_entropy(layer) - entropy(best_head)
        #    Positive => best head is more focused than typical head in that layer.
        # ------------------------------------------------------------------
        chosen_attn_layer = attentions[best_layer][0]  # [heads, seq, seq]
        chosen_final_attn = chosen_attn_layer[:, -1, :]  # [heads, seq]
        p_chosen = torch.clamp(chosen_final_attn, min=1e-8)
        ent_chosen = -(p_chosen * p_chosen.log()).sum(dim=-1)  # [heads]
        avg_entropy_layer = float(ent_chosen.mean().item())
        head_delta = avg_entropy_layer - best_head_entropy

        # ------------------------------------------------------------------
        # 3) Construct a 24x24 attention heatmap from best (layer, head)
        # ------------------------------------------------------------------
        best_head_attn = chosen_final_attn[best_head]  # [seq]
        seq_len = best_head_attn.shape[0]

        # Original code used [5:581] which is 576 = 24*24 positions.
        # We'll mimic that logic but handle variable seq_len robustly.
        start = 5
        num_spatial = 24 * 24
        end = min(start + num_spatial, seq_len)

        attn_slice = best_head_attn[start:end]  # [<=576]
        if attn_slice.shape[0] < num_spatial:
            pad = torch.zeros(num_spatial - attn_slice.shape[0], device=best_head_attn.device)
            attn_slice = torch.cat([attn_slice, pad], dim=0)

        # Normalize slice to sum to 1 for a nicer heatmap
        total = attn_slice.sum()
        if total > 0:
            attn_slice = attn_slice / total

        attention_map = attn_slice.detach().cpu().numpy().reshape(24, 24).astype(np.float32)

        # ------------------------------------------------------------------
        # 4) We are not computing a log-prob-based ground_truth_delta here.
        # ------------------------------------------------------------------
        ground_truth_delta: Optional[float] = None

        return attention_map, float(head_delta), ground_truth_delta

    def _run_head_ablation(
        self,
        outputs,
        predicted_token_ids: Sequence[int],
        *,
        baseline_prob: float,
    ) -> List[AblationResult]:
        """
        Option B (simplified): we no longer perform true logit-based ablation using K/V.
        Instead, we can either:
          - return an empty list, or
          - return placeholder entries describing attention-only stats.

        For now, we return an empty list to keep the API stable and avoid
        mixing attention-space with logit-space in a hacky way.
        """
        return []

    # (No _ablate_head and no _log_probability needed in Option B.)

    @staticmethod
    def _normalize(vector: Sequence[float]) -> List[float]:
        arr = np.asarray(vector, dtype=np.float64)
        arr -= arr.min()
        denom = arr.sum()
        if denom == 0:
            return [0.0 for _ in vector]
        return list(arr / denom)
