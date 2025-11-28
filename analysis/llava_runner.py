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
        generated_ids = generation.sequences[0, inputs["input_ids"].shape[1] :]
        token_logits = torch.stack(generation.scores).to(torch.float32)
        token_probabilities = torch.stack([torch.nn.functional.softmax(score, dim=-1) for score in generation.scores])

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

        from .clustering import track_attention_evolution
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
                    pixel_values[:, :, h_start : h_start + self.patch_size, w_start : w_start + self.patch_size] = 0.0
        return pixel_values

    def _extract_attention_map(
        self,
        outputs,
        predicted_token_ids: Sequence[int],
        ground_truth_token_ids: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, float, Optional[float]]:
        past_key_values = outputs.past_key_values
        token_id = predicted_token_ids[-1] if predicted_token_ids else None

        all_pos_layer_input, all_pos_layer_output, all_last_attn_subvalues = self._transfer_output(past_key_values)
        final_var = torch.tensor(all_pos_layer_output[-1][-1]).pow(2).mean(-1, keepdim=True)

        # compute increases per head
        head_increases: List[Tuple[int, int, float]] = []
        for layer_idx in range(self.num_layers):
            cur_layer_input = torch.tensor(all_pos_layer_input[layer_idx])
            cur_v_heads = torch.tensor(all_last_attn_subvalues[layer_idx])
            o_proj = self.model.language_model.model.layers[layer_idx].self_attn.o_proj.weight.data.T.view(
                self.num_heads, self.head_dim, -1
            )
            attn_recompute = torch.bmm(cur_v_heads, o_proj).permute(1, 0, 2)
            attn_sum = torch.sum(attn_recompute, dim=0)
            layer_input_last = cur_layer_input[-1]
            origin_prob = self._log_probability(layer_input_last, final_var, token_id)

            attn_plus = attn_sum + layer_input_last
            attn_plus_probs = self._log_probability(attn_plus, final_var, token_id)
            head_deltas = attn_plus_probs - origin_prob
            for head_idx, delta in enumerate(head_deltas):
                head_increases.append((layer_idx, head_idx, float(delta.item())))

        head_increases.sort(key=lambda item: item[2], reverse=True)
        best_layer, best_head, _ = head_increases[0]

        cur_layer_input = outputs.past_key_values[best_layer][0][0]
        cur_v_heads = outputs.past_key_values[best_layer][5][0]
        o_proj = self.model.language_model.model.layers[best_layer].self_attn.o_proj.weight.data.T.view(
            self.num_heads, self.head_dim, -1
        )
        attn_recompute = torch.bmm(cur_v_heads, o_proj).permute(1, 0, 2)
        attn_cur_head = attn_recompute[:, best_head, :]

        layer_input_last = cur_layer_input[-1]
        final_var = torch.tensor(all_pos_layer_output[-1][-1]).pow(2).mean(-1, keepdim=True)
        origin_prob = self._log_probability(layer_input_last, final_var, token_id)

        attn_plus = attn_cur_head + layer_input_last
        attn_plus_probs = self._log_probability(attn_plus, final_var, token_id)
        head_pos_increase = attn_plus_probs - origin_prob
        curhead_scores = head_pos_increase.tolist()[5:581]
        normalized_scores = self._normalize(curhead_scores)
        attention_map = np.array(normalized_scores).reshape(24, 24)

        # Head ablation delta: zero out best head and measure delta
        zeroed_scores = self._log_probability(layer_input_last - attn_cur_head, final_var, token_id)
        head_delta = float(origin_prob.exp() - zeroed_scores.exp())

        ground_truth_delta: Optional[float] = None
        if ground_truth_token_ids:
            gt_token = ground_truth_token_ids[-1]
            gt_origin = self._log_probability(layer_input_last, final_var, gt_token)
            gt_zeroed = self._log_probability(layer_input_last - attn_cur_head, final_var, gt_token)
            ground_truth_delta = float(gt_origin.exp() - gt_zeroed.exp())

        return attention_map, head_delta, ground_truth_delta

    def _run_head_ablation(
        self,
        outputs,
        predicted_token_ids: Sequence[int],
        *,
        baseline_prob: float,
    ) -> List[AblationResult]:
        token_id = predicted_token_ids[-1] if predicted_token_ids else None
        results: List[AblationResult] = []

        rng = random.Random(self._config.random_seed)
        for trial in range(self._config.ablations.head_dropout_trials):
            layer = rng.randrange(self.num_layers)
            head = rng.randrange(self.num_heads)
            delta = self._ablate_head(outputs, layer, head, token_id)
            results.append(
                AblationResult(
                    experiment=AblationExperiment(
                        name=f"random_head_{trial}",
                        description="Random head ablation",
                        parameters={"layer": layer, "head": head},
                    ),
                    baseline_confidence=baseline_prob,
                    ablated_confidence=max(delta, 0.0),
                    confidence_delta=baseline_prob - max(delta, 0.0),
                    additional_metrics={},
                )
            )
        return results

    def _ablate_head(self, outputs, layer: int, head: int, token_id: Optional[int]) -> float:
        cur_layer_input = outputs.past_key_values[layer][0][0]
        cur_v_heads = outputs.past_key_values[layer][5][0]
        o_proj = self.model.language_model.model.layers[layer].self_attn.o_proj.weight.data.T.view(
            self.num_heads, self.head_dim, -1
        )
        attn_recompute = torch.bmm(cur_v_heads, o_proj).permute(1, 0, 2)
        attn_cur_head = attn_recompute[:, head, :]
        layer_input_last = cur_layer_input[-1]
        final_layer_output = outputs.past_key_values[self.num_layers - 1][4][0][-1]
        final_var = final_layer_output.pow(2).mean(-1, keepdim=True)

        baseline = self._log_probability(layer_input_last, final_var, token_id).exp()
        ablated = self._log_probability(layer_input_last - attn_cur_head, final_var, token_id).exp()
        return float(ablated)

    def _transfer_output(self, model_output) -> Tuple[List, List, List]:
        all_pos_layer_input = []
        all_pos_layer_output = []
        all_last_attn_subvalues = []
        for layer_idx in range(self.num_layers):
            layer_tuple = model_output[layer_idx]
            all_pos_layer_input.append(layer_tuple[0][0].tolist())
            all_pos_layer_output.append(layer_tuple[4][0].tolist())
            all_last_attn_subvalues.append(layer_tuple[5][0].tolist())
        return all_pos_layer_input, all_pos_layer_output, all_last_attn_subvalues

    def _log_probability(self, vector: torch.Tensor, final_var: torch.Tensor, token_id: Optional[int]) -> torch.Tensor:
        if token_id is None:
            raise ValueError("token_id must be provided")
        scaled = vector * torch.rsqrt(final_var + 1e-6)
        rms = scaled * self.model.language_model.model.norm.weight.data
        logits = self.model.language_model.lm_head(rms).data
        probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return probs[token_id]

    @staticmethod
    def _normalize(vector: Sequence[float]) -> List[float]:
        arr = np.asarray(vector, dtype=np.float64)
        arr -= arr.min()
        denom = arr.sum()
        if denom == 0:
            return [0.0 for _ in vector]
        return list(arr / denom)
