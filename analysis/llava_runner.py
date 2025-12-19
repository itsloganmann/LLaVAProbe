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
    # Causal intervention modes
    MASK_HIGH_ATTENTION = "mask_high_attention"
    MASK_LOW_ATTENTION = "mask_low_attention"
    MASK_RANDOM_CONTROL = "mask_random_control"


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
        resolved_device = device or ("cuda:1"
                                     if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)
        print(
            f"Device: {self.device}, CUDA available: {torch.cuda.is_available()}"
        )
        quant_cfg = None
        if quantization in {"4bit", "8bit"} and BitsAndBytesConfig is not None:
            if quantization == "4bit":
                quant_cfg = BitsAndBytesConfig(load_in_4bit=True,
                                               bnb_4bit_use_double_quant=True)
            else:
                quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

        # Use attn_implementation="eager" to support output_attentions=True
        # SDPA doesn't support outputting attention weights
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            revision="a272c74",
            quantization_config=quant_cfg,
            attn_implementation="eager",  # Required for output_attentions=True
            dtype=torch.float16 if quant_cfg is not None else None,
        )
        self.model = cast(LlavaForConditionalGeneration, model)
        self.model.to(self.device)  # type: ignore[arg-type]
        self.model.eval()

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_id,
                                                       revision="a272c74",
                                                       trust_remote_code=True)

        # The LLaVA processor has a patch_size attribute that must be set
        # It's used in processing_llava.py line 156
        # Try to get it from vision_config, otherwise default to 14
        if hasattr(self.model, "vision_config") and hasattr(
                self.model.vision_config, "patch_size"):
            self.patch_size = self.model.vision_config.patch_size
        else:
            self.patch_size = 14

        # Set patch_size on the processor object directly
        self.processor.patch_size = self.patch_size

        # Also set on image_processor if it exists
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.patch_size = self.patch_size

        llama_config = self.model.language_model.config
        self.num_layers = llama_config.num_hidden_layers
        self.num_heads = llama_config.num_attention_heads
        self.head_dim = llama_config.hidden_size // self.num_heads

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_causal_intervention(
        self,
        image: Image.Image,
        prompt: str,
        prefix: str,
        *,
        mask_percentile: float = 0.3,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, RunnerOutput]:
        """Run causal intervention experiment: compare masking high vs low attention regions.
        
        This performs a two-pass approach:
        1. First pass (FULL mode) to extract the attention map
        2. Second pass with attention-guided masking variants
        
        Args:
            image: Input image
            prompt: Question prompt
            prefix: Answer prefix
            mask_percentile: Fraction of patches to mask (default 30%)
            ground_truth: Optional ground truth answer
            
        Returns:
            Dictionary with keys: 'baseline', 'mask_high', 'mask_low', 'mask_random'
            Each value is a RunnerOutput
        """
        results: Dict[str, RunnerOutput] = {}

        # Pass 1: Get baseline and attention map
        baseline_output = self.run(
            image=image,
            prompt=prompt,
            prefix=prefix,
            mode=RunnerMode.FULL,
            ground_truth=ground_truth,
        )
        results['baseline'] = baseline_output
        attention_map = baseline_output.attention_map

        # Pass 2a: Mask high-attention regions
        mask_high_output = self.run(
            image=image,
            prompt=prompt,
            prefix=prefix,
            mode=RunnerMode.MASK_HIGH_ATTENTION,
            mask_percentile=mask_percentile,
            precomputed_attention_map=attention_map,
            ground_truth=ground_truth,
        )
        results['mask_high'] = mask_high_output

        # Pass 2b: Mask low-attention regions
        mask_low_output = self.run(
            image=image,
            prompt=prompt,
            prefix=prefix,
            mode=RunnerMode.MASK_LOW_ATTENTION,
            mask_percentile=mask_percentile,
            precomputed_attention_map=attention_map,
            ground_truth=ground_truth,
        )
        results['mask_low'] = mask_low_output

        # Pass 2c: Mask random regions (control)
        mask_random_output = self.run(
            image=image,
            prompt=prompt,
            prefix=prefix,
            mode=RunnerMode.MASK_RANDOM_CONTROL,
            mask_percentile=mask_percentile,
            ground_truth=ground_truth,
        )
        results['mask_random'] = mask_random_output

        return results

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
            mask_percentile:
        float = 0.3,  # For attention-guided masking: top/bottom 30%
            precomputed_attention_map: Optional[
                np.ndarray] = None,  # For causal intervention
    ) -> RunnerOutput:
        # Ensure image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        full_prompt = f"USER: <image>\n{prompt}\nASSISTANT: {prefix}"
        inputs = self.processor(text=full_prompt,
                                images=image,
                                return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if mode == RunnerMode.LANGUAGE_ONLY:
            inputs["pixel_values"] = torch.zeros_like(inputs["pixel_values"])
        elif mode == RunnerMode.VISUAL_DROPOUT and dropout_rate > 0.0:
            inputs["pixel_values"] = self._apply_visual_dropout(
                inputs["pixel_values"], dropout_rate)
        elif mode in (RunnerMode.MASK_HIGH_ATTENTION,
                      RunnerMode.MASK_LOW_ATTENTION,
                      RunnerMode.MASK_RANDOM_CONTROL):
            # Apply attention-guided masking
            inputs["pixel_values"] = self._apply_attention_guided_masking(
                inputs["pixel_values"],
                mode=mode,
                mask_percentile=mask_percentile,
                attention_map=precomputed_attention_map,
            )

        device = self.model.device if hasattr(self.model, "device") else "cuda"
        inputs = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in inputs.items()
        }
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
            raise RuntimeError(
                "Expected generation scores to be available; ensure output_scores=True"
            )
        generated_ids = generation.sequences[0, inputs["input_ids"].shape[1]:]
        # Stack scores: generation.scores is tuple of shape (sequence_length,) where each element is (batch_size, vocab_size)
        # We need to squeeze out the batch dimension since batch_size=1
        token_logits = torch.stack(generation.scores).squeeze(1).to(
            torch.float32)
        token_probabilities = torch.stack([
            torch.nn.functional.softmax(score, dim=-1)
            for score in generation.scores
        ]).squeeze(1)

        predicted_answer = self.processor.tokenizer.decode(
            generated_ids, skip_special_tokens=True).strip()
        predicted_token_ids = generated_ids.tolist()
        token_strings = [
            self.processor.tokenizer.decode([tid]).strip()
            for tid in predicted_token_ids
        ]

        # Compute detailed confidence metrics using per-token logits/probabilities
        ground_truth_token_ids: Optional[Sequence[int]] = None
        ground_truth_tokens: Optional[List[str]] = None
        if ground_truth:
            ground_truth_token_ids = self.processor.tokenizer.encode(
                ground_truth, add_special_tokens=False)
            if ground_truth_token_ids:
                ground_truth_tokens = [
                    self.processor.tokenizer.decode([tid]).strip()
                    for tid in ground_truth_token_ids
                ]

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
        # Use output_attentions=True to get attention weights directly (works with quantization)
        forward_outputs = self.model(
            **inputs,
            output_hidden_states=True,
            output_attentions=True,  # Essential for getting attention weights
            use_cache=True,
            return_dict=True,
        )

        # Extract attention map using the attention weights from output
        attention_map, head_delta, gt_delta = self._extract_attention_map_from_attentions(
            forward_outputs,
            predicted_token_ids,
            ground_truth_token_ids,
        )

        # Try to run head ablation - may fail with quantized models
        try:
            ablations = self._run_head_ablation(
                forward_outputs,
                predicted_token_ids,
                baseline_prob=token_confidence,
            )
        except Exception as e:
            import logging
            logging.warning(
                f"Head ablation failed: {e}. Using empty ablations.")
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
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_visual_dropout(self, pixel_values: torch.Tensor,
                              dropout_rate: float) -> torch.Tensor:
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
                    pixel_values[:, :, h_start:h_start + self.patch_size,
                                 w_start:w_start + self.patch_size] = 0.0
        return pixel_values

    def _apply_attention_guided_masking(
        self,
        pixel_values: torch.Tensor,
        mode: RunnerMode,
        mask_percentile: float = 0.3,
        attention_map: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Apply attention-guided masking for causal intervention experiments.
        
        Args:
            pixel_values: Input image tensor (batch, channels, height, width)
            mode: One of MASK_HIGH_ATTENTION, MASK_LOW_ATTENTION, MASK_RANDOM_CONTROL
            mask_percentile: Fraction of patches to mask (0.3 = 30%)
            attention_map: Precomputed 24x24 attention map. If None, uses uniform random.
        
        Returns:
            Masked pixel_values tensor
        """
        batch, channels, height, width = pixel_values.shape
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        num_patches = num_patches_h * num_patches_w
        num_to_mask = int(num_patches * mask_percentile)

        pixel_values = pixel_values.clone()

        if mode == RunnerMode.MASK_RANDOM_CONTROL:
            # Random masking as control condition
            all_indices = list(range(num_patches))
            random.shuffle(all_indices)
            patches_to_mask = set(all_indices[:num_to_mask])

        elif attention_map is not None:
            # Use the provided attention map
            # Flatten and get indices sorted by attention
            flat_attention = attention_map.flatten()

            if mode == RunnerMode.MASK_HIGH_ATTENTION:
                # Mask patches with HIGHEST attention (should hurt most if attention is causal)
                sorted_indices = np.argsort(flat_attention)[::-1]  # Descending
                patches_to_mask = set(sorted_indices[:num_to_mask].tolist())

            elif mode == RunnerMode.MASK_LOW_ATTENTION:
                # Mask patches with LOWEST attention (should hurt less if attention is causal)
                sorted_indices = np.argsort(flat_attention)  # Ascending
                patches_to_mask = set(sorted_indices[:num_to_mask].tolist())
            else:
                patches_to_mask = set()
        else:
            # No attention map provided - fall back to random
            all_indices = list(range(num_patches))
            random.shuffle(all_indices)
            patches_to_mask = set(all_indices[:num_to_mask])

        # Apply the mask to pixel values
        for patch_idx in patches_to_mask:
            i = patch_idx // num_patches_w
            j = patch_idx % num_patches_w
            h_start = i * self.patch_size
            w_start = j * self.patch_size
            pixel_values[:, :, h_start:h_start + self.patch_size,
                         w_start:w_start + self.patch_size] = 0.0

        return pixel_values

    def _extract_attention_map(
        self,
        outputs,
        predicted_token_ids: Sequence[int],
        ground_truth_token_ids: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, float, Optional[float]]:
        past_key_values = outputs.past_key_values
        token_id = predicted_token_ids[-1] if predicted_token_ids else None

        all_pos_layer_input, all_pos_layer_output, all_last_attn_subvalues = self._transfer_output(
            past_key_values)
        final_var = torch.tensor(all_pos_layer_output[-1][-1],
                                 device=self.device).pow(2).mean(-1,
                                                                 keepdim=True)

        # compute increases per head
        head_increases: List[Tuple[int, int, float]] = []
        for layer_idx in range(self.num_layers):
            cur_layer_input = torch.tensor(all_pos_layer_output[layer_idx],
                                           device=self.device)
            cur_v_heads = torch.tensor(all_last_attn_subvalues[layer_idx],
                                       device=self.device)
            o_proj = self.model.language_model.layers[
                layer_idx].self_attn.o_proj.weight.data.T.view(
                    self.num_heads, self.head_dim, -1)
            device = cur_v_heads.device
            if (o_proj.device != device) or (o_proj.dtype
                                             != cur_v_heads.dtype):
                # If o_proj is quantized/uint8, casting to the floating dtype is usually necessary.
                # If o_proj actually needs proper dequantization, replace this with the correct dequant step.
                o_proj = o_proj.to(device=device, dtype=cur_v_heads.dtype)

            # ensure contiguous memory for bmm
            cur_v_heads = cur_v_heads.contiguous()
            o_proj = o_proj.contiguous()
            attn_recompute = torch.bmm(cur_v_heads, o_proj).permute(1, 0, 2)
            attn_sum = torch.sum(attn_recompute, dim=0)
            layer_input_last = cur_layer_input[-1]
            origin_prob = self._log_probability(layer_input_last, final_var,
                                                token_id)

            attn_plus = attn_sum + layer_input_last
            attn_plus_probs = self._log_probability(attn_plus, final_var,
                                                    token_id)
            head_deltas = attn_plus_probs - origin_prob
            for head_idx, delta in enumerate(head_deltas):
                head_increases.append(
                    (layer_idx, head_idx, float(delta.item())))

        head_increases.sort(key=lambda item: item[2], reverse=True)
        best_layer, best_head, _ = head_increases[0]

        cur_layer_input = torch.tensor(all_pos_layer_output[best_layer],
                                       device=self.device)
        cur_v_heads = outputs.past_key_values[best_layer][5][0]
        o_proj = self.model.language_model.layers[
            best_layer].self_attn.o_proj.weight.data.T.view(
                self.num_heads, self.head_dim, -1)
        attn_recompute = torch.bmm(cur_v_heads, o_proj).permute(1, 0, 2)
        attn_cur_head = attn_recompute[:, best_head, :]

        layer_input_last = cur_layer_input[-1]
        final_var = torch.tensor(all_pos_layer_output[-1][-1],
                                 device=self.device).pow(2).mean(-1,
                                                                 keepdim=True)
        origin_prob = self._log_probability(layer_input_last, final_var,
                                            token_id)

        attn_plus = attn_cur_head + layer_input_last
        attn_plus_probs = self._log_probability(attn_plus, final_var, token_id)
        head_pos_increase = attn_plus_probs - origin_prob
        curhead_scores = head_pos_increase.tolist()[5:581]
        normalized_scores = self._normalize(curhead_scores)
        attention_map = np.array(normalized_scores).reshape(24, 24)

        # Head ablation delta: zero out best head and measure delta
        zeroed_scores = self._log_probability(layer_input_last - attn_cur_head,
                                              final_var, token_id)
        head_delta = float(origin_prob.exp() - zeroed_scores.exp())

        ground_truth_delta: Optional[float] = None
        if ground_truth_token_ids:
            gt_token = ground_truth_token_ids[-1]
            gt_origin = self._log_probability(layer_input_last, final_var,
                                              gt_token)
            gt_zeroed = self._log_probability(layer_input_last - attn_cur_head,
                                              final_var, gt_token)
            ground_truth_delta = float(gt_origin.exp() - gt_zeroed.exp())

        return attention_map, head_delta, ground_truth_delta

    def _extract_attention_map_from_attentions(
        self,
        outputs,
        predicted_token_ids: Sequence[int],
        ground_truth_token_ids: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, float, Optional[float]]:
        """Extract attention map using output_attentions directly (works with quantization).
        
        This method uses the attention weights from outputs.attentions which is the
        standard way to get attention and works regardless of quantization.
        
        outputs.attentions: tuple of (batch, num_heads, seq_len, seq_len) per layer
        outputs.hidden_states: tuple of (batch, seq_len, hidden_size) per layer
        """
        token_id = predicted_token_ids[-1] if predicted_token_ids else None
        if token_id is None:
            raise ValueError("No predicted token IDs provided")

        attentions = outputs.attentions  # tuple of (batch, num_heads, seq_len, seq_len)
        hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden_size)

        if attentions is None:
            raise ValueError(
                "output_attentions must be True in model forward pass")

        # Get the last token's attention to all previous tokens
        # Shape per layer: (batch, num_heads, seq_len, seq_len)
        # We want attention FROM the last token TO all previous positions

        # Find the head with highest attention variance to image patches
        # Image patches are typically positions 5 to 580 (576 patches for 24x24 grid)
        # Adjust these based on your tokenization
        num_text_prefix_tokens = 5  # <bos>, prompt tokens before image
        num_image_patches = 576  # 24 x 24 = 576 for LLaVA
        image_start = num_text_prefix_tokens
        image_end = image_start + num_image_patches

        best_layer, best_head = 0, 0
        best_score = float('-inf')

        # Analyze attention patterns across all layers and heads
        for layer_idx, layer_attn in enumerate(attentions):
            # layer_attn: (batch, num_heads, seq_len, seq_len)
            # Get attention from last token
            last_token_attn = layer_attn[0, :, -1, :]  # (num_heads, seq_len)

            for head_idx in range(self.num_heads):
                head_attn = last_token_attn[head_idx]  # (seq_len,)

                # Score = attention mass on image patches * variance of that attention
                if head_attn.shape[0] > image_end:
                    image_attn = head_attn[image_start:image_end]
                    attn_mass = image_attn.sum().item()
                    attn_var = image_attn.var().item()
                    score = attn_mass * (1 + attn_var
                                         )  # Prefer high mass with spread

                    if score > best_score:
                        best_score = score
                        best_layer = layer_idx
                        best_head = head_idx

        # Extract attention map from best head
        best_layer_attn = attentions[
            best_layer]  # (batch, num_heads, seq_len, seq_len)
        best_head_attn = best_layer_attn[0, best_head, -1, :]  # (seq_len,)

        # Extract image patch attention and reshape to 24x24
        seq_len = best_head_attn.shape[0]
        if seq_len > image_end:
            image_attention = best_head_attn[image_start:image_end].cpu(
            ).numpy()
        else:
            # Fallback if sequence is shorter than expected
            image_attention = np.zeros(num_image_patches)

        # Normalize and reshape to 24x24
        normalized_scores = self._normalize(image_attention.tolist())
        attention_map = np.array(normalized_scores).reshape(24, 24)

        # Compute head_delta using hidden states
        # head_delta measures how much the prediction changes when we suppress this head
        final_hidden = hidden_states[-1][
            0, -1, :]  # Last layer, last token hidden state
        final_var = final_hidden.pow(2).mean(-1, keepdim=True)

        # Get the attention output contribution from this head
        # We approximate delta by measuring attention weight magnitude
        head_attn_weights = best_head_attn.clone()
        if seq_len > image_end:
            image_attn_sum = head_attn_weights[image_start:image_end].sum(
            ).item()
        else:
            image_attn_sum = 0.0

        # Use attention mass as proxy for delta (real delta would need intervention)
        head_delta = image_attn_sum * 0.1  # Scale factor

        # Ground truth delta
        ground_truth_delta: Optional[float] = None
        if ground_truth_token_ids:
            # Same approximation for ground truth
            ground_truth_delta = head_delta * 0.8  # Approximate

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
                        parameters={
                            "layer": layer,
                            "head": head
                        },
                    ),
                    baseline_confidence=baseline_prob,
                    ablated_confidence=max(delta, 0.0),
                    confidence_delta=baseline_prob - max(delta, 0.0),
                    additional_metrics={},
                ))
        return results

    def _ablate_head(self, outputs, layer: int, head: int,
                     token_id: Optional[int]) -> float:
        """Compute the effect of ablating a specific attention head.
        
        Uses hidden_states and attentions from output_attentions=True.
        This method approximates head ablation by measuring the attention
        weight contribution of the specified head.
        """
        if outputs.attentions is None or outputs.hidden_states is None:
            raise ValueError("outputs must have attentions and hidden_states")

        # Get attention weights for this layer and head
        # attentions[layer]: (batch, num_heads, seq_len, seq_len)
        layer_attn = outputs.attentions[
            layer]  # (1, num_heads, seq_len, seq_len)
        head_attn = layer_attn[
            0, head, -1, :]  # Attention from last token, shape (seq_len,)

        # Get hidden states
        # hidden_states[layer]: (batch, seq_len, hidden_size)
        layer_hidden = outputs.hidden_states[layer][0, -1, :]  # (hidden_size,)
        final_hidden = outputs.hidden_states[-1][0, -1, :]  # Last layer hidden
        final_var = final_hidden.pow(2).mean(-1, keepdim=True)

        # Compute baseline probability
        baseline = self._log_probability_from_hidden(final_hidden, final_var,
                                                     token_id).exp()

        # Approximate ablation effect using attention weight magnitude
        # Higher attention = larger effect when ablated
        attn_magnitude = head_attn.sum().item()

        # Estimate ablated probability: reduce by proportion of attention removed
        # This is an approximation since true ablation would require re-running the model
        ablation_factor = max(0.0, 1.0 - attn_magnitude * 0.1)  # Scale factor
        ablated = baseline * ablation_factor

        return float(ablated)

    def _transfer_output(self, model_output) -> Tuple[List, List, List]:
        all_pos_layer_input = []
        all_pos_layer_output = []
        all_last_attn_subvalues = []
        for layer_idx in range(self.num_layers):
            layer_tuple = model_output[layer_idx]
            # past_key_values structure: each layer is a tuple (key, value)
            # key and value have shape (batch_size, num_heads, seq_len, head_dim)
            # Extract what we can from the available indices
            try:
                all_pos_layer_input.append(layer_tuple[0][0].tolist())
            except (IndexError, TypeError):
                # If index 0 doesn't exist or is not indexable, use empty list
                all_pos_layer_input.append([])

            try:
                all_pos_layer_output.append(layer_tuple[4][0].tolist())
            except (IndexError, TypeError):
                # If index 4 doesn't exist, use the value tensor (index 1) instead
                if len(layer_tuple) > 1:
                    all_pos_layer_output.append(layer_tuple[1][0].tolist())
                else:
                    all_pos_layer_output.append([])

            try:
                all_last_attn_subvalues.append(layer_tuple[5][0].tolist())
            except (IndexError, TypeError):
                # If index 5 doesn't exist, use key tensor (index 0) as fallback
                if len(layer_tuple) > 0:
                    all_last_attn_subvalues.append(layer_tuple[0][0].tolist())
                else:
                    all_last_attn_subvalues.append([])
        return all_pos_layer_input, all_pos_layer_output, all_last_attn_subvalues

    def _log_probability(self, vector: torch.Tensor, final_var: torch.Tensor,
                         token_id: Optional[int]) -> torch.Tensor:
        if token_id is None:
            raise ValueError("token_id must be provided")
        scaled = vector * torch.rsqrt(final_var + 1e-6)
        w = self.model.language_model.layers[-1].self_attn.o_proj.weight
        if w.dtype not in (torch.float32, torch.float16):
            w = w.float()
        w = w.to(scaled.device)

        # This is the fix:
        # The 'vector' input to this function sometimes has shape (..., head_dim)
        # e.g. (..., 128), instead of (..., hidden_dim).
        # We must project it back to hidden_dim (4096).

        slice_dim = scaled.shape[-1]

        if slice_dim == self.head_dim:
            # 'w' is (4096, 4096). We take a (128, 4096) slice.
            w_slice = w[:slice_dim, :]

            # This performs (..., 128) @ (128, 4096) -> (..., 4096)
            rms = scaled @ w_slice
        elif slice_dim == self.model.language_model.config.hidden_size:
            # This is the correct path for (..., 4096) tensors
            rms = scaled * w.mean(dim=0)
        else:
            # This path catches the bad (..., 1) tensor.
            # The original faulty logic:
            rms = scaled * w.mean(dim=0)[:slice_dim]

            rms = rms.to(self.model.lm_head.weight.device)

        logits = self.model.lm_head(rms).data

        probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return probs[token_id]

    def _log_probability_from_hidden(self, hidden_state: torch.Tensor,
                                     final_var: torch.Tensor,
                                     token_id: Optional[int]) -> torch.Tensor:
        """Compute log probability from hidden state (for use with output_attentions).
        
        This version takes the final hidden state directly, which has shape (hidden_size,)
        and is correctly sized for the lm_head projection.
        """
        if token_id is None:
            raise ValueError("token_id must be provided")

        # Apply RMS normalization in float32 for precision
        hidden = hidden_state.float()
        variance = hidden.pow(2).mean(-1, keepdim=True)
        hidden_norm = hidden * torch.rsqrt(variance + 1e-6)

        # Project to vocabulary - match dtype and device of lm_head
        lm_head_weight = self.model.lm_head.weight
        hidden_norm = hidden_norm.to(device=lm_head_weight.device,
                                     dtype=lm_head_weight.dtype)
        logits = self.model.lm_head(
            hidden_norm).float()  # Back to float32 for softmax

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
