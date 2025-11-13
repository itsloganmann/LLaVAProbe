"""Metric utilities for attention and confidence analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch

from .config import CalibrationConfig, EntropyConfig


@dataclass
class AttentionEntropyMetrics:
    """Entropy measurements over attention distributions."""

    raw_entropy: float
    normalized_entropy: float
    num_elements: int


@dataclass
class ConfidenceBreakdown:
    """Calibration summary for a question subset."""

    subset: str
    num_examples: int
    ece: float
    brier: float


@dataclass
class ConfidenceMetrics:
    """Per-example confidence metrics."""

    predicted_tokens: Sequence[str]
    predicted_probabilities: Sequence[float]
    token_entropy: float
    top1_probability: float
    probability_margin: float
    logit_margin: float
    sequence_probability: float
    per_token_log_probs: Sequence[float]
    ground_truth_tokens: Optional[Sequence[str]] = None


@dataclass
class ConfidenceRecord:
    subset: str
    probability: float
    is_correct: bool


class ConfidenceCalibrator:
    """Accumulates calibration statistics for multiple subsets."""

    def __init__(self, config: CalibrationConfig) -> None:
        self._config = config
        self._records: List[ConfidenceRecord] = []

    def add(self, subset: str, probability: float, is_correct: bool) -> None:
        self._records.append(ConfidenceRecord(subset=subset, probability=probability, is_correct=is_correct))

    def summarize(self) -> List[ConfidenceBreakdown]:
        summaries: List[ConfidenceBreakdown] = []
        for subset in self._config.subsets:
            subset_records = [r for r in self._records if r.subset == subset]
            if not subset_records:
                continue
            ece = self._expected_calibration_error(subset_records)
            brier = self._brier_score(subset_records)
            summaries.append(
                ConfidenceBreakdown(
                    subset=subset,
                    num_examples=len(subset_records),
                    ece=ece,
                    brier=brier,
                )
            )
        return summaries

    def _expected_calibration_error(self, records: Sequence[ConfidenceRecord]) -> float:
        if not records:
            return 0.0
        bin_edges = np.linspace(0.0, 1.0, self._config.num_bins + 1)
        total = len(records)
        ece = 0.0
        for b_start, b_end in zip(bin_edges[:-1], bin_edges[1:]):
            bin_records = [r for r in records if b_start <= r.probability < b_end or (b_end == 1.0 and r.probability == 1.0)]
            if not bin_records:
                continue
            bin_acc = np.mean([1.0 if r.is_correct else 0.0 for r in bin_records])
            bin_conf = np.mean([r.probability for r in bin_records])
            weight = len(bin_records) / total
            ece += abs(bin_acc - bin_conf) * weight
        return float(ece)

    @staticmethod
    def _brier_score(records: Sequence[ConfidenceRecord]) -> float:
        if not records:
            return 0.0
        errors = [(r.probability - (1.0 if r.is_correct else 0.0)) ** 2 for r in records]
        return float(np.mean(errors))


def compute_token_entropy(probabilities: Sequence[float], config: EntropyConfig) -> float:
    """Compute normalized Shannon entropy over the top-k probabilities."""

    array = torch.as_tensor(probabilities, dtype=torch.float32)
    if array.ndim != 1:
        raise ValueError("probabilities must be a 1D vector")
    if len(array) == 0:
        return 0.0
    top_k = min(config.top_k, array.shape[0])
    values, _ = torch.topk(array, k=top_k, largest=True)
    normalized = values / values.sum()
    probs = normalized.clamp_min(config.epsilon)
    entropy = -(probs * probs.log()).sum().item()
    if not config.normalize or top_k <= 1:
        return float(entropy)
    return float(entropy / np.log(top_k))


def compute_attention_entropy(attention_map: np.ndarray, config: EntropyConfig) -> AttentionEntropyMetrics:
    """Compute entropy over a dense attention map."""

    flattened = attention_map.astype(np.float64).reshape(-1)
    total = float(flattened.sum())
    if total <= 0.0:
        return AttentionEntropyMetrics(raw_entropy=0.0, normalized_entropy=0.0, num_elements=len(flattened))
    probs = flattened / total
    probs = np.clip(probs, config.epsilon, None)
    raw_entropy = float(-(probs * np.log(probs)).sum())
    normalized = raw_entropy / float(np.log(len(flattened))) if config.normalize and len(flattened) > 1 else raw_entropy
    return AttentionEntropyMetrics(raw_entropy=raw_entropy, normalized_entropy=normalized, num_elements=len(flattened))


def compute_confidence_metrics(
    logits: torch.Tensor,
    probabilities: torch.Tensor,
    tokenizer,
    predicted_token_ids: Sequence[int],
    ground_truth_token_ids: Optional[Sequence[int]] = None,
    entropy_config: Optional[EntropyConfig] = None,
) -> ConfidenceMetrics:
    """Compute per-example confidence metrics.

    Args:
        logits: Tensor of shape (sequence_length, vocab_size).
        probabilities: Softmax probabilities of the same shape.
        tokenizer: Tokenizer used to decode token ids.
        predicted_token_ids: Token ids of the generated answer sequence.
        ground_truth_token_ids: Optional token ids of ground-truth answer sequence.
        entropy_config: Configuration controlling entropy normalisation.
    """

    if logits.ndim != 2 or probabilities.ndim != 2:
        raise ValueError("logits and probabilities must be 2D tensors")
    if logits.shape != probabilities.shape:
        raise ValueError("logits and probabilities must share shape")

    entropy_cfg = entropy_config or EntropyConfig()
    predicted_tokens = [tokenizer.decode([idx]).strip() for idx in predicted_token_ids]
    ground_truth_tokens = (
        [tokenizer.decode([idx]).strip() for idx in ground_truth_token_ids]
        if ground_truth_token_ids is not None
        else None
    )

    # Compute per-token log probabilities
    device = probabilities.device
    token_indices = torch.tensor(predicted_token_ids, dtype=torch.long, device=device)
    gather_probs = torch.gather(probabilities, dim=1, index=token_indices.unsqueeze(-1)).squeeze(-1)
    gather_logits = torch.gather(logits, dim=1, index=token_indices.unsqueeze(-1)).squeeze(-1)

    gather_logits = torch.gather(logits, dim=1, index=token_indices.unsqueeze(-1)).squeeze(-1)

    per_token_log_probs = gather_probs.clamp_min(entropy_cfg.epsilon).log().tolist()

    top1_probability = float(gather_probs[-1]) if len(gather_probs) > 0 else 0.0
    top2_probability = _second_best_probability(probabilities[-1])
    probability_margin = float(top1_probability - top2_probability)

    top1_logit = float(gather_logits[-1]) if len(gather_logits) > 0 else 0.0
    top2_logit = _second_best_logit(logits[-1])
    logit_margin = float(top1_logit - top2_logit)

    token_entropy = compute_token_entropy(probabilities[-1].tolist(), entropy_cfg)
    sequence_prob = float(torch.exp(torch.tensor(per_token_log_probs).sum())) if per_token_log_probs else 0.0

    return ConfidenceMetrics(
        predicted_tokens=predicted_tokens,
        predicted_probabilities=gather_probs.tolist(),
        token_entropy=token_entropy,
        top1_probability=top1_probability,
        probability_margin=probability_margin,
        logit_margin=logit_margin,
        sequence_probability=sequence_prob,
        per_token_log_probs=per_token_log_probs,
        ground_truth_tokens=ground_truth_tokens,
    )


def _second_best_probability(probabilities: torch.Tensor) -> float:
    if probabilities.numel() < 2:
        return 0.0
    values, _ = torch.topk(probabilities, k=2)
    return float(values[1]) if len(values) > 1 else 0.0


def _second_best_logit(logits: torch.Tensor) -> float:
    if logits.numel() < 2:
        return 0.0
    values, _ = torch.topk(logits, k=2)
    return float(values[1]) if len(values) > 1 else 0.0