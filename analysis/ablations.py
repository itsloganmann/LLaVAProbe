"""Definitions for ablation experiments applied to the attention pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class AblationExperiment:
    """Metadata describing a specific ablation experiment."""

    name: str
    description: str
    parameters: Dict[str, Any]


@dataclass
class AblationResult:
    """Outcome for a single ablation compared to baseline runs."""

    experiment: AblationExperiment
    baseline_confidence: float
    ablated_confidence: float
    confidence_delta: float
    additional_metrics: Dict[str, float]


def visual_dropout_mask(num_patches: int, dropout_rate: float, *, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Create a Bernoulli dropout mask for spatial patches."""

    generator = rng or np.random.default_rng()
    mask = generator.random(num_patches) >= dropout_rate
    return mask.astype(np.float32)