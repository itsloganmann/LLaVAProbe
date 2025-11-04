"""Configuration objects for the analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class ClusterSweepConfig:
    """Hyper-parameter sweep configuration for clustering methods."""

    eps_values: Sequence[float] = field(default_factory=lambda: np.round(np.linspace(0.8, 2.0, num=13), 2).tolist())
    min_samples_values: Sequence[int] = field(default_factory=lambda: list(range(5, 31, 5)))
    weight_exponents: Sequence[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])


@dataclass
class EntropyConfig:
    """Configuration for entropy calculations."""

    top_k: int = 10
    epsilon: float = 1e-12
    normalize: bool = True


@dataclass
class CalibrationConfig:
    """Configuration for confidence calibration metrics."""

    num_bins: int = 10
    subsets: Tuple[str, ...] = ("yes/no", "short_answer")


@dataclass
class AblationConfig:
    """Configuration controlling ablation experiments."""

    language_only: bool = False
    visual_dropout_rates: Sequence[float] = (0.0, 0.2, 0.5)
    head_dropout_trials: int = 3


@dataclass
class PatchSweepConfig:
    """Image resolution / patch size sweep configuration."""

    resolutions: Sequence[int] = (224, 336, 448)


@dataclass
class AnalysisConfig:
    """High-level configuration for running the full analysis pipeline."""

    cluster_sweep: ClusterSweepConfig = field(default_factory=ClusterSweepConfig)
    entropy: EntropyConfig = field(default_factory=EntropyConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    ablations: AblationConfig = field(default_factory=AblationConfig)
    patch_sweep: PatchSweepConfig = field(default_factory=PatchSweepConfig)
    null_model_samples: int = 30
    random_seed: int = 42
    save_intermediate_artifacts: bool = True

    def iter_eps_min_samples(self) -> Iterable[Tuple[float, int]]:
        for eps in self.cluster_sweep.eps_values:
            for min_samples in self.cluster_sweep.min_samples_values:
                yield eps, min_samples

    def iter_weight_exponents(self) -> Iterable[float]:
        yield from self.cluster_sweep.weight_exponents
