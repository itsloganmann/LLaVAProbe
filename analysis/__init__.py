"""Analysis utilities for attention clustering and confidence metrics."""

from .config import AnalysisConfig
from .llava_runner import LlavaRunner, RunnerMode
from .metrics import (
    AttentionEntropyMetrics,
    ConfidenceBreakdown,
    ConfidenceMetrics,
    ConfidenceCalibrator,
    compute_attention_entropy,
    compute_confidence_metrics,
    compute_token_entropy,
)
from .clustering import (
    ClusterSweepResult,
    ClustererType,
    ClusterResult,
    ClusteringPipeline,
    NullModelResult,
)
from .ablations import AblationExperiment, AblationResult
from .outputs import AnalysisRecord, AnalysisWriter, ClusterReport, StructuredOutputs

__all__ = [
    "AnalysisConfig",
    "LlavaRunner",
    "RunnerMode",
    "AttentionEntropyMetrics",
    "ConfidenceBreakdown",
    "ConfidenceMetrics",
    "ConfidenceCalibrator",
    "compute_attention_entropy",
    "compute_token_entropy",
    "compute_confidence_metrics",
    "ClusterSweepResult",
    "ClustererType",
    "ClusterResult",
    "ClusteringPipeline",
    "NullModelResult",
    "AblationExperiment",
    "AblationResult",
    "AnalysisRecord",
    "AnalysisWriter",
    "ClusterReport",
    "StructuredOutputs",
]
