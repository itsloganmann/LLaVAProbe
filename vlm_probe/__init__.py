"""VLM Probe: Attention analysis for Qwen3-VL and PaliGemma2 models."""

from .qwen_runner import QwenVLRunner
from .paligemma_runner import PaliGemmaRunner
from .analysis import VLMAnalyzer, run_comparative_analysis
from .visualizations import (
    plot_attention_heatmap,
    plot_layer_attribution,
    plot_confidence_calibration,
    plot_cross_model_comparison,
    plot_attention_entropy_analysis,
)

__all__ = [
    "QwenVLRunner",
    "PaliGemmaRunner", 
    "VLMAnalyzer",
    "run_comparative_analysis",
    "plot_attention_heatmap",
    "plot_layer_attribution",
    "plot_confidence_calibration",
    "plot_cross_model_comparison",
    "plot_attention_entropy_analysis",
]
