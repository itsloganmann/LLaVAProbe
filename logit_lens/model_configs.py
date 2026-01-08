"""
Multi-Model Logit Lens Configuration

This file defines the models and datasets for cross-model comparison experiments.
"""

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

MODELS = {
    "llava-1.5-7b": {
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "model_class": "LlavaForConditionalGeneration",
        "processor_class": "AutoProcessor",
        "architecture": "llava",
        "hidden_dim": 4096,
        "n_layers": 32,
        "mlp_dim": 11008,
        "description": "LLaVA 1.5 7B - Llama2 backbone",
    },
    "llava-1.5-13b": {
        "model_id": "llava-hf/llava-1.5-13b-hf",
        "model_class": "LlavaForConditionalGeneration",
        "processor_class": "AutoProcessor",
        "architecture": "llava",
        "hidden_dim": 5120,
        "n_layers": 40,
        "mlp_dim": 13824,
        "description": "LLaVA 1.5 13B - Llama2 backbone (larger)",
    },
    "qwen2-vl-7b": {
        "model_id": "Qwen/Qwen2-VL-7B-Instruct",
        "model_class": "Qwen2VLForConditionalGeneration",
        "processor_class": "AutoProcessor",
        "architecture": "qwen2_vl",
        "hidden_dim": 3584,
        "n_layers": 28,
        "mlp_dim": 18944,
        "description":
        "Qwen2-VL 7B - Different architecture, strong multilingual",
    },
    "paligemma2-3b": {
        "model_id": "google/paligemma2-3b-pt-224",
        "model_class": "PaliGemmaForConditionalGeneration",
        "processor_class": "AutoProcessor",
        "architecture": "paligemma",
        "hidden_dim": 2048,
        "n_layers": 18,
        "mlp_dim": 16384,
        "description": "PaliGemma 2 3B - Google's VLM, smaller but efficient",
    },
}

# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================

DATASETS = {
    "vqav2": {
        "name": "VQAv2",
        "hf_path": "HuggingFaceM4/VQAv2",
        "split": "validation",
        "task": "vqa",
        "image_key": "image",
        "question_key": "question",
        "answer_key": "answers",
        "description":
        "Visual Question Answering v2 - multiple choice answers",
    },
    "coco_captions": {
        "name": "COCO Captions",
        "hf_path": "HuggingFaceM4/COCO",
        "split": "validation",
        "task": "captioning",
        "image_key": "image",
        "caption_key": "sentences",
        "description": "COCO image captioning dataset",
    },
    "pope": {
        "name":
        "POPE",
        "hf_path":
        "lmms-lab/POPE",
        "split":
        "test",
        "task":
        "hallucination",
        "image_key":
        "image",
        "question_key":
        "question",
        "answer_key":
        "answer",
        "description":
        "Polling-based Object Probing Evaluation - hallucination detection",
    },
    "textvqa": {
        "name": "TextVQA",
        "hf_path": "textvqa",
        "split": "validation",
        "task": "vqa",
        "image_key": "image",
        "question_key": "question",
        "answer_key": "answers",
        "description": "VQA requiring text reading in images",
    },
}

# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

EXPERIMENTS = {
    "default": {
        "n_samples":
        1000,
        "steps": [
            "image_comparison",  # With vs without image
            "visual_attribution",  # MLP vs Attention breakdown
            "question_type",  # Per question-type analysis
            "neuron_analysis",  # Top predictive neurons
            "ablation_tests",  # Causal ablation
        ],
        "save_intermediate":
        True,
    },
    "quick": {
        "n_samples": 200,
        "steps": [
            "image_comparison",
            "visual_attribution",
        ],
        "save_intermediate": False,
    },
    "full": {
        "n_samples":
        2000,
        "steps": [
            "image_comparison",
            "visual_attribution",
            "question_type",
            "neuron_analysis",
            "ablation_tests",
        ],
        "save_intermediate":
        True,
    },
}

# =============================================================================
# ARCHITECTURE-SPECIFIC HOOK CONFIGS
# =============================================================================

HOOK_CONFIGS = {
    "llava": {
        "residual_path": "model.layers.{}.post_attention_layernorm",
        "attn_path": "model.layers.{}.self_attn",
        "mlp_path": "model.layers.{}.mlp",
        "lm_head_path": "lm_head",
    },
    "qwen2_vl": {
        "residual_path": "model.layers.{}.post_attention_layernorm",
        "attn_path": "model.layers.{}.self_attn",
        "mlp_path": "model.layers.{}.mlp",
        "lm_head_path": "lm_head",
    },
    "paligemma": {
        "residual_path":
        "language_model.model.layers.{}.post_attention_layernorm",
        "attn_path": "language_model.model.layers.{}.self_attn",
        "mlp_path": "language_model.model.layers.{}.mlp",
        "lm_head_path": "language_model.lm_head",
    },
}


def get_model_config(model_name: str) -> dict:
    """Get configuration for a specific model."""
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    return MODELS[model_name]


def get_dataset_config(dataset_name: str) -> dict:
    """Get configuration for a specific dataset."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}"
        )
    return DATASETS[dataset_name]


def get_hook_config(architecture: str) -> dict:
    """Get hook paths for a specific architecture."""
    if architecture not in HOOK_CONFIGS:
        raise ValueError(
            f"Unknown architecture: {architecture}. Available: {list(HOOK_CONFIGS.keys())}"
        )
    return HOOK_CONFIGS[architecture]


if __name__ == "__main__":
    print("Available models:")
    for name, config in MODELS.items():
        print(f"  {name}: {config['description']}")

    print("\nAvailable datasets:")
    for name, config in DATASETS.items():
        print(f"  {name}: {config['description']}")
