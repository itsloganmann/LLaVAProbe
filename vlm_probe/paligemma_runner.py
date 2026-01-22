"""
PaliGemma2 Runner for attention analysis and probing.

Optimized for A100 GPU (80GB VRAM) in Google Colab environment.
Requires HuggingFace gated access to PaliGemma2.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


class RunnerMode(str, enum.Enum):
    """Operating modes for the runner."""
    FULL = "full"
    LANGUAGE_ONLY = "language_only"
    VISUAL_DROPOUT = "visual_dropout"
    MASK_HIGH_ATTENTION = "mask_high_attention"
    MASK_LOW_ATTENTION = "mask_low_attention"


@dataclass
class RunnerOutput:
    """Outputs produced for a single prompt/image pair."""
    prompt: str
    predicted_answer: str
    predicted_token_ids: Sequence[int]
    token_strings: Sequence[str]
    token_probabilities: torch.Tensor
    token_logits: torch.Tensor
    attention_maps: Dict[str, np.ndarray]  # layer_head -> attention map
    aggregated_attention: np.ndarray  # Combined attention visualization
    token_confidence: float
    layer_contributions: Dict[int, float]  # layer -> contribution score
    head_contributions: Dict[str, float]  # layer_head -> contribution score
    hidden_states: Optional[List[torch.Tensor]] = None
    # PaliGemma-specific: SigLIP vision features
    vision_features: Optional[torch.Tensor] = None


class PaliGemmaRunner:
    """
    Runner for PaliGemma2 models with attention extraction.
    
    PaliGemma2 uses SigLIP vision encoder + Gemma2 language model.
    Requires HuggingFace gated access - run `huggingface-cli login` first.
    
    Optimized for A100 GPU with 80GB VRAM.
    """
    
    def __init__(
        self,
        model_id: str = "google/paligemma2-3b-pt-224",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """
        Initialize the PaliGemma2 runner.
        
        Args:
            model_id: HuggingFace model ID. Options:
                - "google/paligemma2-3b-pt-224" (3B, 224px, pretrained)
                - "google/paligemma2-3b-pt-448" (3B, 448px, pretrained)
                - "google/paligemma2-3b-pt-896" (3B, 896px, pretrained)
                - "google/paligemma2-10b-pt-224" (10B params)
                - "google/paligemma2-28b-pt-224" (28B params)
            device: Device to use. Defaults to "cuda" if available.
            torch_dtype: Data type for model. bfloat16 recommended for A100.
        """
        from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
        
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        
        print(f"Loading {model_id} on {self.device}...")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        print("\nNote: PaliGemma2 requires HuggingFace gated access.")
        print("Run 'huggingface-cli login' if you encounter access errors.\n")
        
        # Load model
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        
        # Model configuration
        self.text_config = self.model.config.text_config
        self.vision_config = self.model.config.vision_config
        
        self.num_layers = self.text_config.num_hidden_layers
        self.num_heads = self.text_config.num_attention_heads
        self.hidden_size = self.text_config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        
        # Vision encoder info
        self.image_size = self.vision_config.image_size
        self.patch_size = self.vision_config.patch_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.patches_per_side = self.image_size // self.patch_size
        
        print(f"Language model: {self.num_layers} layers, {self.num_heads} heads")
        print(f"Hidden size: {self.hidden_size}, Head dim: {self.head_dim}")
        print(f"Vision encoder: {self.image_size}px, patch size {self.patch_size}")
        print(f"Image tokens: {self.num_patches} ({self.patches_per_side}x{self.patches_per_side} grid)")
    
    @torch.inference_mode()
    def run(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 128,
        mode: RunnerMode = RunnerMode.FULL,
        extract_attention: bool = True,
        extract_hidden_states: bool = True,
    ) -> RunnerOutput:
        """
        Run inference and extract attention patterns.
        
        Args:
            image: Input PIL Image.
            question: Question/prompt about the image.
            max_new_tokens: Maximum tokens to generate.
            mode: Running mode (full, language_only, etc.)
            extract_attention: Whether to extract attention maps.
            extract_hidden_states: Whether to extract hidden states.
            
        Returns:
            RunnerOutput with predictions and attention analysis.
        """
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # PaliGemma expects specific image size
        image = image.resize((self.image_size, self.image_size))
        
        # Apply mode-specific modifications
        if mode == RunnerMode.LANGUAGE_ONLY:
            image = Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))
        elif mode == RunnerMode.VISUAL_DROPOUT:
            image = self._apply_visual_dropout(image, dropout_rate=0.3)
        
        # Prepare inputs - PaliGemma uses simple prompt format
        prompt = question
        
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        input_len = inputs["input_ids"].shape[1]
        
        # Generate with attention extraction
        # Note: For attention, we need eager attention implementation
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            output_attentions=extract_attention,
            output_hidden_states=extract_hidden_states,
        )
        
        # Extract generated tokens
        generated_ids = outputs.sequences[0, input_len:]
        predicted_answer = self.processor.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        
        token_ids = generated_ids.tolist()
        token_strings = [
            self.processor.decode([tid]).strip() for tid in token_ids
        ]
        
        # Compute token probabilities
        if outputs.scores:
            token_logits = torch.stack(outputs.scores).squeeze(1)
            token_probs = torch.softmax(token_logits, dim=-1)
            
            confidences = []
            for i, tid in enumerate(token_ids):
                if i < len(token_probs):
                    confidences.append(token_probs[i, tid].item())
            token_confidence = np.mean(confidences) if confidences else 0.0
        else:
            token_logits = torch.zeros(len(token_ids), self.text_config.vocab_size)
            token_probs = torch.zeros_like(token_logits)
            token_confidence = 0.0
        
        # Extract attention maps
        attention_maps = {}
        aggregated_attention = np.zeros((self.patches_per_side, self.patches_per_side))
        layer_contributions = {}
        head_contributions = {}
        
        if extract_attention and hasattr(outputs, 'attentions') and outputs.attentions:
            attention_maps, aggregated_attention, layer_contributions, head_contributions = \
                self._extract_attention_analysis(outputs.attentions)
        
        # Extract hidden states
        hidden_states = None
        if extract_hidden_states and hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            if outputs.hidden_states:
                hidden_states = [
                    hs[0, -1, :].cpu() for hs in outputs.hidden_states[0]
                ]
        
        return RunnerOutput(
            prompt=question,
            predicted_answer=predicted_answer,
            predicted_token_ids=token_ids,
            token_strings=token_strings,
            token_probabilities=token_probs.cpu(),
            token_logits=token_logits.cpu(),
            attention_maps=attention_maps,
            aggregated_attention=aggregated_attention,
            token_confidence=token_confidence,
            layer_contributions=layer_contributions,
            head_contributions=head_contributions,
            hidden_states=hidden_states,
        )
    
    def _extract_attention_analysis(
        self,
        attentions: Tuple,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[int, float], Dict[str, float]]:
        """
        Extract and analyze attention patterns for PaliGemma2.
        
        PaliGemma2 places image tokens at the beginning of the sequence.
        """
        attention_maps = {}
        layer_contributions = {}
        head_contributions = {}
        
        all_image_attention = []
        
        for layer_idx, layer_attn in enumerate(attentions):
            if layer_attn is None:
                continue
            
            # Handle case where layer_attn might be a tuple or have unexpected structure
            try:
                # layer_attn: (batch, num_heads, seq_len, seq_len)
                if isinstance(layer_attn, tuple):
                    layer_attn = layer_attn[0]
                if layer_attn is None:
                    continue
                layer_attn = layer_attn[0]  # Remove batch dim
                if layer_attn is None or not hasattr(layer_attn, 'shape'):
                    continue
            except (IndexError, TypeError, AttributeError):
                continue
            
            layer_score = 0.0
            
            for head_idx in range(layer_attn.shape[0]):
                # Get attention from last token to all previous
                head_attn = layer_attn[head_idx, -1, :].cpu().numpy()
                
                # Image tokens are at positions 0 to num_patches-1
                image_attn = head_attn[:self.num_patches]
                
                # Normalize
                if image_attn.sum() > 0:
                    image_attn = image_attn / image_attn.sum()
                
                # Reshape to grid
                attn_grid = image_attn.reshape(self.patches_per_side, self.patches_per_side)
                
                key = f"{layer_idx}_{head_idx}"
                attention_maps[key] = attn_grid
                
                # Compute contribution score
                entropy = -np.sum(image_attn * np.log(image_attn + 1e-10))
                variance = np.var(image_attn)
                score = variance * (1 + entropy)
                
                head_contributions[key] = score
                layer_score += score
                
                all_image_attention.append(attn_grid)
            
            layer_contributions[layer_idx] = layer_score
        
        # Aggregate attention
        if all_image_attention:
            aggregated = np.mean(all_image_attention, axis=0)
            aggregated = aggregated / aggregated.sum() if aggregated.sum() > 0 else aggregated
        else:
            aggregated = np.zeros((self.patches_per_side, self.patches_per_side))
        
        return attention_maps, aggregated, layer_contributions, head_contributions
    
    def _apply_visual_dropout(
        self,
        image: Image.Image,
        dropout_rate: float = 0.3,
    ) -> Image.Image:
        """Apply random patch dropout to image."""
        import random
        
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        for i in range(0, h, self.patch_size):
            for j in range(0, w, self.patch_size):
                if random.random() < dropout_rate:
                    img_array[i:i+self.patch_size, j:j+self.patch_size] = 128
        
        return Image.fromarray(img_array)
    
    def run_logit_lens(
        self,
        image: Image.Image,
        question: str,
    ) -> Dict[str, Any]:
        """
        Run logit lens analysis to understand information flow across layers.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        
        inputs = self.processor(
            text=question,
            images=image,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        
        hidden_states = outputs.hidden_states
        
        layer_predictions = {}
        layer_confidences = {}
        
        # Get LM head from the language model
        lm_head = self.model.language_model.lm_head
        
        for layer_idx, hidden in enumerate(hidden_states):
            last_hidden = hidden[0, -1, :]
            
            logits = lm_head(last_hidden.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1)
            
            top_prob, top_idx = probs[0].max(dim=-1)
            top_token = self.processor.decode([top_idx.item()])
            
            layer_predictions[layer_idx] = {
                "token": top_token,
                "token_id": top_idx.item(),
                "probability": top_prob.item(),
            }
            layer_confidences[layer_idx] = top_prob.item()
        
        return {
            "layer_predictions": layer_predictions,
            "layer_confidences": layer_confidences,
            "num_layers": len(hidden_states),
        }
    
    def extract_vision_features(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        """
        Extract SigLIP vision encoder features.
        
        Useful for analyzing what visual information is captured.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        
        inputs = self.processor(
            text="",
            images=image,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            # Get vision encoder output
            vision_outputs = self.model.vision_tower(
                inputs["pixel_values"],
                output_hidden_states=True,
            )
            
        return vision_outputs.last_hidden_state.cpu()
