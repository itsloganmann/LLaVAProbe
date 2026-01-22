"""
Qwen3-VL Runner for attention analysis and probing.

Optimized for A100 GPU (80GB VRAM) in Google Colab environment.
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


class QwenVLRunner:
    """
    Runner for Qwen2.5-VL / Qwen3-VL models with attention extraction.
    
    Optimized for A100 GPU with 80GB VRAM.
    """
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        use_flash_attention: bool = True,
        num_image_tokens: int = 256,
        num_system_tokens: int = 10,
        attention_grid_size: int = 24,
    ) -> None:
        """
        Initialize the Qwen-VL runner.
        
        Args:
            model_id: HuggingFace model ID. Options:
                - "Qwen/Qwen2.5-VL-7B-Instruct" (default, 7B params)
                - "Qwen/Qwen2.5-VL-72B-Instruct" (72B params, needs multi-GPU)
            device: Device to use. Defaults to "cuda" if available.
            torch_dtype: Data type for model. bfloat16 recommended for A100.
            use_flash_attention: Use Flash Attention 2 for efficiency.
            num_image_tokens: Expected number of image tokens (256-1024 depending on resolution).
            num_system_tokens: Number of system/prompt tokens before image tokens.
            attention_grid_size: Target size for attention map visualization grid.
        """
        from transformers import AutoModelForVision2Seq, AutoProcessor
        
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.num_image_tokens = num_image_tokens
        self.num_system_tokens = num_system_tokens
        self.attention_grid_size = attention_grid_size
        
        print(f"Loading {model_id} on {self.device}...")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load model with optimizations for A100
        # Use AutoModelForVision2Seq to auto-detect correct class (Qwen2VL vs Qwen2_5_VL)
        attn_impl = "flash_attention_2" if use_flash_attention else "eager"
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        self.model.eval()
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        
        # Model configuration
        self.config = self.model.config
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        
        print(f"Model loaded: {self.num_layers} layers, {self.num_heads} heads")
        print(f"Hidden size: {self.hidden_size}, Head dim: {self.head_dim}")
    
    def _prepare_messages(
        self,
        image: Image.Image,
        question: str,
    ) -> List[Dict[str, Any]]:
        """Prepare messages in Qwen-VL chat format."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        return messages
    
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
            question: Question about the image.
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
        
        # Apply mode-specific modifications
        if mode == RunnerMode.LANGUAGE_ONLY:
            # Create blank image
            image = Image.new("RGB", image.size, (128, 128, 128))
        elif mode == RunnerMode.VISUAL_DROPOUT:
            # Apply random dropout to image patches
            image = self._apply_visual_dropout(image, dropout_rate=0.3)
        
        # Prepare inputs
        messages = self._prepare_messages(image, question)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Get input length for slicing generated tokens
        input_len = inputs["input_ids"].shape[1]
        
        # Forward pass with attention and hidden states
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
        
        # Token information
        token_ids = generated_ids.tolist()
        token_strings = [
            self.processor.decode([tid]).strip() for tid in token_ids
        ]
        
        # Compute token probabilities from scores
        if outputs.scores:
            token_logits = torch.stack(outputs.scores).squeeze(1)
            token_probs = torch.softmax(token_logits, dim=-1)
            
            # Get confidence for predicted tokens
            confidences = []
            for i, tid in enumerate(token_ids):
                if i < len(token_probs):
                    confidences.append(token_probs[i, tid].item())
            token_confidence = np.mean(confidences) if confidences else 0.0
        else:
            token_logits = torch.zeros(len(token_ids), self.config.vocab_size)
            token_probs = torch.zeros_like(token_logits)
            token_confidence = 0.0
        
        # Extract attention maps
        attention_maps = {}
        aggregated_attention = np.zeros((1, 1))
        layer_contributions = {}
        head_contributions = {}
        
        if extract_attention and hasattr(outputs, 'attentions') and outputs.attentions:
            attention_maps, aggregated_attention, layer_contributions, head_contributions = \
                self._extract_attention_analysis(outputs.attentions, inputs)
        
        # Extract hidden states
        hidden_states = None
        if extract_hidden_states and hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            # Get hidden states from first generated token
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
        inputs: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[int, float], Dict[str, float]]:
        """
        Extract and analyze attention patterns.
        
        Returns:
            - attention_maps: Dict mapping "layer_head" to attention array
            - aggregated_attention: Combined attention visualization
            - layer_contributions: Per-layer contribution scores
            - head_contributions: Per-head contribution scores
        """
        attention_maps = {}
        layer_contributions = {}
        head_contributions = {}
        
        all_image_attention = []
        
        for layer_idx, layer_attn in enumerate(attentions):
            if layer_attn is None:
                continue
                
            # layer_attn shape: (batch, num_heads, seq_len, seq_len)
            layer_attn = layer_attn[0]  # Remove batch dim
            
            layer_score = 0.0
            
            for head_idx in range(layer_attn.shape[0]):
                # Get attention from last token to all previous
                head_attn = layer_attn[head_idx, -1, :].cpu().numpy()
                
                # Extract attention to image tokens (approximate region)
                # Image tokens typically come after system tokens
                start_idx = self.num_system_tokens
                end_idx = min(start_idx + self.num_image_tokens, len(head_attn))
                
                if end_idx > start_idx:
                    image_attn = head_attn[start_idx:end_idx]
                    
                    # Normalize
                    if image_attn.sum() > 0:
                        image_attn = image_attn / image_attn.sum()
                    
                    # Reshape to square grid (approximate)
                    grid_size = int(np.sqrt(len(image_attn)))
                    if grid_size > 0:
                        cropped_len = grid_size * grid_size
                        attn_grid = image_attn[:cropped_len].reshape(grid_size, grid_size)
                        
                        key = f"{layer_idx}_{head_idx}"
                        attention_maps[key] = attn_grid
                        
                        # Compute contribution score (entropy-based)
                        entropy = -np.sum(image_attn * np.log(image_attn + 1e-10))
                        variance = np.var(image_attn)
                        score = variance * (1 + entropy)
                        
                        head_contributions[key] = score
                        layer_score += score
                        
                        all_image_attention.append(attn_grid)
            
            layer_contributions[layer_idx] = layer_score
        
        # Aggregate attention across all heads (weighted by contribution)
        if all_image_attention:
            # Resize all to common size for aggregation
            target_size = self.attention_grid_size
            resized = []
            for attn in all_image_attention:
                from scipy.ndimage import zoom
                if attn.shape[0] > 0 and attn.shape[1] > 0:
                    scale = (target_size / attn.shape[0], target_size / attn.shape[1])
                    resized.append(zoom(attn, scale, order=1))
            
            if resized:
                aggregated = np.mean(resized, axis=0)
                aggregated = aggregated / aggregated.sum() if aggregated.sum() > 0 else aggregated
            else:
                aggregated = np.zeros((target_size, target_size))
        else:
            aggregated = np.zeros((self.attention_grid_size, self.attention_grid_size))
        
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
        patch_size = 32  # Dropout at patch level
        
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                if random.random() < dropout_rate:
                    img_array[i:i+patch_size, j:j+patch_size] = 128
        
        return Image.fromarray(img_array)
    
    def run_logit_lens(
        self,
        image: Image.Image,
        question: str,
    ) -> Dict[str, Any]:
        """
        Run logit lens analysis to understand information flow across layers.
        
        Returns per-layer predictions and confidence trajectories.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        messages = self._prepare_messages(image, question)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Forward with hidden states
        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        
        hidden_states = outputs.hidden_states
        
        # Apply LM head to each layer's hidden state
        layer_predictions = {}
        layer_confidences = {}
        
        lm_head = self.model.lm_head
        
        for layer_idx, hidden in enumerate(hidden_states):
            # Get last token's hidden state
            last_hidden = hidden[0, -1, :]
            
            # Project to vocabulary
            logits = lm_head(last_hidden.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1)
            
            # Top prediction
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
