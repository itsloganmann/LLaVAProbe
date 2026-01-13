#!/usr/bin/env python3
"""
Multi-Model Top-K Attention Entropy Analysis

Supports: Qwen2-VL-7B, PaliGemma-3B
Computes per-head Δ log P contributions, selects top-K heads,
and measures attention entropy correlation with token confidence.

Methodology follows the original LLaVA 7B implementation in topk.py:
1. Compute per-head Δ log P = log P(head + residual) - log P(residual)
2. Select top-K heads by Δ log P
3. Aggregate per-PATCH contributions from those heads (not just last token)
4. Normalize to 576-d vector for clustering/entropy analysis

Usage:
    python run_topk_multimodel.py --model qwen2-vl --top_k 5 --n_samples 100
    python run_topk_multimodel.py --model paligemma --top_k 5 --n_samples 100
"""

import torch
import numpy as np
import pandas as pd
import argparse
import os
import json
import time
import requests
from PIL import Image
from collections import Counter
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# Model imports
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


# ============================================================================
# Helper Functions (matching original LLaVA methodology)
# ============================================================================

def get_bsvalues_qwen(vector, model):
    """
    Convert a hidden state vector to logit space using RMSNorm + lm_head.
    Matches the original LLaVA methodology for Qwen2-VL.
    
    Computes the variance from the vector itself (like original LLaVA code).
    Qwen2-VL path: model.model.language_model.norm, model.lm_head
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    vector = vector.to(device).to(dtype)
    
    # Compute variance from this vector (like original LLaVA code)
    final_var = vector.pow(2).mean(-1, keepdim=True)
    
    # RMS normalization: x * rsqrt(variance + eps)
    vector_scaled = vector * torch.rsqrt(final_var + 1e-6)
    
    # Apply learned RMSNorm weight
    norm_weight = model.model.language_model.norm.weight.data.to(device).to(dtype)
    vector_normed = vector_scaled * norm_weight
    
    # Project to vocabulary
    logits = model.lm_head(vector_normed)
    return logits


def get_bsvalues_paligemma(vector, model):
    """
    Convert a hidden state vector to logit space using RMSNorm + lm_head.
    Matches the original LLaVA methodology for PaliGemma.
    
    Computes the variance from the vector itself.
    PaliGemma path: model.language_model.norm, model.lm_head
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    vector = vector.to(device).to(dtype)
    
    # Compute variance from this vector
    final_var = vector.pow(2).mean(-1, keepdim=True)
    
    # RMS normalization
    vector_scaled = vector * torch.rsqrt(final_var + 1e-6)
    
    # Apply learned RMSNorm weight
    norm_weight = model.language_model.norm.weight.data.to(device).to(dtype)
    vector_normed = vector_scaled * norm_weight
    
    # Project to vocabulary
    logits = model.lm_head(vector_normed)
    return logits


def get_prob(logits):
    """Return softmax probabilities over vocabulary."""
    return torch.nn.functional.softmax(logits, dim=-1)

# ============================================================================
# Configuration
# ============================================================================

MODEL_CONFIGS = {
    "qwen2-vl": {
        "model_id": "Qwen/Qwen2-VL-7B-Instruct",
        "num_layers": 28,
        "num_heads": 28,
        "num_kv_heads": 4,  # GQA: 4 KV heads shared across 28 query heads
        "head_dim": 128,
        "hidden_dim": 3584,
        "patch_grid": 24,  # For 336x336 images (dynamic resolution)
        # Note: Qwen2-VL uses dynamic resolution, so actual patch count varies
        # We'll detect image tokens dynamically during inference
    },
    "paligemma": {
        "model_id": "google/paligemma-3b-mix-224",
        "num_layers": 18,
        "num_heads": 8,
        "num_kv_heads": 1,  # GQA with single KV head
        "head_dim": 256,
        "hidden_dim": 2048,
        "patch_grid": 16,  # 224/14 = 16, so 256 patches
        "num_image_tokens": 256,  # Fixed for 224x224 images
        # PaliGemma: image tokens are at the START of the sequence
    },
}

@dataclass
class ClusterData:
    n_points: int
    ave_strength: float


# ============================================================================
# Utility Functions
# ============================================================================

def normalize(vector):
    """Normalize vector to probability distribution."""
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()
    max_val = max(vector)
    min_val = min(vector)
    
    if max_val == min_val:
        if sum(vector) == 0:
            return vector
        return [1.0/len(vector)] * len(vector)
    
    vector1 = [(x - min_val) / (max_val - min_val) for x in vector]
    sum_v = sum(vector1)
    if sum_v == 0:
        return [0.0] * len(vector1)
    return [x / sum_v for x in vector1]


def calculate_entropy(probabilities):
    """Compute normalized Shannon entropy."""
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    N = len(probabilities)
    if N <= 1:
        return 0.0
    max_entropy = np.log2(N)
    if max_entropy == 0:
        return 0.0
    return entropy / max_entropy


def transform_matrix_to_3d_points(array_2d):
    """Convert 2D attention matrix to 3D points (x, y, attention)."""
    rows, cols = array_2d.shape
    result = np.empty([rows * cols, 3], dtype=object)
    for x in range(rows):
        for y in range(cols):
            result[x * cols + y] = [y, -x + rows - 1, array_2d[x, y]]
    return result


def apply_threshold(datapoints, percentile=80):
    """Filter low-attention patches by percentile threshold."""
    if len(datapoints) == 0:
        return datapoints
    z_values = datapoints[:, 2]
    p_value = np.percentile(z_values, percentile)
    if len(np.unique(z_values)) == 1:
        return datapoints
    return datapoints[datapoints[:, 2] > p_value]


def duplicate_points(datapoints, min_dup=1, max_dup=9):
    """Duplicate points proportional to attention strength."""
    if len(datapoints) == 0:
        return datapoints
    values = datapoints[:, 2].astype(float)
    if values.max() == values.min():
        scaled = np.full(len(values), min_dup + 1, dtype=int)
    else:
        scaled = ((values - values.min()) / (values.max() - values.min()) * (max_dup - min_dup) + min_dup + 1).astype(int)
    weighted_points = np.concatenate([np.repeat([pt], rep, axis=0) for pt, rep in zip(datapoints, scaled)], axis=0)
    return weighted_points


def find_clusters(attentions_with_locations, eps=1.3, min_samples=15):
    """Run DBSCAN on 2D coordinates."""
    x_coords = attentions_with_locations[:, 0].astype(float)
    y_coords = attentions_with_locations[:, 1].astype(float)
    coords = np.stack((x_coords, y_coords), axis=-1)
    
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    return db, n_clusters, n_noise


def calculate_metrics(db, weighted_attentions):
    """Compute per-cluster metrics."""
    labels = db.labels_
    unique_clusters = set(labels) - {-1}
    cluster_strengths = {}
    cluster_counts = Counter(labels)
    
    if len(weighted_attentions) == 0:
        return cluster_strengths
    
    z_values = weighted_attentions[:, 2].astype(float)
    for cluster in unique_clusters:
        cluster_points = z_values[labels == cluster]
        count = cluster_counts.get(cluster, 0)
        avg_strength = np.mean(cluster_points) if len(cluster_points) > 0 else 0.0
        cluster_strengths[cluster] = ClusterData(count, avg_strength)
    
    return cluster_strengths


def cluster_entropy(db, weighted_attentions):
    """Compute normalized entropy for each cluster."""
    labels = db.labels_
    unique_clusters = set(labels) - {-1}
    cluster_entropies = {}
    
    if len(weighted_attentions) == 0:
        return cluster_entropies
    
    z_values = weighted_attentions[:, 2].astype(float)
    for cluster in unique_clusters:
        cluster_points = z_values[labels == cluster]
        total_count = len(cluster_points)
        if total_count > 1:
            counts = Counter(cluster_points)
            if len(counts) <= 1:
                normalized_entropy = 0.0
            else:
                probabilities = [c / total_count for c in counts.values()]
                entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
                max_entropy = np.log2(len(counts))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            normalized_entropy = 0.0
        cluster_entropies[cluster] = normalized_entropy
    
    return cluster_entropies


# ============================================================================
# Multi-Model Mechanism Classes
# ============================================================================

class Qwen2VLMechanism:
    """Top-K attention analysis for Qwen2-VL using same methodology as LLaVA.
    
    Qwen2-VL uses Grouped Query Attention (GQA):
    - num_heads = 28 (query heads)
    - num_kv_heads = 4 (key-value heads)
    - Each KV head is shared by 7 query heads (28/4 = 7)
    - head_dim = 128
    """
    
    def __init__(self, device="cuda"):
        self.config = MODEL_CONFIGS["qwen2-vl"]
        self.device = device
        
        print(f"Loading Qwen2-VL from {self.config['model_id']}...")
        # Use bfloat16 for numerical stability (float16 causes NaN logits)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config["model_id"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
        self.processor = AutoProcessor.from_pretrained(self.config["model_id"])
        self.model.eval()
        
        # Get GQA parameters from first layer
        # Qwen2-VL path: model.model.language_model.layers
        first_layer = self.model.model.language_model.layers[0].self_attn
        self.num_heads = first_layer.num_heads  # 28
        self.num_kv_heads = first_layer.num_key_value_heads  # 4
        self.head_dim = first_layer.head_dim  # 128
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # 7
        
        print(f"Qwen2-VL loaded successfully")
        print(f"  GQA config: {self.num_heads} heads, {self.num_kv_heads} KV heads, {self.num_kv_groups} groups")
    
    def get_layers(self):
        """Get transformer layers - Qwen2-VL uses model.model.language_model.layers"""
        return self.model.model.language_model.layers
    
    def _find_image_token_range(self, inputs):
        """
        Find the range of image token positions in the input sequence.
        For Qwen2-VL, image tokens are embedded after system prompt tokens.
        
        Returns: (start_idx, end_idx) - the range of image token positions
        """
        # Qwen2-VL encodes images as special tokens <|image_pad|>
        # We can detect them by looking at the input_ids or use image_grid_thw
        
        if "image_grid_thw" in inputs:
            # image_grid_thw gives us [temporal, height, width] of the image grid
            grid = inputs["image_grid_thw"][0]  # First image
            num_image_tokens = int(grid[0] * grid[1] * grid[2])
        else:
            # Fallback: estimate from typical 336x336 image (24x24 patches)
            num_image_tokens = 576
        
        # In Qwen2-VL, image tokens come after the system prompt but before the question
        # We'll find them by looking for consecutive token patterns
        # For simplicity, return the estimated range
        # The actual start position depends on the prompt template
        
        input_ids = inputs["input_ids"][0]
        seq_len = len(input_ids)
        
        # Heuristic: image tokens are typically in the middle, after <|im_start|>user\n
        # and before the question text. We'll estimate based on sequence length.
        # A more robust approach would decode tokens and find the image placeholder.
        
        # For now, return the number of image tokens; actual positions will be
        # determined during per-patch computation
        return num_image_tokens
    
    def get_attention_patches(self, image, question, K=5):
        """Compute top-K head aggregated attention map using Δ log P methodology.
        
        This follows the original LLaVA methodology:
        1. Compute per-head Δ log P for head selection (scalar per head)
        2. For top-K heads, compute per-PATCH Δ log P contributions
        3. Aggregate into a spatial attention map
        
        Handles GQA by expanding KV heads.
        """
        # Prepare input
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Cast pixel_values to bfloat16 to match model dtype
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        
        # Get number of image tokens
        num_image_tokens = self._find_image_token_range(inputs)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # Get predicted token
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        predicted_idx = probs.argmax().item()
        token_confidence = float(torch.log(probs[predicted_idx]).item())
        
        # Get hidden states and attentions
        hidden_states = outputs.hidden_states
        attentions = outputs.attentions
        layers = self.get_layers()
        num_layers = min(self.config["num_layers"], len(layers))
        
        # ================================================================
        # STEP 1: Compute per-head Δ log P for head selection
        # ================================================================
        all_head_increase = []
        
        for layer_idx in range(num_layers):
            if layer_idx >= len(hidden_states) - 1 or layer_idx >= len(attentions):
                continue
                
            layer_input = hidden_states[layer_idx][0].clone().detach()  # [seq_len, hidden]
            attn_weights = attentions[layer_idx][0].clone().detach()  # [num_heads, seq_len, seq_len]
            
            layer = layers[layer_idx]
            seq_len = layer_input.shape[0]
            
            # Get V projection
            v_proj = layer.self_attn.v_proj
            V = v_proj(layer_input.to(v_proj.weight.dtype))
            
            # Handle GQA: reshape and expand KV heads
            V_kv = V.view(seq_len, self.num_kv_heads, self.head_dim)
            V_expanded = V_kv.repeat_interleave(self.num_kv_groups, dim=1)
            V_heads = V_expanded.permute(1, 0, 2)  # [num_heads, seq_len, head_dim]
            
            # Compute attention output per head
            attn_out = torch.bmm(attn_weights, V_heads)  # [num_heads, seq_len, head_dim]
            
            # Get o_proj weight split by head
            o_proj = layer.self_attn.o_proj
            o_weight = o_proj.weight.data.to(attn_out.dtype)
            hidden_size = o_weight.shape[0]
            o_weight_split = o_weight.view(hidden_size, self.num_heads, self.head_dim).permute(1, 2, 0)
            
            # Compute head contributions: [num_heads, seq_len, hidden]
            head_outputs = torch.bmm(attn_out, o_weight_split)
            
            # Baseline: layer input at last position
            residual_last = layer_input[-1]
            base_logits = get_bsvalues_qwen(residual_last, self.model)
            base_log_prob = torch.log(get_prob(base_logits)[predicted_idx] + 1e-10)
            
            # Compute Δ log P for each head (using last token position)
            for head_idx in range(self.num_heads):
                head_contrib = head_outputs[head_idx, -1, :]
                added = head_contrib + residual_last
                added_logits = get_bsvalues_qwen(added, self.model)
                added_log_prob = torch.log(get_prob(added_logits)[predicted_idx] + 1e-10)
                delta = (added_log_prob - base_log_prob).item()
                all_head_increase.append((layer_idx, head_idx, delta))
        
        # Select top-K heads
        all_head_increase.sort(key=lambda x: x[2], reverse=True)
        top_k_heads = all_head_increase[:K]
        
        print(f"Top {K} heads (layer, head, delta):")
        for layer_idx, head_idx, delta in top_k_heads:
            print(f"  Layer {layer_idx}, Head {head_idx}: {delta:.4f}")
        
        # ================================================================
        # STEP 2: Compute per-PATCH Δ log P for top-K heads
        # This is the key difference from the broken version
        # ================================================================
        
        if len(top_k_heads) == 0:
            return normalize(np.zeros(576).tolist()), token_confidence, self.processor.decode(predicted_idx)
        
        # Compute softmax weights over head scores
        scores = np.array([delta for _, _, delta in top_k_heads])
        if len(scores) > 1 and not np.allclose(scores, scores[0]):
            weights = np.exp(scores - scores.max())
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(scores)) / len(scores)
        
        # Create weight map
        weight_map = {(l, h): w for (l, h, _), w in zip(top_k_heads, weights)}
        
        # Group heads by layer for efficient computation
        heads_by_layer = {}
        for layer_idx, head_idx, _ in top_k_heads:
            if layer_idx not in heads_by_layer:
                heads_by_layer[layer_idx] = []
            heads_by_layer[layer_idx].append(head_idx)
        
        # Aggregate per-patch contributions
        # We use the sequence length, focusing on image token positions
        seq_len = hidden_states[0].shape[1]
        aggregated_scores = np.zeros(seq_len, dtype=float)
        
        for layer_idx, head_indices in heads_by_layer.items():
            layer_input = hidden_states[layer_idx][0].clone().detach()
            attn_weights = attentions[layer_idx][0].clone().detach()
            
            layer = layers[layer_idx]
            cur_seq_len = layer_input.shape[0]
            
            # Compute V projection
            v_proj = layer.self_attn.v_proj
            V = v_proj(layer_input.to(v_proj.weight.dtype))
            V_kv = V.view(cur_seq_len, self.num_kv_heads, self.head_dim)
            V_expanded = V_kv.repeat_interleave(self.num_kv_groups, dim=1)
            V_heads = V_expanded.permute(1, 0, 2)
            
            # Get o_proj weight
            o_proj = layer.self_attn.o_proj
            o_weight = o_proj.weight.data.to(V_heads.dtype)
            hidden_size = o_weight.shape[0]
            o_weight_split = o_weight.view(hidden_size, self.num_heads, self.head_dim).permute(1, 2, 0)
            
            # Baseline for this layer
            residual_last = layer_input[-1]
            base_logits = get_bsvalues_qwen(residual_last, self.model)
            base_log_prob = torch.log(get_prob(base_logits)[predicted_idx] + 1e-10)
            
            for head_idx in head_indices:
                # Compute this head's per-position contribution
                # attn_weights[head_idx]: [seq_len, seq_len] - attention from each position
                # V_heads[head_idx]: [seq_len, head_dim] - value vectors
                
                # For each source position, compute its contribution to the last token
                # This is: attn[head, -1, pos] * V[pos] @ o_proj_head
                attn_to_last = attn_weights[head_idx, -1, :]  # [seq_len] - attention from last token to all
                
                # Compute contribution of each position
                # head_contribution[pos] = attn[-1, pos] * (V[pos] @ o_proj)
                V_this_head = V_heads[head_idx]  # [seq_len, head_dim]
                o_proj_this_head = o_weight_split[head_idx]  # [head_dim, hidden]
                
                # Per-position contribution to hidden state
                per_pos_hidden = torch.mm(V_this_head, o_proj_this_head)  # [seq_len, hidden]
                
                # Weighted by attention
                # For each position p, compute Δ log P if we only include contribution from p
                per_pos_increase = []
                for pos in range(cur_seq_len):
                    # Contribution from position p, weighted by attention
                    pos_contrib = attn_to_last[pos] * per_pos_hidden[pos]
                    added = pos_contrib + residual_last
                    added_logits = get_bsvalues_qwen(added, self.model)
                    added_log_prob = torch.log(get_prob(added_logits)[predicted_idx] + 1e-10)
                    delta = (added_log_prob - base_log_prob).item()
                    per_pos_increase.append(delta)
                
                # Weight by head weight and accumulate
                w = weight_map.get((layer_idx, head_idx), 0.0)
                per_pos_array = np.array(per_pos_increase, dtype=float)
                if len(per_pos_array) == len(aggregated_scores):
                    aggregated_scores += w * per_pos_array
                elif len(per_pos_array) < len(aggregated_scores):
                    aggregated_scores[:len(per_pos_array)] += w * per_pos_array
                else:
                    aggregated_scores += w * per_pos_array[:len(aggregated_scores)]
        
        # Extract image token contributions
        # For Qwen2-VL, image tokens are typically after system prompt
        # We'll take the first num_image_tokens positions that have significant values
        # Or resize to 576 (24x24) for compatibility
        
        if num_image_tokens > 0 and num_image_tokens < len(aggregated_scores):
            # Try to extract image region - take middle portion where image tokens likely are
            # This is a heuristic; exact positions depend on prompt template
            start_idx = 5  # Skip initial special tokens
            end_idx = min(start_idx + num_image_tokens, len(aggregated_scores))
            image_scores = aggregated_scores[start_idx:end_idx]
        else:
            image_scores = aggregated_scores
        
        # Resize to 576 for compatibility with clustering (24x24 grid)
        if len(image_scores) != 576:
            # Interpolate or pad
            if len(image_scores) > 576:
                # Take first 576
                image_scores = image_scores[:576]
            else:
                # Pad with zeros
                padded = np.zeros(576)
                padded[:len(image_scores)] = image_scores
                image_scores = padded
        
        # Normalize
        aggregated_norm = normalize(image_scores.tolist())
        
        print(f"DEBUG: Aggregated map mean: {np.mean(image_scores):.6f}")
        print(f"DEBUG: Aggregated map max: {np.max(image_scores):.6f}")
        print(f"DEBUG: Aggregated map min: {np.min(image_scores):.6f}")
        
        return aggregated_norm, token_confidence, self.processor.decode(predicted_idx)


class PaliGemmaMechanism:
    """Top-K attention analysis for PaliGemma using same methodology as LLaVA.
    
    PaliGemma-3B uses GQA with:
    - num_heads = 8 (query heads)
    - num_kv_heads = 1 (single KV head shared across all query heads)
    - head_dim = 256
    - hidden_dim = 2048
    
    Image tokens are at the START of the sequence (first 256 tokens for 224x224 images).
    """
    
    def __init__(self, device="cuda"):
        self.config = MODEL_CONFIGS["paligemma"]
        self.device = device
        
        print(f"Loading PaliGemma from {self.config['model_id']}...")
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.config["model_id"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
        self.processor = AutoProcessor.from_pretrained(self.config["model_id"])
        self.model.eval()
        
        # Get attention parameters from config
        config = self.model.language_model.config
        self.num_heads = config.num_attention_heads  # 8
        self.num_kv_heads = config.num_key_value_heads  # 1
        self.head_dim = config.hidden_size // config.num_attention_heads  # 256
        self.num_kv_groups = self.num_heads // self.num_kv_heads if self.num_kv_heads else 1  # 8
        
        # PaliGemma: 224x224 image with 14x14 patches = 16x16 = 256 image tokens
        self.num_image_tokens = 256
        
        print(f"PaliGemma loaded successfully")
        print(f"  GQA config: {self.num_heads} heads, {self.num_kv_heads} KV heads, {self.num_kv_groups} groups")
        print(f"  Image tokens: {self.num_image_tokens} (at sequence start)")
    
    def get_layers(self):
        """Get transformer layers - PaliGemma uses model.language_model.layers"""
        return self.model.language_model.layers
    
    def get_attention_patches(self, image, question, K=5):
        """Compute top-K head aggregated attention map using Δ log P methodology.
        
        This follows the original LLaVA methodology:
        1. Compute per-head Δ log P for head selection
        2. For top-K heads, compute per-PATCH Δ log P contributions
        3. Aggregate into a spatial attention map
        
        PaliGemma: image tokens are at positions [0:256] (first 256 tokens).
        """
        # Prepare input
        prompt = f"<image>{question}"
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device)
        
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # Get predicted token
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        predicted_idx = probs.argmax().item()
        token_confidence = float(torch.log(probs[predicted_idx]).item())
        
        # Get hidden states and attentions
        hidden_states = outputs.hidden_states
        attentions = outputs.attentions
        layers = self.get_layers()
        num_layers = min(self.config["num_layers"], len(layers))
        
        # ================================================================
        # STEP 1: Compute per-head Δ log P for head selection
        # ================================================================
        all_head_increase = []
        
        for layer_idx in range(num_layers):
            if layer_idx >= len(hidden_states) - 1 or layer_idx >= len(attentions):
                continue
                
            layer_input = hidden_states[layer_idx][0].clone().detach()
            attn_weights = attentions[layer_idx][0].clone().detach()
            
            layer = layers[layer_idx]
            seq_len = layer_input.shape[0]
            
            # Get V projection
            v_proj = layer.self_attn.v_proj
            V = v_proj(layer_input.to(v_proj.weight.dtype))
            
            # Handle GQA: reshape and expand KV heads
            V_kv = V.view(seq_len, self.num_kv_heads, self.head_dim)
            V_expanded = V_kv.repeat_interleave(self.num_kv_groups, dim=1)
            V_heads = V_expanded.permute(1, 0, 2)  # [num_heads, seq_len, head_dim]
            
            # Compute attention output per head
            attn_out = torch.bmm(attn_weights, V_heads)
            
            # Get o_proj weight split by head
            o_proj = layer.self_attn.o_proj
            o_weight = o_proj.weight.data.to(attn_out.dtype)
            hidden_size = o_weight.shape[0]
            o_weight_split = o_weight.view(hidden_size, self.num_heads, self.head_dim).permute(1, 2, 0)
            
            # Compute head contributions
            head_outputs = torch.bmm(attn_out, o_weight_split)
            
            # Baseline
            residual_last = layer_input[-1]
            base_logits = get_bsvalues_paligemma(residual_last, self.model)
            base_log_prob = torch.log(get_prob(base_logits)[predicted_idx] + 1e-10)
            
            # Compute Δ log P for each head
            for head_idx in range(self.num_heads):
                head_contrib = head_outputs[head_idx, -1, :]
                added = head_contrib + residual_last
                added_logits = get_bsvalues_paligemma(added, self.model)
                added_log_prob = torch.log(get_prob(added_logits)[predicted_idx] + 1e-10)
                delta = (added_log_prob - base_log_prob).item()
                all_head_increase.append((layer_idx, head_idx, delta))
        
        # Select top-K heads
        all_head_increase.sort(key=lambda x: x[2], reverse=True)
        top_k_heads = all_head_increase[:K]
        
        print(f"Top {K} heads (layer, head, delta):")
        for layer_idx, head_idx, delta in top_k_heads:
            print(f"  Layer {layer_idx}, Head {head_idx}: {delta:.4f}")
        
        # ================================================================
        # STEP 2: Compute per-PATCH Δ log P for top-K heads
        # ================================================================
        
        if len(top_k_heads) == 0:
            return normalize(np.zeros(256).tolist()), token_confidence, self.processor.decode(predicted_idx)
        
        # Compute softmax weights over head scores
        scores = np.array([delta for _, _, delta in top_k_heads])
        if len(scores) > 1 and not np.allclose(scores, scores[0]):
            weights = np.exp(scores - scores.max())
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(scores)) / len(scores)
        
        weight_map = {(l, h): w for (l, h, _), w in zip(top_k_heads, weights)}
        
        # Group heads by layer
        heads_by_layer = {}
        for layer_idx, head_idx, _ in top_k_heads:
            if layer_idx not in heads_by_layer:
                heads_by_layer[layer_idx] = []
            heads_by_layer[layer_idx].append(head_idx)
        
        # Aggregate per-patch contributions
        seq_len = hidden_states[0].shape[1]
        aggregated_scores = np.zeros(seq_len, dtype=float)
        
        for layer_idx, head_indices in heads_by_layer.items():
            layer_input = hidden_states[layer_idx][0].clone().detach()
            attn_weights = attentions[layer_idx][0].clone().detach()
            
            layer = layers[layer_idx]
            cur_seq_len = layer_input.shape[0]
            
            # Compute V projection
            v_proj = layer.self_attn.v_proj
            V = v_proj(layer_input.to(v_proj.weight.dtype))
            V_kv = V.view(cur_seq_len, self.num_kv_heads, self.head_dim)
            V_expanded = V_kv.repeat_interleave(self.num_kv_groups, dim=1)
            V_heads = V_expanded.permute(1, 0, 2)
            
            # Get o_proj weight
            o_proj = layer.self_attn.o_proj
            o_weight = o_proj.weight.data.to(V_heads.dtype)
            hidden_size = o_weight.shape[0]
            o_weight_split = o_weight.view(hidden_size, self.num_heads, self.head_dim).permute(1, 2, 0)
            
            # Baseline for this layer
            residual_last = layer_input[-1]
            base_logits = get_bsvalues_paligemma(residual_last, self.model)
            base_log_prob = torch.log(get_prob(base_logits)[predicted_idx] + 1e-10)
            
            for head_idx in head_indices:
                # Get attention from last token to all positions
                attn_to_last = attn_weights[head_idx, -1, :]  # [seq_len]
                
                # Per-position contribution
                V_this_head = V_heads[head_idx]  # [seq_len, head_dim]
                o_proj_this_head = o_weight_split[head_idx]  # [head_dim, hidden]
                per_pos_hidden = torch.mm(V_this_head, o_proj_this_head)  # [seq_len, hidden]
                
                # Compute per-position Δ log P
                per_pos_increase = []
                for pos in range(cur_seq_len):
                    pos_contrib = attn_to_last[pos] * per_pos_hidden[pos]
                    added = pos_contrib + residual_last
                    added_logits = get_bsvalues_paligemma(added, self.model)
                    added_log_prob = torch.log(get_prob(added_logits)[predicted_idx] + 1e-10)
                    delta = (added_log_prob - base_log_prob).item()
                    per_pos_increase.append(delta)
                
                # Weight and accumulate
                w = weight_map.get((layer_idx, head_idx), 0.0)
                per_pos_array = np.array(per_pos_increase, dtype=float)
                if len(per_pos_array) == len(aggregated_scores):
                    aggregated_scores += w * per_pos_array
                elif len(per_pos_array) < len(aggregated_scores):
                    aggregated_scores[:len(per_pos_array)] += w * per_pos_array
                else:
                    aggregated_scores += w * per_pos_array[:len(aggregated_scores)]
        
        # Extract image token contributions
        # PaliGemma: image tokens are at positions [0:256]
        image_scores = aggregated_scores[:self.num_image_tokens]
        
        # Normalize
        aggregated_norm = normalize(image_scores.tolist())
        
        print(f"DEBUG: Aggregated map mean: {np.mean(image_scores):.6f}")
        print(f"DEBUG: Aggregated map max: {np.max(image_scores):.6f}")
        print(f"DEBUG: Aggregated map min: {np.min(image_scores):.6f}")
        
        return aggregated_norm, token_confidence, self.processor.decode(predicted_idx)


# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def load_vqa_data(n_samples=100):
    """Load VQA data from the standard dataset location."""
    # Try multiple possible paths for CSV
    possible_paths = [
        "results.csv",
        "../results.csv",
        "/home/ubuntu/LLaVAProbe/results.csv",
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    
    if df is None:
        raise FileNotFoundError("Could not find VQA dataset (results.csv)")
    
    # Convert to list of dicts
    data = []
    for _, row in df.iterrows():
        data.append({
            "question_type": row.get("question_type", "unknown"),
            "question": row.get("question", ""),
            "ground_truth": row.get("ground_truth", ""),
            "image_url": row.get("image_url", ""),
        })
    
    # Sample data
    if n_samples and n_samples < len(data):
        import random
        random.seed(42)
        data = random.sample(data, n_samples)
    
    return data


def run_analysis(model_name, top_k=5, n_samples=100):
    """Run top-K attention analysis for specified model."""
    
    print(f"\n{'='*80}")
    print(f"Running Top-{top_k} Analysis for {model_name.upper()}")
    print(f"{'='*80}\n")
    
    # Initialize mechanism
    if model_name == "qwen2-vl":
        mechanism = Qwen2VLMechanism()
        patch_size = MODEL_CONFIGS["qwen2-vl"]["patch_grid"]
    elif model_name == "paligemma":
        mechanism = PaliGemmaMechanism()
        patch_size = MODEL_CONFIGS["paligemma"]["patch_grid"]
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load data
    print("Loading VQA data...")
    data = load_vqa_data(n_samples)
    print(f"Loaded {len(data)} samples")
    
    results = []
    
    for idx, item in enumerate(tqdm(data, desc="Processing")):
        try:
            # Load image
            image_url = item.get("image_url", item.get("image", ""))
            if image_url.startswith("http"):
                response = requests.get(image_url, stream=True, verify=False, timeout=10)
                image = Image.open(response.raw).convert("RGB")
            else:
                image = Image.open(image_url).convert("RGB")
            
            question = item.get("question", "")
            ground_truth = item.get("answer", item.get("ground_truth", ""))
            question_type = item.get("question_type", "unknown")
            
            # Get attention patches
            aggregated_norm, token_confidence, predicted_token = mechanism.get_attention_patches(
                image, question, K=top_k
            )
            
            # Compute metrics
            attention_entropy = calculate_entropy(aggregated_norm)
            
            # Skip spatial clustering for models with dynamic resolution
            # Just compute basic metrics
            n_clusters = 0
            n_noise = 0
            cluster_metrics = {}
            cluster_ents = {}
            
            # Try clustering only if we have a fixed grid (not for Qwen2-VL)
            if model_name != "qwen2-vl" and len(aggregated_norm) == patch_size * patch_size:
                try:
                    scores_2d = np.array(aggregated_norm).reshape(patch_size, patch_size)
                    attentions_3d = transform_matrix_to_3d_points(scores_2d)
                    
                    filtered = apply_threshold(attentions_3d, 80)
                    weighted = duplicate_points(filtered, 1, 9)
                    
                    if len(weighted) > 0:
                        db, n_clusters, n_noise = find_clusters(weighted)
                        cluster_metrics = calculate_metrics(db, weighted)
                        cluster_ents = cluster_entropy(db, weighted)
                except Exception:
                    pass  # Fall back to no clustering
            
            # Build result dict
            result = {
                "image_id": idx,
                "question_type": question_type,
                "question": question,
                "ground_truth": ground_truth,
                "image_url": image_url,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "attention_entropy": attention_entropy,
                "token_confidence": token_confidence,
                "predicted_token": predicted_token,
                "cluster_count": len(cluster_metrics),
            }
            
            for cid, metrics in cluster_metrics.items():
                result[f"cluster_{cid}_n_points"] = metrics.n_points
                result[f"cluster_{cid}_avg_strength"] = metrics.ave_strength
            
            for cid, ent in cluster_ents.items():
                result[f"cluster_{cid}_entropy"] = ent
            
            results.append(result)
            
            if (idx + 1) % 50 == 0:
                print(f"\nProcessed {idx + 1}/{len(data)} samples")
                print(f"  Last entropy: {attention_entropy:.4f}, confidence: {token_confidence:.4f}")
        
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # Save results
    output_path = f"{model_name}_topk_{top_k}_results.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Compute summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total samples: {len(results)}")
    print(f"Mean attention entropy: {df['attention_entropy'].mean():.4f}")
    print(f"Mean token confidence: {df['token_confidence'].mean():.4f}")
    
    # Correlation
    from scipy.stats import pearsonr
    r, p = pearsonr(df['attention_entropy'], df['token_confidence'])
    print(f"Pearson R (entropy vs confidence): {r:.4f} (p={p:.4f})")
    print(f"R²: {r**2:.4f}")
    
    # By question type
    print("\nBy Question Type:")
    for qt in df['question_type'].unique():
        subset = df[df['question_type'] == qt]
        if len(subset) >= 3:
            r, _ = pearsonr(subset['attention_entropy'], subset['token_confidence'])
            print(f"  {qt}: N={len(subset)}, R={r:.4f}, R²={r**2:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-model Top-K attention analysis")
    parser.add_argument("--model", type=str, required=True, choices=["qwen2-vl", "paligemma"],
                        help="Model to analyze")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top heads (K)")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to process")
    
    args = parser.parse_args()
    
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    run_analysis(args.model, args.top_k, args.n_samples)


if __name__ == "__main__":
    main()
