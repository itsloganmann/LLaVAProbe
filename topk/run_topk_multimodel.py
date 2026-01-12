#!/usr/bin/env python3
"""
Multi-Model Top-K Attention Entropy Analysis

Supports: Qwen2-VL-7B, PaliGemma-3B
Computes per-head Δ log P contributions, selects top-K heads,
and measures attention entropy correlation with token confidence.

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
# Configuration
# ============================================================================

MODEL_CONFIGS = {
    "qwen2-vl": {
        "model_id": "Qwen/Qwen2-VL-7B-Instruct",
        "num_layers": 28,
        "num_heads": 28,
        "head_dim": 128,
        "hidden_dim": 3584,
        "layer_path": "model.layers",  # Qwen2-VL: model.model.layers
        "patch_grid": 24,  # May vary based on image size
    },
    "paligemma": {
        "model_id": "google/paligemma-3b-mix-224",
        "num_layers": 18,
        "num_heads": 8,
        "head_dim": 256,
        "hidden_dim": 2048,
        "layer_path": "language_model.model.layers",
        "patch_grid": 16,  # 224/14 = 16
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
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config["model_id"],
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
        )
        self.processor = AutoProcessor.from_pretrained(self.config["model_id"])
        self.model.eval()
        
        # Get GQA parameters from first layer
        first_layer = self.model.model.layers[0].self_attn
        self.num_heads = first_layer.num_heads  # 28
        self.num_kv_heads = first_layer.num_key_value_heads  # 4
        self.head_dim = first_layer.head_dim  # 128
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # 7
        
        print(f"Qwen2-VL loaded successfully")
        print(f"  GQA config: {self.num_heads} heads, {self.num_kv_heads} KV heads, {self.num_kv_groups} groups")
    
    def get_layers(self):
        """Get transformer layers - Qwen2-VL uses model.model.layers"""
        return self.model.model.layers
    
    def get_attention_patches(self, image, question, K=5):
        """Compute top-K head aggregated attention map using Δ log P methodology.
        
        Same approach as LLaVA but handles GQA by expanding KV heads.
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
        
        # Run inference
        with torch.inference_mode():
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
        
        # Compute per-head contributions using Δ log P
        all_head_increase = []
        layers = self.get_layers()
        num_layers = min(self.config["num_layers"], len(layers))
        hidden_states = outputs.hidden_states
        attentions = outputs.attentions
        
        for layer_idx in range(num_layers):
            if layer_idx >= len(hidden_states) - 1 or layer_idx >= len(attentions):
                continue
                
            # Clone and detach to avoid inference mode issues
            layer_input = hidden_states[layer_idx][0].clone().detach()  # [seq_len, hidden]
            attn_weights = attentions[layer_idx][0].clone().detach()  # [num_heads, seq_len, seq_len]
            
            layer = layers[layer_idx]
            num_heads = attn_weights.shape[0]
            
            # Get V projection - outputs [seq_len, num_kv_heads * head_dim]
            v_proj = layer.self_attn.v_proj
            with torch.no_grad():
                V = v_proj(layer_input.to(v_proj.weight.dtype))
            
            # Reshape V for KV heads: [seq_len, num_kv_heads, head_dim]
            seq_len = V.shape[0]
            V_kv = V.view(seq_len, self.num_kv_heads, self.head_dim)
            
            # Expand KV heads to match query heads (GQA expansion)
            # Each KV head is repeated num_kv_groups times
            # [seq_len, num_kv_heads, head_dim] -> [seq_len, num_heads, head_dim]
            V_expanded = V_kv.repeat_interleave(self.num_kv_groups, dim=1)
            
            # Permute to [num_heads, seq_len, head_dim]
            V_heads = V_expanded.permute(1, 0, 2)
            
            # Compute per-head output: attn_weights @ V_heads
            # attn_weights: [num_heads, seq_len, seq_len]
            # V_heads: [num_heads, seq_len, head_dim]
            attn_out = torch.bmm(attn_weights, V_heads)  # [num_heads, seq_len, head_dim]
            
            # Get o_proj weight and split by head
            o_proj = layer.self_attn.o_proj
            o_weight = o_proj.weight.data.to(attn_out.dtype)
            hidden_size = o_weight.shape[0]
            
            # o_proj expects [batch, seq, num_heads * head_dim] -> [batch, seq, hidden]
            # Split weight: [hidden_size, num_heads * head_dim] -> [num_heads, head_dim, hidden_size]
            o_weight_split = o_weight.view(hidden_size, num_heads, self.head_dim).permute(1, 2, 0)
            
            # Compute head contributions: [num_heads, seq_len, head_dim] @ [num_heads, head_dim, hidden] 
            head_outputs = torch.bmm(attn_out, o_weight_split)  # [num_heads, seq_len, hidden_size]
            
            # Baseline: residual only (last position)
            residual_last = layer_input[-1].to(self.model.dtype)
            
            for head_idx in range(num_heads):
                head_contrib = head_outputs[head_idx, -1, :]
                added = (head_contrib + residual_last).float()
                
                with torch.no_grad():
                    # Use final layer norm + lm_head
                    normed = self.model.model.norm(added.unsqueeze(0).half())
                    head_logits = self.model.lm_head(normed)[0]
                    head_probs = torch.softmax(head_logits, dim=-1)
                    head_log_prob = torch.log(head_probs[predicted_idx] + 1e-10)
                    
                    base_normed = self.model.model.norm(residual_last.unsqueeze(0))
                    base_logits = self.model.lm_head(base_normed)[0]
                    base_probs = torch.softmax(base_logits, dim=-1)
                    base_log_prob = torch.log(base_probs[predicted_idx] + 1e-10)
                    
                    delta = (head_log_prob - base_log_prob).item()
                
                all_head_increase.append((f"{layer_idx}_{head_idx}", delta))
        
        # Select top-K heads by Δ log P
        all_head_increase.sort(key=lambda x: x[1], reverse=True)
        top_k_heads = all_head_increase[:K]
        
        # Aggregate attention from top-K heads
        if len(top_k_heads) > 0 and len(attentions) > 0:
            seq_len = attentions[0].shape[-1]
            aggregated = np.zeros(seq_len, dtype=float)
            
            scores = np.array([s for _, s in top_k_heads])
            if len(scores) > 1 and not np.allclose(scores, scores[0]):
                weights = np.exp(scores - scores.max())
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(scores)) / len(scores)
            
            for idx, (head_info, _) in enumerate(top_k_heads):
                layer_idx, head_idx = map(int, head_info.split("_"))
                if layer_idx < len(attentions):
                    attn = attentions[layer_idx][0, head_idx, -1, :].cpu().numpy()
                    if len(attn) == seq_len:
                        aggregated += weights[idx] * attn
                    elif len(attn) > seq_len:
                        aggregated += weights[idx] * attn[:seq_len]
                    else:
                        aggregated[:len(attn)] += weights[idx] * attn
        else:
            aggregated = np.zeros(100, dtype=float)
        
        aggregated_norm = normalize(aggregated.tolist())
        
        return aggregated_norm, token_confidence, self.processor.decode(predicted_idx)


class PaliGemmaMechanism:
    """Top-K attention analysis for PaliGemma using same methodology as LLaVA.
    
    PaliGemma-3B uses standard Multi-Head Attention:
    - num_heads = 8
    - head_dim = 256
    - hidden_dim = 2048
    """
    
    def __init__(self, device="cuda"):
        self.config = MODEL_CONFIGS["paligemma"]
        self.device = device
        
        print(f"Loading PaliGemma from {self.config['model_id']}...")
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.config["model_id"],
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
        )
        self.processor = AutoProcessor.from_pretrained(self.config["model_id"])
        self.model.eval()
        
        # Get attention parameters from first layer
        first_layer = self.model.language_model.model.layers[0].self_attn
        self.num_heads = first_layer.num_heads
        self.head_dim = first_layer.head_dim
        # Check for GQA
        self.num_kv_heads = getattr(first_layer, 'num_key_value_heads', self.num_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads if self.num_kv_heads else 1
        self.uses_gqa = self.num_kv_heads != self.num_heads
        
        print(f"PaliGemma loaded successfully")
        print(f"  MHA config: {self.num_heads} heads, head_dim={self.head_dim}, GQA={self.uses_gqa}")
    
    def get_layers(self):
        """Get transformer layers - PaliGemma uses model.language_model.model.layers"""
        return self.model.language_model.model.layers
    
    def get_attention_patches(self, image, question, K=5):
        """Compute top-K head aggregated attention map."""
        # Prepare input
        prompt = f"<image>{question}"
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device)
        
        # Run inference
        with torch.inference_mode():
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
        
        # Compute per-head contributions using Δ log P
        all_head_increase = []
        layers = self.get_layers()
        num_layers = min(self.config["num_layers"], len(layers))
        hidden_states = outputs.hidden_states
        attentions = outputs.attentions
        
        for layer_idx in range(num_layers):
            if layer_idx >= len(hidden_states) - 1 or layer_idx >= len(attentions):
                continue
                
            # Clone and detach to avoid inference mode issues
            layer_input = hidden_states[layer_idx][0].clone().detach()
            attn_weights = attentions[layer_idx][0].clone().detach()
            
            layer = layers[layer_idx]
            num_heads = attn_weights.shape[0]
            
            v_proj = layer.self_attn.v_proj
            with torch.no_grad():
                V = v_proj(layer_input.to(v_proj.weight.dtype))
            
            seq_len = V.shape[0]
            
            # Handle GQA if present
            if self.uses_gqa:
                # Reshape V for KV heads
                V_kv = V.view(seq_len, self.num_kv_heads, self.head_dim)
                # Expand KV heads to match query heads
                V_expanded = V_kv.repeat_interleave(self.num_kv_groups, dim=1)
                V_heads = V_expanded.permute(1, 0, 2)
            else:
                # Standard MHA
                V_heads = V.view(seq_len, num_heads, self.head_dim).permute(1, 0, 2)
            
            attn_out = torch.bmm(attn_weights, V_heads)
            
            o_proj = layer.self_attn.o_proj
            o_weight = o_proj.weight.data.to(attn_out.dtype)
            hidden_size = o_weight.shape[0]
            
            o_weight_split = o_weight.view(hidden_size, num_heads, self.head_dim).permute(1, 2, 0)
            head_outputs = torch.bmm(attn_out, o_weight_split)
            
            residual_last = layer_input[-1].to(self.model.language_model.model.embed_tokens.weight.dtype)
            
            for head_idx in range(num_heads):
                head_contrib = head_outputs[head_idx, -1, :]
                added = (head_contrib + residual_last).float()
                
                with torch.no_grad():
                    # PaliGemma: model.language_model.model.norm and model.lm_head
                    normed = self.model.language_model.model.norm(added.unsqueeze(0).half())
                    head_logits = self.model.lm_head(normed)[0]
                    head_probs = torch.softmax(head_logits, dim=-1)
                    head_log_prob = torch.log(head_probs[predicted_idx] + 1e-10)
                    
                    base_normed = self.model.language_model.model.norm(residual_last.unsqueeze(0))
                    base_logits = self.model.lm_head(base_normed)[0]
                    base_probs = torch.softmax(base_logits, dim=-1)
                    base_log_prob = torch.log(base_probs[predicted_idx] + 1e-10)
                    
                    delta = (head_log_prob - base_log_prob).item()
                
                all_head_increase.append((f"{layer_idx}_{head_idx}", delta))
        
        # Select top-K heads by Δ log P
        all_head_increase.sort(key=lambda x: x[1], reverse=True)
        top_k_heads = all_head_increase[:K]
        
        # Aggregate attention - use dynamic size
        if len(top_k_heads) > 0 and len(attentions) > 0:
            seq_len = attentions[0].shape[-1]
            aggregated = np.zeros(seq_len, dtype=float)
            
            scores = np.array([s for _, s in top_k_heads])
            if len(scores) > 1 and not np.allclose(scores, scores[0]):
                weights = np.exp(scores - scores.max())
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(scores)) / len(scores)
            
            for idx, (head_info, _) in enumerate(top_k_heads):
                layer_idx, head_idx = map(int, head_info.split("_"))
                if layer_idx < len(attentions):
                    attn = attentions[layer_idx][0, head_idx, -1, :].cpu().numpy()
                    if len(attn) == seq_len:
                        aggregated += weights[idx] * attn
                    elif len(attn) > seq_len:
                        aggregated += weights[idx] * attn[:seq_len]
                    else:
                        aggregated[:len(attn)] += weights[idx] * attn
        else:
            aggregated = np.zeros(100, dtype=float)
        
        aggregated_norm = normalize(aggregated.tolist())
        
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
