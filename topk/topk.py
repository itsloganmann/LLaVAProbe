#!/usr/bin/env python3
# topk.py (annotated for readers of the paper)
#
# Purpose:
#   This script implements the procedure used in our paper to measure
#   individual attention heads' contributions to token-level confidence
#   in the LLaVA model, select the top-k heads by influence (∆ log P),
#   and produce a weighted aggregation of per-patch contributions.
#
#   The code follows Section 3.3 (Attention Head Analysis), Section 3.4
#   (Clustering Analysis), and the entropy/metrics pipeline described in
#   the manuscript. Comments intentionally explain the *why* and *what*
#   (how it maps to the paper) rather than restating obvious Python.
#
# Usage (example):
#   python topk.py --top_k 5
#
# Note:
#   The core numerical operations are left unchanged — only comments
#   were replaced and a single CLI argument (--top_k) was added to set K.

import torch
import copy, math, pickle, json, os
import bertviz, uuid
import numpy as np
import matplotlib.pyplot as plt
import requests
import cv2
import argparse
import os
import random
import urllib.parse
import pandas as pd

from PIL import Image
from transformers.image_transforms import (
    convert_to_rgb,
    get_resize_output_image_size,
    resize,
    center_crop)
from transformers.image_utils import (
    infer_channel_dimension_format,
    to_numpy_array)
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import time
import torch.nn as nn
from matplotlib.ticker import MultipleLocator
from sklearn import metrics
from sklearn.cluster import DBSCAN
from collections import Counter
from dataclasses import dataclass
import urllib3

# Constants describing the LLaVA architecture we assume in the code.
# These should match the model architecture used in experiments.
LAYER_NUM = 32
HEAD_NUM = 32
HEAD_DIM = 128
HIDDEN_DIM = HEAD_NUM * HEAD_DIM

# Batch size for inference. The environment used in the paper ran with batch_size = 1.
batch_size = 1  # increase at your own memory risk

# Small dataclasses used as simple structured containers for clarity.
@dataclass
class ImagePrompt:
    # Holds one CSV row's information: image URL, the VQA question, and a prompt prefix.
    image_url: str
    prompt: str
    prefix: str

@dataclass
class Entropy:
    # Lightweight container for cluster entropy-related outputs (unused as a return type).
    cluster: int
    count_points: int
    average_strength: float


# -----------------------
# Utility functions
# -----------------------

def normalize(vector):
    """
    Convert a list or numpy array of raw patch scores into a normalized,
    probability-like vector that sums to 1. This mirrors the paper's
    description of converting per-patch contributions into a distribution
    before computing entropy or clustering. Handles edge cases:
      - accepts numpy arrays or lists
      - if all values identical and sum==0 returns the original vector
      - if all identical but non-zero returns uniform distribution
    """
    # Accept numpy arrays or lists
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()
    # Handle degenerate case: all values identical
    max_value = max(vector)
    min_value = min(vector)

    if max_value == min_value:
        # If everything is zero, return unchanged (keeps downstream behavior).
        if sum(vector) == 0:
            return vector
        # If identical non-zero values, return uniform distribution.
        else:
            return [1.0 / len(vector)] * len(vector)

    # Min-max then normalize to sum to 1
    vector1 = [(x - min_value) / (max_value - min_value) for x in vector]
    sum_vector1 = sum(vector1)
    if sum_vector1 == 0:
        return [0.0] * len(vector1)

    vector2 = [x / sum_vector1 for x in vector1]
    return vector2


def transfer_output(outputs, batch_size):
    """
    Extract hidden states and attention tensors from the model outputs, and
    reformat them into per-layer lists of per-batch tensors. This function
    centralizes the bookkeeping so the rest of the code can iterate over
    layers/heads more simply.

    Returns three lists:
      - all_pos_layer_input: per-layer list of [batch tensors]
      - all_pos_layer_output: per-layer list of [batch tensors]
      - all_last_attn_subvalues: per-layer list of [batch attention tensors]
    """
    hs = outputs.hidden_states    # often length = num_layers + 1
    att = getattr(outputs, "attentions", None)

    all_pos_layer_input = []
    all_pos_layer_output = []
    all_last_attn_subvalues = []

    # Only iterate up to LAYER_NUM or available hidden states, whichever smaller.
    for layer_i in range(min(LAYER_NUM, len(hs))):
        layer_tensor = hs[layer_i]  # (batch, seq_len, hidden)
        next_layer_tensor = hs[layer_i + 1] if (layer_i + 1) < len(hs) else layer_tensor
        att_tensor = att[layer_i] if (att is not None and layer_i < len(att)) else None

        # Detach & move to CPU for persistent storage; functions expect CPU/cached tensors.
        batch_inputs = [layer_tensor[b].detach().cpu() for b in range(batch_size)]
        batch_outputs = [next_layer_tensor[b].detach().cpu() for b in range(batch_size)]
        if att_tensor is not None:
            batch_subvalues = [att_tensor[b].detach().cpu() for b in range(batch_size)]
        else:
            # Placeholder attention if attentions are not provided.
            batch_subvalues = [torch.zeros(HEAD_NUM, layer_tensor.shape[1], layer_tensor.shape[1]) for _ in range(batch_size)]

        all_pos_layer_input.append(batch_inputs)
        all_pos_layer_output.append(batch_outputs)
        all_last_attn_subvalues.append(batch_subvalues)

    return all_pos_layer_input, all_pos_layer_output, all_last_attn_subvalues


def get_bsvalues(vector, model, final_var):
    """
    Convert a representation vector into model logit space following the
    same RMSNorm + lm_head wiring used in the model. This mirrors the
    paper's 'z_j' computation where we project (hj + x) through the
    final readout to compute token probabilities.
    """
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    vector = vector.to(device)
    final_var = final_var.to(device)

    # RMS-normalization style scaling used by the checkpoint
    vector = vector * torch.rsqrt(final_var + 1e-6)
    vector_rmsn = vector * model.language_model.norm.weight.data.to(device)
    vector_bsvalues = model.lm_head(vector_rmsn).data
    return vector_bsvalues


def get_prob(vector):
    """Return softmax probabilities over the vocabulary for a logit vector."""
    prob = torch.nn.Softmax(-1)(vector)
    return prob


def transfer_l(l):
    """Helper to split a list-of-pairs into two lists (used by some plotting utilities)."""
    new_x, new_y = [], []
    for x in l:
        new_x.append(x[0])
        new_y.append(x[1])
    return new_x, new_y


def plt_bar(x, y, yname="log increase"):
    """
    Small plotting helper used for debugging/visualization: plots per-layer
    bar chart for attention vs. FFN (kept unchanged).
    """
    x_major_locator = MultipleLocator(1)
    plt.figure(figsize=(8, 3))
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt_x = [a / 2 for a in x]
    plt.xlim(-0.5, plt_x[-1] + 0.49)
    x_attn, y_attn, x_ffn, y_ffn = [], [], [], []
    for i in range(len(x)):
        if i % 2 == 0:
            x_attn.append(x[i] / 2)
            y_attn.append(y[i])
        else:
            x_ffn.append(x[i] / 2)
            y_ffn.append(y[i])
    plt.bar(x_attn, y_attn, color="darksalmon", label="attention layers")
    plt.bar(x_ffn, y_ffn, color="lightseagreen", label="FFN layers")
    plt.xlabel("layer")
    plt.ylabel(yname)
    plt.legend()
    plt.show()


def plt_heatmap(data):
    """
    Debug helper: render a heatmap of head log-increase values. Useful to
    visually inspect which heads/layers are influential before clustering.
    """
    xLabel = range(len(data[0]))
    yLabel = range(len(data))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.set_xticks(range(len(xLabel)))
    ax.set_xticklabels(xLabel)
    ax.set_yticks(range(len(yLabel)))
    ax.set_yticklabels(yLabel)
    im = ax.imshow(data, cmap=plt.cm.hot_r)
    plt.title("attn head log increase heatmap")
    plt.show()


# -----------------------
# Core mechanism wrapper
# -----------------------
class LlavaMechanism:
    """
    Thin wrapper around LLaVA model + processor that provides one
    convenience method: get_attention_patches(..., K) which
    - runs inference,
    - computes per-head ∆ log P contributions for the predicted token,
    - selects top-K heads by ∆ log P,
    - aggregates per-patch contributions from those heads into a 576-d vector,
    - normalizes and returns the result plus model probabilities.

    This mirrors the pipeline in Section 3.3 and Section 3.4 of the paper.
    """
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf", device="cuda"):
        # Force default device to CUDA if available (matches experimental setup).
        torch.set_default_device('cuda')

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
        )
        # Load model; low_cpu_mem_usage reduces host memory during load.
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
        ).to(device)
        # Eager attention implementation is used in our experiments.
        self.model.set_attn_implementation("eager")
        print(f"Model loaded on {self.model.device}")
        self.model.eval()

        # Processor handles text+image tokenization and image pre-processing.
        self.processor = AutoProcessor.from_pretrained(model_id, revision='a272c74')
        # patch_size used to map model patch grid to our 24x24 representation.
        self.processor.patch_size = 14
        print(f"Model loaded on {self.model.device}")
        self.model.eval()

        self.output_dir = "output_images"
        os.makedirs(self.output_dir, exist_ok=True)

    def get_attention_patches(self, images, prompts, prefixes, K=5):
        """
        Run the model on a batch of images + prompts and return a list of tuples:
          (center-cropped image, normalized 576-d vector, sorted token indices, probs tensor)

        K: number of top heads to aggregate (see Section 3.3 top-k head selection).
        """
        batch_results = []
        batch_size = len(images)
        t = time.time()

        # Build prompts the same way used in the experiments (prefix engineering).
        full_prompts = [f"USER: <image>\n{p}\nASSISTANT: {pref}" for p, pref in zip(prompts, prefixes)]
        inputs = self.processor(text=full_prompts, images=images, return_tensors="pt", padding=True).to(self.model.device)

        # Run inference with attentions and hidden states to measure internal contributions.
        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True
            )

        print(f'Finished inference time {time.time() - t}')

        # Reformat outputs for easier per-layer/per-batch indexing.
        transfer_outputs = transfer_output(outputs, batch_size)

        # Process each item in the batch independently.
        for i in range(batch_size):
            all_pos_layer_input_i = [layer[i] for layer in transfer_outputs[0]]
            all_pos_layer_output_i = [layer[i] for layer in transfer_outputs[1]]
            all_last_attn_subvalues_i = [layer[i] for layer in transfer_outputs[2]]

            # The model's logits may have shape [seq_len, 1, vocab] or [seq_len, vocab].
            logits_i = outputs["logits"][i]
            if logits_i.ndim == 3:
                logits_i = logits_i[:, 0, :]
            outputs_probs = get_prob(logits_i[-1])
            outputs_probs_sort = torch.argsort(outputs_probs, descending=True)
            print([self.processor.decode(x) for x in outputs_probs_sort[:10]])

            # Compute a scaling term final_var derived from the last-layer representation
            # (used consistently when projecting candidate vectors to logit space).
            last_layer_batch = all_pos_layer_output_i[-1]
            last_elem = last_layer_batch if isinstance(last_layer_batch, torch.Tensor) else torch.as_tensor(last_layer_batch)

            if last_elem.dim() == 0:
                vec = last_elem.unsqueeze(0)
            elif last_elem.dim() == 1:
                vec = last_elem
            else:
                vec = last_elem.view(-1, last_elem.shape[-1])[-1]
            vec = vec.to(self.model.device)
            final_var = vec.pow(2).mean(-1, keepdim=True)

            # Prepare a 336x336 center crop for visualization and patch mapping.
            image_i = images[i]
            image_convert = convert_to_rgb(image_i)
            image_numpy = to_numpy_array(image_convert)
            input_data_format = infer_channel_dimension_format(image_numpy)
            output_size = get_resize_output_image_size(image_numpy, size=336, default_to_square=False)
            image_resize = resize(image_numpy, output_size, resample=3, input_data_format=input_data_format)
            image_center_crop = center_crop(image_resize, size=(336, 336), input_data_format=input_data_format)
            demo_img = image_center_crop

            # Predicted token index for the top predicted token; used as the 'y' in ∆ log P.
            predict_index = outputs_probs_sort[0].item()
            print(predict_index, self.processor.decode(predict_index))

            # ----------------------------
            # Per-head ∆ log P computation
            # (implements equation (4) in the paper)
            # ----------------------------
            # We compute for every head j in every layer:
            #   ∆ log P(y | z_with_head) - log P(y | residual_only)
            # where z_with_head = (head_contribution + residual) for the last token.
            # This produces an influence score per (layer, head).
            all_head_increase = []
            for test_layer in range(min(LAYER_NUM, len(all_pos_layer_input_i))):
                layer_input = all_pos_layer_input_i[test_layer]
                layer_output = all_pos_layer_output_i[test_layer]
                att_layer = all_last_attn_subvalues_i[test_layer]

                # Move tensors to the same dtype/device as the model for numeric ops.
                layer_input = layer_input.to(self.model.dtype).to(self.model.device)
                layer_output = layer_output.to(self.model.dtype).to(self.model.device)
                att_layer = att_layer.to(self.model.dtype).to(self.model.device)

                seq_len = layer_input.shape[0]

                # 1) Compute V = v_proj(layer_input) — the value vectors from this layer
                v_proj_module = self.model.language_model.layers[test_layer].self_attn.v_proj
                V = v_proj_module(layer_input)

                # 2) Reshape V into per-head chunks (HEAD_NUM, seq_len, HEAD_DIM)
                V_heads = V.view(seq_len, HEAD_NUM, HEAD_DIM).permute(1, 0, 2)

                # 3) Multiply attention weights by V to yield head outputs per patch
                attn_out_per_head_raw = torch.bmm(att_layer, V_heads)

                # 4) Apply the head-specific output projection (o_proj) split by head
                o_proj_module = self.model.language_model.layers[test_layer].self_attn.o_proj
                o_proj_weight = o_proj_module.weight.data.to(self.model.dtype).to(self.model.device)
                hidden_size = o_proj_weight.shape[0]

                # Reshape o_proj weight to isolate each head's contribution mapping
                o_proj_weight_split = o_proj_weight.view(hidden_size, HEAD_NUM, HEAD_DIM)
                o_proj_weight_split = o_proj_weight_split.permute(1, 2, 0)

                # 5) Compute head-wise contribution vectors (per-head, per-patch)
                attn_subvalues_per_head = torch.bmm(attn_out_per_head_raw, o_proj_weight_split)

                # 6) Measure base log-probability for the residual-only vector at the last token
                cur_layer_input_last = layer_input[-1]
                origin_prob = torch.log(get_prob(get_bsvalues(cur_layer_input_last, self.model, final_var))[predict_index])

                # 7) For each head, compute ∆ log P when adding only that head's output
                #    This yields one scalar influence score per (layer, head).
                for head_i in range(HEAD_NUM):
                    head_contribution_last_token = attn_subvalues_per_head[head_i, -1, :]
                    added = head_contribution_last_token + cur_layer_input_last
                    added_probs = torch.log(get_prob(get_bsvalues(added, self.model, final_var))[predict_index])
                    increase = added_probs - origin_prob
                    all_head_increase.append([f"{test_layer}_{head_i}", float(increase.item())])

            print(f'Finished head-level increase time {time.time() - t}')

            # ----------------------------
            # Top-K selection and aggregation
            # ----------------------------
            # Select the K heads with largest positive ∆ log P and aggregate their
            # per-patch contributions into a single 576-dimensional map.
            # Aggregation uses a stable softmax over the head scores to form weights.
            all_head_increase_sort = sorted(all_head_increase, key=lambda x: x[-1])[::-1]
            top_k_heads_scores = all_head_increase_sort[:K]

            # Build a map from layer -> list of (head_index, score) for efficient iteration.
            heads_by_layer = {}
            for head_info, score in top_k_heads_scores:
                layer_s, head_s = head_info.split("_")
                layer_i = int(layer_s)
                head_i = int(head_s)
                if layer_i not in heads_by_layer:
                    heads_by_layer[layer_i] = []
                heads_by_layer[layer_i].append((head_i, float(score)))

            print(f"Top {K} heads (Layer_Head, LogIncrease):")
            for h_info, h_score in top_k_heads_scores:
                print(f"  {h_info}, {h_score:.4f}")

            # If no top heads were found, use a zero map (defensive).
            if len(top_k_heads_scores) == 0:
                final_aggregated_scores_raw = np.zeros(576, dtype=float)
            else:
                # Compute stable softmax weights over the selected heads' scores.
                scores_arr = np.array([s for (_, s) in top_k_heads_scores], dtype=float)
                if len(scores_arr) == 1 or np.allclose(scores_arr, scores_arr[0]):
                    weights = np.ones_like(scores_arr) / len(scores_arr)
                else:
                    exps = np.exp(scores_arr - scores_arr.max())
                    weights = exps / exps.sum()

                # Map (layer, head) -> normalized weight
                weight_map = {}
                for idx, (head_info, _) in enumerate(top_k_heads_scores):
                    layer_s, head_s = head_info.split("_")
                    weight_map[(int(layer_s), int(head_s))] = float(weights[idx])

                # Accumulate the weighted per-patch contributions from the selected heads.
                aggregated_scores = np.zeros(576, dtype=float)
                for test_layer, heads_info in heads_by_layer.items():
                    layer_input = all_pos_layer_input_i[test_layer].to(self.model.dtype).to(self.model.device)
                    att_layer = all_last_attn_subvalues_i[test_layer].to(self.model.dtype).to(self.model.device)
                    seq_len = layer_input.shape[0]

                    v_proj_module = self.model.language_model.layers[test_layer].self_attn.v_proj
                    V = v_proj_module(layer_input)
                    V_heads = V.view(seq_len, HEAD_NUM, HEAD_DIM).permute(1, 0, 2)

                    o_proj_module = self.model.language_model.layers[test_layer].self_attn.o_proj
                    o_proj_weight = o_proj_module.weight.data.to(self.model.dtype).to(self.model.device)
                    hidden_size = o_proj_weight.shape[0]
                    o_proj_weight_split = o_proj_weight.view(hidden_size, HEAD_NUM, HEAD_DIM).permute(1, 2, 0)

                    cur_layer_input_last = layer_input[-1]
                    origin_prob = torch.log(get_prob(get_bsvalues(cur_layer_input_last, self.model, final_var))[predict_index])

                    for head_index, _rawscore in heads_info:
                        # Recover this head's per-patch contribution vector
                        attn_out_this_head = torch.bmm(att_layer[head_index:head_index + 1], V_heads[head_index:head_index + 1])
                        attn_subvalues_this_head = torch.bmm(attn_out_this_head, o_proj_weight_split[head_index:head_index + 1])
                        attn_subvalues_this_head = attn_subvalues_this_head.squeeze(0)

                        # Compute per-patch ∆ log P contributions analogous to earlier step,
                        # but keep the whole per-patch vector (not just the last token scalar).
                        cur_attn_subvalues_plus = attn_subvalues_this_head + cur_layer_input_last.unsqueeze(0)
                        cur_attn_plus_probs = torch.log(get_prob(get_bsvalues(cur_attn_subvalues_plus, self.model, final_var))[:, predict_index])
                        cur_attn_plus_probs_increase = cur_attn_plus_probs - origin_prob
                        head_pos_increase = cur_attn_plus_probs_increase.tolist()

                        # Convert to the 576-element patch vector. Indexing [5:581] mirrors
                        # the original code's selection of image patch positions inside the sequence.
                        curhead_increase_scores = np.array(head_pos_increase[5:581], dtype=float)

                        # Weight by the normalized head weight and accumulate.
                        w = weight_map.get((test_layer, head_index), 0.0)
                        aggregated_scores += curhead_increase_scores * w

                final_aggregated_scores_raw = aggregated_scores

            # Debugging outputs that help reproduce the summary statistics in the paper.
            print(f"DEBUG: Aggregated map mean: {final_aggregated_scores_raw.mean():.6f}")
            print(f"DEBUG: Aggregated map max: {final_aggregated_scores_raw.max():.6f}")
            print(f"DEBUG: Aggregated map min: {final_aggregated_scores_raw.min():.6f}")

            # Normalize the final aggregated 576-d vector into a probability-like vector
            # suitable for entropy computation, clustering and visualization.
            increase_scores_normalize = normalize(final_aggregated_scores_raw.tolist())

            print(f'Finished getting patches time {time.time() - t}')

            # Return the processed image, normalized map, and model outputs for downstream analysis.
            batch_results.append((demo_img, increase_scores_normalize, outputs_probs_sort, outputs_probs))

        return batch_results

    # def save_vis(self, demo_img, increase_scores_normalize, output_path=None):
    #     ...
    #     (visualization helper left commented out intentionally)


# -----------------------
# Clustering & metrics helpers
# -----------------------
def transform_matrix_to_3d_points(array_2d: np.ndarray):
    """
    Convert a 24x24 attention matrix into an N x 3 array of (x, y, attention)
    coordinates used by the clustering pipeline. The y coordinate is flipped
    to align image coordinate conventions.
    """
    rows, cols = array_2d.shape
    result = np.empty([rows * cols, 3], dtype=object)

    for x in range(rows):
        for y in range(cols):
            result[x * cols + y] = [y, -x + 23, array_2d[x, y]]

    return result


def find_clusters(attentions_with_locations: np.ndarray, eps: float, min_samples: int, metric: str = "euclidean") -> (DBSCAN, int, int):
    """
    Run DBSCAN on the 2D coordinates (x, y) to identify dense attention regions.
    Returns the DBSCAN object and counts of clusters and noise points. This
    implements the 'DBSCAN Clustering' step from Section 3.4.
    """
    x_coords = attentions_with_locations[:, 0]
    y_coords = attentions_with_locations[:, 1]
    coords = np.stack((x_coords, y_coords), axis=-1)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(coords)
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    return db, n_clusters_, n_noise_


def apply_threshold(datapoints: np.ndarray, percentile: float) -> np.ndarray:
    """
    Filter out low-attention patches by percentile threshold (default 80th).
    This reduces noise and focuses DBSCAN on the strongest regions.
    """
    if len(datapoints) == 0:
        return datapoints
    z_values = datapoints[:, 2]
    p_value = np.percentile(z_values, percentile)
    print(f"{percentile}th percentile value: {p_value}")

    # If all z_values identical, return all points to avoid accidental drop.
    if len(np.unique(z_values)) == 1:
        return datapoints

    return datapoints[datapoints[:, 2] > p_value]


def duplicate_points(datapoints: np.ndarray, min_dup: int, max_dup: int) -> np.ndarray:
    """
    Duplicate points proportional to their attention strength so that DBSCAN
    sees stronger patches with higher density (Strength-Based Duplication).
    """
    if len(datapoints) == 0:
        return datapoints

    values = datapoints[:, 2]

    if values.max() == values.min():
        scaled = np.full(len(values), min_dup + 1, dtype=int)
    else:
        scaled = ((values - values.min()) / (values.max() - values.min()) * (max_dup - min_dup) + min_dup + 1).astype(int)

    weighted_points = np.concatenate([np.repeat([pt], rep, axis=0) for pt, rep in zip(datapoints, scaled)], axis=0)
    print(f"Original points: {len(datapoints)} -> After weighting: {len(weighted_points)}")
    return weighted_points


from dataclasses import dataclass

@dataclass
class ClusterData:
    # Stores summary values for clusters: number of weighted points and average attention strength.
    n_points: int
    ave_strength: float


def calculate_metrics(db: DBSCAN, weighted_attentions_with_locations):
    """
    Compute per-cluster metrics:
      - number of points per cluster (size)
      - average attention strength per cluster
    These are used in the paper's regression / correlation analysis with token confidence.
    """
    labels = db.labels_
    unique_clusters = set(labels) - {-1}  # Remove noise (-1)

    cluster_strengths = {}
    cluster_counts = Counter(labels)

    if len(weighted_attentions_with_locations) == 0:
        print("Warning: No data points to calculate metrics from.")
        return cluster_strengths

    z_values = weighted_attentions_with_locations[:, 2]

    for cluster in unique_clusters:
        cluster_points = z_values[labels == cluster]
        count = cluster_counts[cluster] if cluster in cluster_counts else 0

        if len(cluster_points) > 0:
            avg_strength = np.mean(cluster_points)
        else:
            avg_strength = 0.0

        cluster_strengths[cluster] = ClusterData(count, avg_strength)

    print("Average Strength and Number of Points per Cluster:")
    for cluster, data in cluster_strengths.items():
        print(f"Cluster {cluster}: {data}")

    return cluster_strengths


def calculate_entropy(probabilities: list):
    """
    Compute normalized Shannon entropy of a distribution (used in Section 3.5).
    Lower entropy indicates concentrated attention; higher entropy indicates dispersion.
    """
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    N = len(probabilities)
    if N <= 1:
        return 0.0
    max_entropy = np.log2(N)
    if max_entropy == 0:
        return 0.0
    normalized_entropy = entropy / max_entropy
    return normalized_entropy


def cluster_entropy(db: DBSCAN, weighted_attentions_with_locations):
    """
    Compute normalized entropy for each discovered cluster. This quantifies
    how concentrated attention is inside each detected cluster (Section 3.5).
    """
    labels = db.labels_
    unique_clusters = set(labels) - {-1}
    cluster_entropies = {}

    if len(weighted_attentions_with_locations) == 0:
        print("Warning: No data points to calculate cluster entropy from.")
        return cluster_entropies

    z_values = weighted_attentions_with_locations[:, 2]

    for cluster in unique_clusters:
        cluster_points = z_values[labels == cluster]
        total_count = len(cluster_points)
        if total_count > 1:
            counts = Counter(cluster_points)
            if len(counts) <= 1:
                normalized_entropy = 0.0
            else:
                probabilities = [count / total_count for count in counts.values()]
                entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
                max_entropy = np.log2(len(counts))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            normalized_entropy = 0.0
        cluster_entropies[cluster] = normalized_entropy

    print("Entropy per Cluster:")
    for cluster, entropy in cluster_entropies.items():
        print(f"Cluster {cluster}: {entropy:.4f}")

    return cluster_entropies


def calculate_token_entropy(token_list: list):
    """
    Compute entropy over a list of predicted tokens. Useful for token-level
    diversity / uncertainty analysis described in the methodology.
    """
    if not token_list:
        return 0.0

    total_count = len(token_list)
    counts = Counter(token_list)

    if len(counts) <= 1:
        return 0.0

    probabilities = [count / total_count for count in counts.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    max_entropy = np.log2(len(counts))

    if max_entropy == 0:
        return 0.0

    normalized_entropy = entropy / max_entropy
    return normalized_entropy


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# -----------------------
# Data I/O helpers
# -----------------------
def load_dataset_from_csv(csv_path='results.csv', limit=None):
    """
    Load a CSV file that contains the VQA examples:
      - expected columns: image_url, question, prefix, ground_truth, question_type
    The script converts 'https' to 'http' to avoid some SSL issues when fetching COCO images.
    """
    df = pd.read_csv(csv_path)
    df['image_url'] = df['image_url'].str.replace('https://', 'http://')
    if limit:
        df = df.head(limit)
    imagePrompts = []
    for _, row in df.iterrows():
        imagePrompts.append(ImagePrompt(
            image_url=row['image_url'],
            prompt=row['question'],
            prefix=row['prefix']
        ))
    return imagePrompts, df


def save_results_to_csv(results_data, output_path='analysis_results.csv'):
    """
    Persist aggregated metrics for downstream regression and plotting (the paper's results).
    """
    df = pd.DataFrame(results_data)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


# -----------------------
# Main execution loop
# -----------------------
def main():
    """
    Iterate over the dataset, compute top-k aggregated patch maps per image,
    perform clustering and entropy calculations, and save summary metrics.

    The --top_k CLI argument controls how many top heads to include in the aggregation.
    """
    parser = argparse.ArgumentParser(description="Top-K head aggregation and clustering analysis (matches paper pipeline).")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top heads to aggregate (K).")
    parser.add_argument("--csv", type=str, default="results.csv", help="CSV file with dataset rows.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of rows to process.")
    args = parser.parse_args()

    mechanism = LlavaMechanism()

    print("Loading dataset from CSV...")
    imagePrompts, original_df = load_dataset_from_csv(args.csv, limit=args.limit)
    print(f"Loaded {len(imagePrompts)} images")
    all_results = []

    TOP_K_HEADS = args.top_k
    print(f"Processing images using Top-{TOP_K_HEADS} head aggregation...")

    for i in range(0, len(imagePrompts), batch_size):
        chunk = imagePrompts[i:i + batch_size]
        chunk_df = original_df.iloc[i:i + batch_size]
        try:
            images = [Image.open(requests.get(ip.image_url, stream=True, verify=False).raw) for ip in chunk]
            prompts = [ip.prompt for ip in chunk]
            prefixes = [ip.prefix for ip in chunk]
        except Exception as e:
            # If an image fails to load, skip the batch and continue processing.
            print(f"Error loading images in batch {i // batch_size}: {e}")
            continue

        try:
            batch_results = mechanism.get_attention_patches(images, prompts, prefixes, K=TOP_K_HEADS)
        except Exception as e:
            # Catch processing errors per-batch and continue.
            print(f"Error processing batch {i // batch_size}: {e}")
            continue

        # Each item returned by get_attention_patches includes:
        # (cropped_image, normalized_576d_vector, sorted_token_indices, probs_tensor)
        for j, (ip, (demo_img, increase_scores_normalize, outputs_probs_sort, outputs_probs)) in enumerate(zip(chunk, batch_results)):
            idx = i + j
            row = chunk_df.iloc[j]
            print(f"\n{'=' * 80}")
            print(f"Process image {idx} - {row['question_type']}")
            print(f"Question: {row['question']}")
            print(f"Ground truth: {row['ground_truth']}")
            print(f"Image: {ip.image_url}")
            print(f"{'=' * 80}")

            # 1) Compute global entropy across the 576 patches (Section 3.5).
            scores_normalize_1d_list = increase_scores_normalize
            attention_entropy = calculate_entropy(scores_normalize_1d_list)

            # 2) Convert the 576-d vector into a 24x24 matrix and transform to 3D (x,y,score)
            scores_normalize_2d_array = np.array(scores_normalize_1d_list).reshape(24, 24)
            attentions_with_locations = transform_matrix_to_3d_points(scores_normalize_2d_array)
            print(f"Attentions with locations: {attentions_with_locations.shape}")

            # 3) Filter low-attention patches by percentile, weight by strength, cluster, and compute metrics
            threshold_percentile = 80
            filtered_attentions_with_locations = apply_threshold(attentions_with_locations, threshold_percentile)
            print(f"Attentions without the lowest {threshold_percentile}% datapoints: {filtered_attentions_with_locations.shape}")
            weighted_attentions_with_locations = duplicate_points(filtered_attentions_with_locations, 1, 9)
            db, n_clusters, n_noise = find_clusters(weighted_attentions_with_locations, 1.3, 15)
            cluster_metrics = calculate_metrics(db, weighted_attentions_with_locations)

            # Token-level confidence (log-prob of the top predicted token)
            predict_index = outputs_probs_sort[0].item()
            token_confidence = float(torch.log(outputs_probs[predict_index]).item())

            # Print metrics used in the paper for correlation/regression analysis.
            print(f"Attention Entropy: {attention_entropy:.4f}")
            print(f"Token Confidence: {token_confidence:.4f}")

            cluster_entropies = cluster_entropy(db, weighted_attentions_with_locations)

            # Compose a results row for downstream analysis (R^2 regressions, tables, etc.)
            result_dict = {
                'image_id': idx,
                'question_type': row['question_type'],
                'question': row['question'],
                'ground_truth': row['ground_truth'],
                'image_url': ip.image_url,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'attention_entropy': attention_entropy,
                'token_confidence': token_confidence,
                'predicted_token': mechanism.processor.decode(predict_index),
                'cluster_count': len(cluster_metrics),
            }

            # Append cluster-specific metrics (counts and average strengths)
            for cluster_id, metrics in cluster_metrics.items():
                result_dict[f'cluster_{cluster_id}_n_points'] = metrics.n_points
                result_dict[f'cluster_{cluster_id}_avg_strength'] = metrics.ave_strength

            # Add per-cluster entropies
            for cluster_id, entropy in cluster_entropies.items():
                result_dict[f'cluster_{cluster_id}_entropy'] = entropy

            all_results.append(result_dict)

            # Periodically persist intermediate results (useful for long runs).
            if (idx + 1) % 10 == 0:
                save_results_to_csv(all_results, f'analysis_results_partial_{idx + 1}.csv')

    # Final save at the end of the run.
    save_results_to_csv(all_results, 'analysis_results_final_topk.csv')
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"Processed {len(all_results)} images")
    print("=" * 80)


if __name__ == "__main__":
    main()
