"""Legacy batching module superseded by analysis.pipeline_runner."""

from analysis.pipeline_runner import main as pipeline_main


def main() -> None:
    pipeline_main()


if __name__ == "__main__":  # pragma: no cover
    main()
    raise SystemExit

# ---------------------------------------------------------------------------
# Legacy code retained for archival purposes below.
# ---------------------------------------------------------------------------

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

LAYER_NUM = 32
HEAD_NUM = 32
HEAD_DIM = 128
HIDDEN_DIM = HEAD_NUM * HEAD_DIM

batch_size = 4 # 5 is the highest it can go without running out of CUDA memory

@dataclass
class ImagePrompt:
    image_url: str
    prompt: str
    prefix: str

@dataclass
class Entropy:
    cluster: int
    count_points: int
    average_strength: float

def normalize(vector):
    max_value = max(vector)
    min_value = min(vector)
    vector1 = [(x-min_value)/(max_value-min_value) for x in vector]
    vector2 = [x/sum(vector1) for x in vector1]
    return vector2

def transfer_output(model_output, batch_size=5):
    all_pos_layer_input = []
    all_pos_layer_output = []
    all_last_attn_subvalues = []

    for layer_i in range(LAYER_NUM):
        batch_inputs = []
        batch_outputs = []
        batch_subvalues = []
        
        for b in range(batch_size):
            layer_data = model_output[layer_i]
            cur_layer_input = layer_data[0][b] if layer_data[0].dim() > 1 else layer_data[0]
            cur_layer_output = layer_data[4][b] if layer_data[4].dim() > 1 else layer_data[4]
            cur_last_attn_subvalues = layer_data[5][b] if layer_data[5].dim() > 1 else layer_data[5]
            
            batch_inputs.append(cur_layer_input.tolist())
            batch_outputs.append(cur_layer_output.tolist())
            batch_subvalues.append(cur_last_attn_subvalues.tolist())
        
        all_pos_layer_input.append(batch_inputs)
        all_pos_layer_output.append(batch_outputs)
        all_last_attn_subvalues.append(batch_subvalues)

    return all_pos_layer_input, all_pos_layer_output, all_last_attn_subvalues

def get_bsvalues(vector, model, final_var):
    vector = vector * torch.rsqrt(final_var + 1e-6)
    vector_rmsn = vector * model.language_model.model.norm.weight.data
    vector_bsvalues = model.language_model.lm_head(vector_rmsn).data
    return vector_bsvalues

def get_prob(vector):
    prob = torch.nn.Softmax(-1)(vector)
    return prob

def transfer_l(l):
    new_x, new_y = [], []
    for x in l:
        new_x.append(x[0])
        new_y.append(x[1])
    return new_x, new_y

def plt_bar(x, y, yname="log increase"):
    x_major_locator=MultipleLocator(1)
    plt.figure(figsize=(8, 3))
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt_x = [a/2 for a in x]
    plt.xlim(-0.5, plt_x[-1]+0.49)
    x_attn, y_attn, x_ffn, y_ffn = [], [], [], []
    for i in range(len(x)):
        if i%2 == 0:
            x_attn.append(x[i]/2)
            y_attn.append(y[i])
        else:
            x_ffn.append(x[i]/2)
            y_ffn.append(y[i])
    plt.bar(x_attn, y_attn, color="darksalmon", label="attention layers")
    plt.bar(x_ffn, y_ffn, color="lightseagreen", label="FFN layers")
    plt.xlabel("layer")
    plt.ylabel(yname)
    plt.legend()
    plt.show()

def plt_heatmap(data):
    xLabel = range(len(data[0]))
    yLabel = range(len(data))
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.set_xticks(range(len(xLabel)))
    ax.set_xticklabels(xLabel)
    ax.set_yticks(range(len(yLabel)))
    ax.set_yticklabels(yLabel)
    im = ax.imshow(data, cmap=plt.cm.hot_r)
    plt.title("attn head log increase heatmap")
    plt.show()

class LlavaMechanism:
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf", device="cuda"):
        """
        Initialize the LlavaMechanism class by loading the model and processor.
        
        Args:
            model_id (str): The model ID to load
            device (str): Device to run the model on
        """
        # Setup CUDA
        torch.set_default_device('cuda')
        
        # Load model
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            low_cpu_mem_usage=True, 
            revision='a272c74'
        ).to(device)
        
        print(f"Model loaded on {self.model.device}")
        self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained(model_id, revision='a272c74')
        self.processor.patch_size = 14 # added
        print(f"Model loaded on {self.model.device}")
        self.model.eval()
        
        # Create output directory for saved visualizations
        self.output_dir = "output_images"
        os.makedirs(self.output_dir, exist_ok=True)

    def get_attention_patches(self, images, prompts, prefixes):
        batch_results = []
        batch_size = len(images)
        t = time.time()
        
        full_prompts = [f"USER: <image>\n{p}\nASSISTANT: {pref}" for p, pref in zip(prompts, prefixes)]
        inputs = self.processor(text=full_prompts, images=images, return_tensors="pt", padding=True).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
        
        print(f'Finished inference time {time.time() - t}')

        transfer_outputs = transfer_output(outputs[2], batch_size)
        
        for i in range(batch_size):
            all_pos_layer_input_i = [layer[i] for layer in transfer_outputs[0]]
            all_pos_layer_output_i = [layer[i] for layer in transfer_outputs[1]] 
            all_last_attn_subvalues_i = [layer[i] for layer in transfer_outputs[2]]
            
            logits_i = outputs["logits"][i]
            outputs_probs = get_prob(logits_i[-1])
            outputs_probs_sort = torch.argsort(outputs_probs, descending=True)
            print([self.processor.decode(x) for x in outputs_probs_sort[:10]])
            
            final_var = torch.tensor(all_pos_layer_output_i[-1][-1]).pow(2).mean(-1, keepdim=True)
            
            image_i = images[i]
            image_convert = convert_to_rgb(image_i)
            image_numpy = to_numpy_array(image_convert)
            input_data_format = infer_channel_dimension_format(image_numpy)
            output_size = get_resize_output_image_size(image_numpy, size=336, default_to_square=False)
            image_resize = resize(image_numpy, output_size, resample=3, input_data_format=input_data_format)
            image_center_crop = center_crop(image_resize, size=(336, 336), input_data_format=input_data_format)
            demo_img = image_center_crop
            
            predict_index = outputs_probs_sort[0].item()
            print(predict_index, self.processor.decode(predict_index))

            all_head_increase = []
            for test_layer in range(LAYER_NUM):
                cur_layer_input = torch.tensor(all_pos_layer_input_i[test_layer])
                cur_v_heads = torch.tensor(all_last_attn_subvalues_i[test_layer])
                cur_attn_o_split = self.model.language_model.model.layers[test_layer].self_attn.o_proj.weight.data.T.view(HEAD_NUM, HEAD_DIM, -1)
                
                cur_attn_subvalues_headrecompute = torch.bmm(cur_v_heads, cur_attn_o_split).permute(1, 0, 2)
                cur_attn_subvalues_head_sum = torch.sum(cur_attn_subvalues_headrecompute, 0)
                cur_layer_input_last = cur_layer_input[-1]
                
                origin_prob = torch.log(get_prob(get_bsvalues(cur_layer_input_last, self.model, final_var))[predict_index])
                cur_attn_subvalues_head_plus = cur_attn_subvalues_head_sum + cur_layer_input_last
                cur_attn_plus_probs = torch.log(get_prob(get_bsvalues(
                        cur_attn_subvalues_head_plus, self.model, final_var))[:, predict_index])
                cur_attn_plus_probs_increase = cur_attn_plus_probs - origin_prob
                for j in range(len(cur_attn_plus_probs_increase)):
                    all_head_increase.append([f"{test_layer}_{j}", round(cur_attn_plus_probs_increase[j].item(), 4)])

            print(f'Finished head-level increase time {time.time() - t}')
            
            all_head_increase_sort = sorted(all_head_increase, key=lambda x: x[-1])[::-1]
            test_layer, head_index = all_head_increase_sort[0][0].split("_")
            test_layer, head_index = int(test_layer), int(head_index)
            
            cur_layer_input = outputs[2][test_layer][0][i]
            cur_v_heads = outputs[2][test_layer][5][i]
            cur_attn_o_split = self.model.language_model.model.layers[test_layer].self_attn.o_proj.weight.data.T.view(HEAD_NUM, HEAD_DIM, -1)
            cur_attn_subvalues_headrecompute = torch.bmm(cur_v_heads, cur_attn_o_split).permute(1, 0, 2)
            cur_attn_subvalues_headrecompute_curhead = cur_attn_subvalues_headrecompute[:, head_index, :]
            
            cur_layer_input_last = cur_layer_input[-1]
            origin_prob = torch.log(get_prob(get_bsvalues(
                cur_layer_input_last, self.model, final_var))[predict_index])
            
            cur_attn_subvalues_headrecompute_curhead_plus = cur_attn_subvalues_headrecompute_curhead + cur_layer_input_last
            cur_attn_plus_probs = torch.log(get_prob(get_bsvalues(
                cur_attn_subvalues_headrecompute_curhead_plus, self.model, final_var))[:, predict_index])
            
            cur_attn_plus_probs_increase = cur_attn_plus_probs - origin_prob
            head_pos_increase = cur_attn_plus_probs_increase.tolist()
            curhead_increase_scores = head_pos_increase[5:581]
            
            increase_scores_normalize = normalize(curhead_increase_scores)
            
            print(f'Finished getting patches time {time.time() - t}')

            batch_results.append((demo_img, increase_scores_normalize, outputs_probs_sort))
            
        return batch_results

    # def save_vis(self, demo_img, increase_scores_normalize, output_path=None):
    #     if output_path is None:
    #         output_path = os.path.join(self.output_dir, "attention_analysis.png")
    #     demo_img_h, demo_img_w, demo_img_c = demo_img.shape
        
    #     demo_img_inc = np.array(increase_scores_normalize).reshape((24, 24))
    #     demo_img_inc = cv2.resize(demo_img_inc,
    #                             dsize=(demo_img_w, demo_img_h),
    #                             interpolation=cv2.INTER_CUBIC)
        
    #     plt.figure(figsize=(25, 6))
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(demo_img)
    #     plt.axis("off")
    #     plt.title("image")
        
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(demo_img)
    #     plt.imshow(demo_img_inc, alpha=0.8, cmap="gray")
    #     plt.axis("off")
    #     plt.title("log increase")
        
    #     plt.savefig(output_path)
    #     plt.close()
    #     print(f"Visualization saved to {output_path}")

def transform_matrix_to_3d_points(array_2d: np.ndarray):
    rows, cols = array_2d.shape    
    result = np.empty([rows * cols, 3], dtype=object)

    for x in range(rows):
        for y in range(cols):
            result[x * cols + y] = [y, -x + 23, array_2d[x, y]]

    return result

def find_clusters(attentions_with_locations: np.ndarray, eps: float, min_samples: int, metric: str="euclidean") -> (DBSCAN, int, int):
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
    z_values = datapoints[:, 2]
    p_value = np.percentile(z_values, percentile)
    print(f"{percentile}th percentile value: {p_value}")
    return datapoints[datapoints[:, 2] > p_value]

def duplicate_points(datapoints: np.ndarray, min_dup: int, max_dup: int) -> np.ndarray:
    values = datapoints[:, 2]
    scaled = ((values - values.min()) / (values.max() - values.min()) * (max_dup - min_dup) + min_dup + 1).astype(int)
    weighted_points = np.concatenate([np.repeat([pt], rep, axis=0) for pt, rep in zip(datapoints, scaled)], axis=0)
    print(f"Original points: {len(datapoints)} -> After weighting: {len(weighted_points)}")
    return weighted_points

# def save_attentions(weighted_attentions_with_locations: np.ndarray, db: DBSCAN, image_url: str):
#     parsed_url = urllib.parse.urlparse(image_url)
#     filename = os.path.basename(parsed_url.path)
#     plt.scatter(weighted_attentions_with_locations[:, 0], weighted_attentions_with_locations[:, 1], c=db.labels_)
#     plt.show()
#     plt.savefig(os.path.join("output_images", "attention_analysis_" + filename))
#     plt.close()

from dataclasses import dataclass

@dataclass
class ClusterData:
    n_points: int
    ave_strength: float
    
def calculate_metrics(db: DBSCAN, weighted_attentions_with_locations):
    labels = db.labels_
    unique_clusters = set(labels) - {-1}  # Remove noise (-1)

    cluster_strengths = {}

    # Count the number of points per cluster
#    cluster_counts = Counter(labels)

#    print("Points per cluster:")
#    for label, count in cluster_counts.items():
#        if label != -1:
#            cluster_strengths[label] = ClusterData(count, 0)

    # 
    z_values = weighted_attentions_with_locations[:, 2]

    for cluster in unique_clusters:
        cluster_points = z_values[labels == cluster]  # Get strength values for the cluster

        if cluster in cluster_strengths:
            cluster_strengths[cluster].ave_strength = np.mean(cluster_points)
        else:
            cluster_strengths[cluster] = ClusterData(0, np.mean(cluster_points))

    # Print average strength of each cluster
    print("Average Strength and Number of Points per Cluster:")
    for cluster, data in cluster_strengths.items():
        print(f"Cluster {cluster}: {data}")
        
    return cluster_strengths

def calculate_entropy(datapoints: list):
    flat_list = [item for sublist in datapoints for item in sublist]
    total_count = len(flat_list)
    counts = Counter(flat_list)
    probabilities = [count / total_count for count in counts.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    max_entropy = np.log2(len(counts))
    normalized_entropy = entropy / max_entropy
    return normalized_entropy

def cluster_entropy(db:DBSCAN, weighted_attentions_with_locations):
    labels = db.labels_
    unique_clusters = set(labels) - {-1}  # Remove noise (-1)

    cluster_strengths = {}
    
    z_values = weighted_attentions_with_locations[:, 2]

    for cluster in unique_clusters:
        cluster_points = z_values[labels == cluster]  # Get strength values for the cluster
        
        if cluster in cluster_strengths:
            total_count = len(cluster_points)
            counts = Counter(cluster_points)
            probabilities = [count / total_count for count in counts.values()]
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            max_entropy = np.log2(len(counts))
            normalized_entropy = entropy / max_entropy
            cluster_strengths[cluster].ave_strength = normalized_entropy

    # Print average strength of each cluster
    print("Entropy per Cluster:")
    for cluster, data in cluster_strengths.items():
        print(f"Cluster {cluster}: {data}")
        
    return cluster_strengths

def calculate_token_entropy(token_list: list):
    total_count = len(token_list)
    counts = Counter(token_list)
    probabilities = [count / total_count for count in counts.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    max_entropy = np.log2(len(counts))
    normalized_entropy = entropy / max_entropy
    return normalized_entropy

def main():
    """
    Main function to demonstrate the usage of LlavaMechanism class.
    """
    # Create LlavaMechanism instance
    mechanism = LlavaMechanism()
    
    # Create image prompts
    imagePrompts = [
        ImagePrompt(
            image_url="http://images.cocodataset.org/val2017/000000219578.jpg",
            prompt="What is the color of the dog?",
            prefix="the color of the dog is"
        ) for _ in range(100)
    ]

    print("Processing images...")
    # Process images in batches
    for i in range(0, len(imagePrompts), batch_size):
        chunk = imagePrompts[i:i + batch_size]
        images = [Image.open(requests.get(ip.image_url, stream=True).raw) for ip in chunk]
        prompts = [ip.prompt for ip in chunk]
        prefixes = [ip.prefix for ip in chunk]

        # Get attention patches
        batch_results = mechanism.get_attention_patches(images, prompts, prefixes)
        
        # Process each image result
        for j, (ip, (demo_img, increase_scores_normalize, outputs_probs_sort)) in enumerate(zip(chunk, batch_results)):
            print(f"\nProcess image {i+j} - {ip.image_url}")
            
            # Convert attention scores to 2D array
            increase_scores_normalize = np.array(increase_scores_normalize)
            increase_scores_normalize = increase_scores_normalize.reshape(24, 24)
        
            # Transform to 3D points
            attentions_with_locations = transform_matrix_to_3d_points(increase_scores_normalize)
            print(f"Attentions with locations: ", attentions_with_locations.shape)
            
            # Apply threshold
            threshold_percentile = 80
            filtered_attentions_with_locations = apply_threshold(attentions_with_locations, threshold_percentile)
            print(f"Attentions without the lowest {threshold_percentile}% datapoints: ", filtered_attentions_with_locations.shape)
            
            # Duplicate points based on attention strength
            weighted_attentions_with_locations = duplicate_points(filtered_attentions_with_locations, 1, 9)
            
            # Find clusters
            db, n_clusters, n_noise = find_clusters(weighted_attentions_with_locations, 1.3, 15)

            # Apply Euclidean distance to evaluate spatial proximity
            # epsilon = 1.5 - eps should be >=1 since the minimum distance between 2 adjacent attentions is 1.
            # min_samples = 15
            db, _, _ = find_clusters(weighted_attentions_with_locations, 1.3, 15)
            calculate_metrics(db, weighted_attentions_with_locations)
    
            # Calculate entropy.
            entropy = calculate_entropy(increase_scores_normalize)
            token_entropy = calculate_token_entropy(outputs_probs_sort)
            print(f"Attention Entropy: {entropy:.4f}")
            print(f"Token Entropy: {token_entropy:.4f}")
            cluster_entropy(db, weighted_attentions_with_locations)

if __name__ == "__main__":
    main()
