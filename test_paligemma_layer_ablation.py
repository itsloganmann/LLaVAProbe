"""
PaliGemma Layer Ablation Analysis - Run on 500 Images
Tests vision cutoff ablation (early fusion / late fusion) on PaliGemma.
"""

from PIL import Image
import requests
import torch
import json
import os
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# ========= Google Drive Setup (Auto-detects Colab) =========
try:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DRIVE_DIR = "/content/drive/MyDrive/paligemma_ablation_runs"
    IS_COLAB = True
    print("🟢 Google Drive mounted successfully")
except Exception:
    BASE_DRIVE_DIR = "./paligemma_ablation_runs"
    IS_COLAB = False
    print("🟡 Colab not detected — saving locally instead")

# Create timestamped folder for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_FOLDER = os.path.join(BASE_DRIVE_DIR, f"run_{timestamp}")
os.makedirs(RUN_FOLDER, exist_ok=True)
print(f"📁 Results will be saved to: {RUN_FOLDER}")
# ======================================================================

print("="*70)
print("PALIGEMMA LAYER ABLATION ANALYSIS - 500 IMAGES")
print("="*70)
print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")

# Configuration
NUM_IMAGES = 500
OUTPUT_FILE = os.path.join(RUN_FOLDER, "layer_ablation_results.json")
CHECKPOINT_DIR = os.path.join(RUN_FOLDER, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CHECKPOINT_INTERVAL = 50  # Save checkpoint every N images
PROCESSED_DATA_PATH = "data_processing/data/processed/filtered_vqa_with_links.json"

DEFAULT_PROMPT = "What do you see in this image?"
DEFAULT_PREFIX = "This image shows"

# Ablation configurations to test
ABLATION_CONFIGS = [
    {"mode": "baseline", "vision_cutoff_mode": None, "vision_cutoff_layer": None},
    {"mode": "early_cut_2", "vision_cutoff_mode": "early_cut", "vision_cutoff_layer": 2},
    {"mode": "early_cut_4", "vision_cutoff_mode": "early_cut", "vision_cutoff_layer": 4},
    {"mode": "early_cut_8", "vision_cutoff_mode": "early_cut", "vision_cutoff_layer": 8},
    {"mode": "late_only_2", "vision_cutoff_mode": "late_only", "vision_cutoff_layer": 2},
    {"mode": "late_only_4", "vision_cutoff_mode": "late_only", "vision_cutoff_layer": 4},
    {"mode": "late_only_8", "vision_cutoff_mode": "late_only", "vision_cutoff_layer": 8},
]


def load_image_from_url(url: str, timeout: int = 10) -> Image.Image:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"  ⚠️  Error loading {url}: {e}")
        return None


def load_dataset(num_images: int) -> List[Dict[str, Any]]:
    if os.path.exists(PROCESSED_DATA_PATH):
        print(f"✅ Loading dataset from {PROCESSED_DATA_PATH}")
        with open(PROCESSED_DATA_PATH, "r") as f:
            data = json.load(f)
        print(f"✅ Found {len(data)} samples in dataset")
        return data[:num_images]
    
    print(f"⚠️  VQA dataset not found at {PROCESSED_DATA_PATH}")
    print(f"   Using fallback: COCO val2017 images")
    
    base_valid_ids = [
        139, 285, 632, 724, 776, 785, 802, 872, 885, 1000,
        1268, 1296, 1425, 1503, 1584, 1761, 1818, 1993, 2006, 2149,
        2153, 2157, 2261, 2299, 2473, 2532, 2587, 2592, 2685, 2923,
        2975, 3156, 3255, 3501, 3661, 3845, 4134, 4395, 4765, 5001,
        5037, 5060, 5477, 5992, 6040, 6213, 6460, 6471, 6614, 6723,
        6954, 7108, 7278, 7386, 7511, 7574, 7816, 7888, 8021, 8277,
        8532, 8629, 9448, 9590, 9769, 9891, 10092, 10363, 10707, 10977,
        11197, 11511, 12062, 12280, 12576, 12667, 13004, 13177, 13546, 14007,
        14205, 14439, 14473, 14888, 15029, 15254, 15335, 15597, 15956, 16228,
        16439, 16598, 17029, 17178, 17207, 17714, 17899, 18150, 18380, 18519
    ]
    
    fallback_data = []
    for i in range(num_images):
        image_id = base_valid_ids[i % len(base_valid_ids)]
        image_id_str = str(image_id).zfill(12)
        fallback_data.append({
            "image_url": f"http://images.cocodataset.org/val2017/{image_id_str}.jpg",
            "question_text": DEFAULT_PROMPT,
            "answer": "unknown",
            "category": "general"
        })
    
    return fallback_data


def serialize_result(output, image_url: str, index: int, question: str, ablation_config: Dict, layer_evolution: Dict = None) -> Dict[str, Any]:
    """Convert model output to serializable dictionary."""
    result = {
        "index": index,
        "image_url": image_url,
        "question": question,
        "ablation_mode": ablation_config["mode"],
        "predicted_answer": output.predicted_answer,
        "token_confidence": float(output.token_confidence),
        "head_delta": float(output.head_delta),
        "attention_map_shape": list(output.attention_map.shape),
        "attention_map_mean": float(np.mean(output.attention_map)),
        "attention_map_std": float(np.std(output.attention_map)),
        "num_generated_tokens": len(output.predicted_token_ids),
        "token_strings": output.token_strings,
        "token_ids": [int(tid) for tid in output.predicted_token_ids],
    }
    
    # Add layer evolution metrics if available
    if layer_evolution and "summary" in layer_evolution:
        result["layer_evolution"] = {
            "most_diffuse_layer": int(layer_evolution["summary"]["most_diffuse_layer"]),
            "most_focused_layer": int(layer_evolution["summary"]["most_focused_layer"]),
            "highest_shift_layer": int(layer_evolution["summary"].get("highest_shift_layer", -1)),
            "most_diverse_heads_layer": int(layer_evolution["summary"].get("most_diverse_heads_layer", -1)),
            "total_entropy_change": float(layer_evolution["summary"]["total_entropy_change"]),
            "max_single_shift": float(layer_evolution["summary"].get("max_single_shift", 0.0)),
        }
        
        # Per-layer metrics
        if "per_layer_metrics" in layer_evolution:
            result["per_layer_metrics"] = [
                {
                    "layer": int(m["layer_index"]),
                    "entropy": float(m["entropy"]),
                    "entropy_shift": float(m["entropy_shift"]),
                    "sparsity": float(m.get("sparsity", 0.0)),
                    "sparsity_shift": float(m.get("sparsity_shift", 0.0)),
                    "kl_divergence": float(m.get("kl_divergence", 0.0)),
                    "head_diversity": float(m.get("head_diversity", 0.0)),
                    "is_critical_entropy": bool(m.get("is_critical_entropy", False)),
                    "is_critical_multimetric": bool(m.get("is_critical_multimetric", False)),
                }
                for m in layer_evolution["per_layer_metrics"]
            ]
    
    return result


def save_checkpoint(results: List[Dict], checkpoint_num: int):
    """Save checkpoint to Google Drive folder."""
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"checkpoint_{checkpoint_num}.json")
    with open(checkpoint_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  💾 Checkpoint saved to Drive: checkpoint_{checkpoint_num}.json")


def main():
    print(f"Testing {len(ABLATION_CONFIGS)} ablation configurations")
    print()
    
    print(f"📦 Loading dataset ({NUM_IMAGES} images)...")
    dataset = load_dataset(NUM_IMAGES)
    actual_num = min(NUM_IMAGES, len(dataset))
    print(f"✅ Will process {actual_num} images\n")
    
    print("🔧 Loading PaliGemma model...")
    from analysis.paligemma_runner import PaliGemmaRunner
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = PaliGemmaRunner(
        model_id="google/paligemma-3b-pt-224",
        device=device,
        quantization="4bit" if device == "cuda" else None
    )
    print(f"✅ Model loaded on {device}")
    print(f"   Number of layers: {runner.num_layers}")
    print()
    
    all_results = []
    successful = 0
    failed = 0
    
    print("🚀 Starting ablation experiments...\n")
    start_time = datetime.now()
    
    for idx, sample in enumerate(dataset[:actual_num], start=1):
        image_url = sample.get("image_url", "")
        question = sample.get("question_text", DEFAULT_PROMPT)
        
        print(f"[{idx}/{actual_num}] Processing: {image_url[:60]}...")
        image = load_image_from_url(image_url)
        if image is None:
            failed += 1
            print(f"  ❌ Skipped (failed to load)\n")
            continue
        
        # Run each ablation configuration
        for ablation_config in ABLATION_CONFIGS:
            try:
                # Reset vision cutoff state
                if ablation_config["vision_cutoff_mode"]:
                    runner.set_vision_cutoff(
                        ablation_config["vision_cutoff_mode"],
                        ablation_config["vision_cutoff_layer"]
                    )
                else:
                    # Baseline: disable vision cutoff
                    runner.model.language_model.vision_cutoff["enabled"] = False
                
                output = runner.run(
                    image=image,
                    prompt=question,
                    prefix=DEFAULT_PREFIX,
                )
                
                # Print layer evolution if available
                if output.layer_evolution:
                    from analysis.clustering import print_layer_evolution
                    print(f"\n==== Layer Evolution ({ablation_config['mode']}) ====")
                    print_layer_evolution(output.layer_evolution)
                
                result = serialize_result(output, image_url, idx, question, ablation_config, output.layer_evolution)
                all_results.append(result)
                
            except Exception as e:
                print(f"  ❌ Error for {ablation_config['mode']}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        successful += 1
        print(f"  ✅ Success | Processed {len(ABLATION_CONFIGS)} configurations\n")
        
        # Save checkpoint every 50 images
        if idx % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(all_results, idx)
            print()
    
    print("="*70)
    print("💾 Saving final results...")
    
    output_data = {
        "metadata": {
            "total_images": actual_num,
            "successful": successful,
            "failed": failed,
            "timestamp": datetime.now().isoformat(),
            "device": device,
            "model": "google/paligemma-3b-pt-224",
            "ablation_configs": ABLATION_CONFIGS,
            "num_layers": runner.num_layers,
            "prompt": DEFAULT_PROMPT,
            "prefix": DEFAULT_PREFIX,
        },
        "results": all_results
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output_data, f, indent=2)
    
    # Save summary CSV
    import pandas as pd
    df_results = pd.json_normalize(all_results)
    csv_file = os.path.join(RUN_FOLDER, "ablation_results.csv")
    df_results.to_csv(csv_file, index=False)
    
    elapsed = datetime.now() - start_time
    
    print("="*70)
    print("✅ ALL RESULTS SAVED TO GOOGLE DRIVE")
    print("="*70)
    print(f"📁 Run folder: {RUN_FOLDER}")
    print(f"📄 Final results: layer_ablation_results.json")
    print(f"📦 Checkpoints folder: checkpoints/")
    print(f"   (Contains checkpoint_50.json, checkpoint_100.json, etc.)")
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total images processed: {successful}/{actual_num}")
    print(f"Failed: {failed}")
    print(f"Total ablation runs: {len(all_results)}")
    print(f"Time elapsed: {elapsed}")
    if successful > 0:
        print(f"Average time per image: {elapsed.total_seconds()/successful:.2f}s")
    if IS_COLAB:
        print(f"\n🔗 Access your results in Google Drive:")
        print(f"   MyDrive/paligemma_ablation_runs/run_{timestamp}/")
    print("="*70)
    
    return all_results


if __name__ == "__main__":
    main()

