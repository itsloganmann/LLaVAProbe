"""
Layer Evolution Analysis - PaliGemma 2 - Run on 1000 Images
Processes multiple images and saves layer evolution results to JSON.
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
    BASE_DRIVE_DIR = "/content/drive/MyDrive/paligemma_probe_runs"
    IS_COLAB = True
    print("🟢 Google Drive mounted successfully")
except Exception:
    BASE_DRIVE_DIR = "./paligemma_probe_runs"
    IS_COLAB = False
    print("🟡 Colab not detected — saving locally instead")

# Create timestamped folder for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_FOLDER = os.path.join(BASE_DRIVE_DIR, f"run_{timestamp}")
os.makedirs(RUN_FOLDER, exist_ok=True)
print(f"📁 Results will be saved to: {RUN_FOLDER}")
# ======================================================================

print("="*70)
print("PALIGEMMA 2 - LAYER EVOLUTION ANALYSIS")
print("="*70)
print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")

# Configuration
NUM_IMAGES = 1000   # Total number of images to process

# All results save to Google Drive folder
OUTPUT_FILE = os.path.join(RUN_FOLDER, "layer_evolution_results.json")
CHECKPOINT_DIR = os.path.join(RUN_FOLDER, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CHECKPOINT_INTERVAL = 50  # Save checkpoint every N images
PROCESSED_DATA_PATH = "data_processing/data/processed/filtered_vqa_with_links.json"

DEFAULT_PROMPT = "What do you see in this image?"
DEFAULT_PREFIX = "answer:"

def load_image_from_url(url: str, timeout: int = 10) -> Image.Image:
    """Load image from URL with error handling."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"  ⚠️  Error loading {url}: {e}")
        return None

def load_dataset(num_images: int) -> List[Dict[str, Any]]:
    """Load image dataset from VQA JSON or use fallback URLs."""
    
    # Try to load from VQA dataset
    if os.path.exists(PROCESSED_DATA_PATH):
        print(f"✅ Loading dataset from {PROCESSED_DATA_PATH}")
        with open(PROCESSED_DATA_PATH, "r") as f:
            data = json.load(f)
        print(f"✅ Found {len(data)} samples in dataset")
        return data[:num_images]
    
    # Fallback: Use actual valid COCO val2017 image IDs
    print(f"⚠️  VQA dataset not found at {PROCESSED_DATA_PATH}")
    print(f"   Using fallback: COCO val2017 images")
    
    # Real COCO val2017 image IDs
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
        # Cycle through valid IDs
        image_id = base_valid_ids[i % len(base_valid_ids)]
        image_id_str = str(image_id).zfill(12)
        
        fallback_data.append({
            "image_url": f"http://images.cocodataset.org/val2017/{image_id_str}.jpg",
            "question_text": DEFAULT_PROMPT,
            "answer": "unknown",
            "category": "general"
        })
    
    return fallback_data

def serialize_result(output, image_url: str, index: int, question: str, layer_evolution: Dict = None) -> Dict[str, Any]:
    """Convert model output to serializable dictionary."""
    result = {
        "index": index,
        "image_url": image_url,
        "question": question,
        "predicted_answer": output.predicted_answer,
        "token_confidence": float(output.token_confidence),
        "head_delta": float(output.head_delta),
        "attention_map_shape": list(output.attention_map.shape),
        "attention_map_mean": float(np.mean(output.attention_map)),
        "attention_map_std": float(np.std(output.attention_map)),
        "num_generated_tokens": len(output.predicted_token_ids),
        "token_strings": output.token_strings,
        "token_ids": [int(tid) for tid in output.predicted_token_ids],
        "attention_map": output.attention_map.tolist() if output.attention_map.size < 100000 else None,
    }
    
    # Add layer evolution metrics
    if layer_evolution and "summary" in layer_evolution:
        result["layer_evolution"] = {
            "most_diffuse_layer": int(layer_evolution["summary"]["most_diffuse_layer"]),
            "most_focused_layer": int(layer_evolution["summary"]["most_focused_layer"]),
            "highest_shift_layer": int(layer_evolution["summary"]["highest_shift_layer"]),
            "most_diverse_heads_layer": int(layer_evolution["summary"]["most_diverse_heads_layer"]),
            "total_entropy_change": float(layer_evolution["summary"]["total_entropy_change"]),
            "max_single_shift": float(layer_evolution["summary"]["max_single_shift"]),
        }
        
        # Per-layer metrics
        result["per_layer_metrics"] = [
            {
                "layer": int(m["layer_index"]),
                "entropy": float(m["entropy"]),
                "entropy_shift": float(m["entropy_shift"]),
                "sparsity": float(m["sparsity"]),
                "sparsity_shift": float(m["sparsity_shift"]),
                "kl_divergence": float(m["kl_divergence"]),
                "head_diversity": float(m["head_diversity"]),
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
    # Load dataset
    print(f"\n📦 Loading dataset ({NUM_IMAGES} images)...")
    dataset = load_dataset(NUM_IMAGES)
    actual_num = min(NUM_IMAGES, len(dataset))
    print(f"✅ Will process {actual_num} images\n")
    
    # Initialize PaliGemma 2 model
    print("🔧 Loading PaliGemma 2 model...")
    from analysis.paligemma_runner import PaliGemmaRunner
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = PaliGemmaRunner(
        model_id="google/paligemma2-3b-pt-224",  # PaliGemma 2 3B model
        device=device,
        quantization=None  # Set to "4bit" or "8bit" for lower memory usage
    )
    print(f"✅ Model loaded on {device}\n")
    
    # Process images
    results = []
    successful = 0
    failed = 0
    
    print("🚀 Starting batch processing...\n")
    start_time = datetime.now()
    
    for idx, sample in enumerate(dataset[:actual_num], start=1):
        image_url = sample.get("image_url", "")
        question = sample.get("question_text", DEFAULT_PROMPT)
        
        print(f"[{idx}/{actual_num}] Processing: {image_url[:60]}...")
        
        # Load image
        image = load_image_from_url(image_url)
        if image is None:
            failed += 1
            print(f"  ❌ Skipped (failed to load)\n")
            continue
        
        try:
            # Run inference
            output = runner.run(
                image=image,
                prompt=question,
                prefix=DEFAULT_PREFIX,
            )
            
            # Serialize and save result
            result = serialize_result(output, image_url, idx, question, output.layer_evolution)
            results.append(result)
            successful += 1
            
            print(f"  ✅ Success | Answer: '{output.predicted_answer[:50]}...' | Confidence: {output.token_confidence:.4f}\n")
            
        except Exception as e:
            failed += 1
            print(f"  ❌ Error during inference: {e}\n")
            continue
        
        # Save checkpoint periodically
        if idx % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(results, idx)
    
    # Save final results
    print("="*70)
    print("💾 Saving final results...")
    
    output_data = {
        "metadata": {
            "total_images": actual_num,
            "successful": successful,
            "failed": failed,
            "timestamp": datetime.now().isoformat(),
            "device": device,
            "model": "google/paligemma2-3b-pt-224",
            "prompt": DEFAULT_PROMPT,
            "prefix": DEFAULT_PREFIX,
        },
        "results": results
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output_data, f, indent=2)
    
    elapsed = datetime.now() - start_time
    
    print("="*70)
    print("✅ ALL RESULTS SAVED TO GOOGLE DRIVE")
    print("="*70)
    print(f"📁 Run folder: {RUN_FOLDER}")
    print(f"📄 Final results: layer_evolution_results.json")
    print(f"📦 Checkpoints folder: checkpoints/")
    print(f"   (Contains checkpoint_50.json, checkpoint_100.json, etc.)")
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total processed: {successful}/{actual_num}")
    print(f"Failed: {failed}")
    print(f"Success rate: {100*successful/actual_num:.1f}%")
    print(f"Time elapsed: {elapsed}")
    print(f"Average time per image: {elapsed.total_seconds()/successful:.2f}s")
    if IS_COLAB:
        print(f"\n🔗 Access your results in Google Drive:")
        print(f"   MyDrive/paligemma_probe_runs/run_{timestamp}/")
    print("="*70)
    
    return results

if __name__ == "__main__":
    results = main()

