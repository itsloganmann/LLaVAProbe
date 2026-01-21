"""
PaliGemma Vision Cutoff Ablation - Single Colab Cell
Run this entire script in one Colab cell.

This script:
- Fixes PIL/Pillow compatibility issues
- Clones the repository and checks out the branch
- Downloads 500 images from VQA 2.0
- Runs vision cutoff ablation experiments on layers 1, 2, 4, 8, 16
- Saves results to Google Drive
"""

# ==============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ==============================================================================
TARGET_LAYERS = [1, 2, 4, 8, 16]  # Layers to test for cutoff
MAX_SAMPLES = 500  # Number of images to process
HF_TOKEN = "YOUR_TOKEN_HERE"  # Replace with your HuggingFace token or leave as is
BRANCH = "Idea-#3---Attention-Evolution-Tracking-Paligemma-2"  # Branch to use

# ==============================================================================
# SETUP - Run this first
# ==============================================================================
import os
import sys
import subprocess
import json
import requests
import concurrent.futures
import torch
import gc
import shutil
from io import BytesIO
from datetime import datetime
from google.colab import drive

print("⚙️ SETTING UP ENVIRONMENT...", flush=True)

# CRITICAL: Fix Pillow FIRST before any other imports
print("   🔧 Fixing Pillow compatibility (CRITICAL)...", flush=True)
try:
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "Pillow", "PIL"], 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except:
    pass

# Clear ALL PIL modules from memory
modules_to_clear_pil = [k for k in list(sys.modules.keys()) if 'PIL' in k or 'pil' in k.lower() or 'ImageFont' in k]
for mod in modules_to_clear_pil:
    try:
        del sys.modules[mod]
    except:
        pass

# Reinstall working version
subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow==10.1.0", "--quiet", "--force-reinstall", "--no-cache-dir", "--break-system-packages"])

# Verify it works
try:
    from PIL import Image
    test_img = Image.new('RGB', (1, 1))
    print("   ✅ Pillow fixed and verified!", flush=True)
except Exception as e:
    print(f"   ⚠️  First attempt failed: {e}, trying Pillow 10.0.0...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow==10.0.0", "--quiet", "--force-reinstall", "--no-cache-dir", "--break-system-packages"])
    from PIL import Image
    print("   ✅ Pillow fixed with alternative version!", flush=True)

# Fix numpy compatibility
print("   🔧 Fixing numpy compatibility...", flush=True)
try:
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "numpy"], 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except:
    pass

modules_to_clear = [k for k in list(sys.modules.keys()) if 'numpy' in k.lower() or 'scipy' in k.lower() or 'pandas' in k.lower()]
for mod in modules_to_clear:
    try:
        del sys.modules[mod]
    except:
        pass

subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.26.4", "--quiet", "--force-reinstall", "--break-system-packages"])
import numpy as np
print(f"   ✅ numpy {np.__version__} installed", flush=True)

# Install pandas and tqdm
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "tqdm", "--quiet", "--break-system-packages"])
import pandas as pd
from tqdm.notebook import tqdm
print(f"   ✅ pandas and tqdm installed", flush=True)

# ==============================================================================
# CLONE REPOSITORY
# ==============================================================================
print("\n📥 CLONING REPOSITORY...", flush=True)
REPO_URL = "https://github.com/itsloganmann/LLaVAProbe.git"
REPO_DIR = "/content/LLaVAProbe"
OUTPUT_BASE_DIR = "/content/drive/MyDrive/paligemma_vision_cutoff_runs"

os.chdir("/content")

# Remove old repo if it exists
if os.path.exists(REPO_DIR):
    print(f"   🗑️  Removing existing repo...", flush=True)
    try:
        shutil.rmtree(REPO_DIR)
    except Exception as e:
        print(f"   ⚠️  Could not remove old repo: {e}", flush=True)

# Clone fresh
print(f"   ⬇️  Cloning {REPO_URL}...", flush=True)
result = subprocess.run(["git", "clone", REPO_URL, REPO_DIR], 
                       capture_output=True, text=True)
if result.returncode != 0:
    print(f"   ❌ Git clone failed: {result.stderr}", flush=True)
    raise RuntimeError(f"Failed to clone repo: {result.stderr}")

# Checkout branch
print(f"   🔀 Checking out branch {BRANCH}...", flush=True)
result = subprocess.run(["git", "-C", REPO_DIR, "checkout", BRANCH],
                       capture_output=True, text=True)
if result.returncode != 0:
    print(f"   ⚠️  Branch checkout failed: {result.stderr}", flush=True)
    print(f"   Trying to fetch and checkout...", flush=True)
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "origin"], 
                   capture_output=True)
    result = subprocess.run(["git", "-C", REPO_DIR, "checkout", BRANCH],
                           capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to checkout branch {BRANCH}: {result.stderr}")

# Add to path and change directory
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

print(f"   ✅ Repo cloned and checked out to {BRANCH}", flush=True)

# Install dependencies
print("   📦 Installing dependencies...", flush=True)
os.system("apt-get install -y aria2 > /dev/null 2>&1")

# Ensure Pillow is fixed before installing other packages
print("   🔧 Ensuring Pillow is fixed...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow==10.1.0", "--quiet", "--force-reinstall", "--no-cache-dir", "--break-system-packages"])

subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==1.3.2", "--quiet", "--force-reinstall", "--break-system-packages"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt", "--break-system-packages"])

# Re-fix Pillow after requirements (in case something downgraded it)
subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow==10.1.0", "--quiet", "--force-reinstall", "--no-cache-dir", "--break-system-packages"])

# Final numpy pin
subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.26.4", "--quiet", "--force-reinstall", "--no-deps", "--break-system-packages"])

# Mount Drive
try:
    drive.mount('/content/drive', force_remount=False)
except:
    pass

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
os.makedirs(os.path.join(REPO_DIR, "data/vqa/images"), exist_ok=True)

# HuggingFace Login
if HF_TOKEN and HF_TOKEN != "YOUR_TOKEN_HERE":
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN)
        print("   ✅ Logged into HuggingFace", flush=True)
    except:
        pass

# ==============================================================================
# DATA PREPARATION
# ==============================================================================
print(f"\n📦 PREPARING DATA ({MAX_SAMPLES} images)...", flush=True)

# COCO val2017 image IDs (validated to work)
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

DEFAULT_PROMPT = "What do you see in this image?"
DEFAULT_PREFIX = "This image shows"

target_data = []
for i in range(MAX_SAMPLES):
    image_id = base_valid_ids[i % len(base_valid_ids)]
    target_data.append(image_id)

unique_imgs = list(set(target_data))

def download_img(img_id):
    fname = f"COCO_val2017_{str(img_id).zfill(12)}.jpg"
    img_dir = os.path.join(REPO_DIR, "data", "vqa", "images")
    os.makedirs(img_dir, exist_ok=True)
    path = os.path.join(img_dir, fname)
    
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return (img_id, path)
    try:
        r = requests.get(f"http://images.cocodataset.org/val2017/{fname}", timeout=10)
        if r.status_code == 200 and len(r.content) > 0:
            with open(path, 'wb') as f:
                f.write(r.content)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return (img_id, path)
    except:
        pass
    return (img_id, None)

print(f"   ⬇️ Downloading {len(unique_imgs)} images...", flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    downloaded_results = list(tqdm(executor.map(download_img, unique_imgs), total=len(unique_imgs), desc="Downloading"))

data_records = []
img_id_to_path = {}
for img_id, path in downloaded_results:
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        img_id_to_path[img_id] = path

for i, img_id in enumerate(target_data):
    if img_id in img_id_to_path:
        data_records.append({
            "image_url": img_id_to_path[img_id],
            "question_text": DEFAULT_PROMPT,
            "ground_truth": "unknown",
            "original_id": img_id
        })

print(f"   ✅ Ready to process {len(data_records)} valid samples.", flush=True)

# ==============================================================================
# ABLATION CONFIGURATIONS
# ==============================================================================
ABLATION_CONFIGS = [
    {"mode": "baseline", "vision_cutoff_mode": None, "vision_cutoff_layer": None},
]

# Add early_cut and late_only for each target layer
for layer in TARGET_LAYERS:
    ABLATION_CONFIGS.append({
        "mode": f"early_cut_{layer}", 
        "vision_cutoff_mode": "early_cut", 
        "vision_cutoff_layer": layer
    })
    ABLATION_CONFIGS.append({
        "mode": f"late_only_{layer}", 
        "vision_cutoff_mode": "late_only", 
        "vision_cutoff_layer": layer
    })

print(f"\n🚀 ABLATION CONFIGURATIONS")
print(f"   Testing {len(ABLATION_CONFIGS)} configurations:")
for config in ABLATION_CONFIGS:
    print(f"     - {config['mode']}")
print(f"   Target layers: {TARGET_LAYERS}", flush=True)

# ==============================================================================
# LOAD MODEL
# ==============================================================================
print(f"\n🔧 LOADING PALIGEMMA MODEL...", flush=True)

# Double-check PIL is working before importing anything that might use it
try:
    from PIL import Image
    _ = Image.new('RGB', (1, 1))
except Exception as e:
    print(f"   ⚠️  PIL check failed: {e}, reinstalling...", flush=True)
    modules_to_clear_pil = [k for k in list(sys.modules.keys()) if 'PIL' in k or 'pil' in k.lower()]
    for mod in modules_to_clear_pil:
        try:
            del sys.modules[mod]
        except:
            pass
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow==10.1.0", "--quiet", "--force-reinstall", "--no-cache-dir", "--break-system-packages"])
    from PIL import Image
    print(f"   ✅ PIL reinstalled and verified", flush=True)

# Clear analysis modules (but keep PIL working)
modules_to_remove = [k for k in sys.modules.keys() if ('paligemma' in k.lower() or 'analysis' in k.lower()) and 'PIL' not in k and 'pil' not in k.lower()]
for mod in modules_to_remove:
    if mod in sys.modules:
        try:
            del sys.modules[mod]
        except:
            pass

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

print("   📦 Importing PaliGemmaRunner...", flush=True)
from analysis.paligemma_runner import PaliGemmaRunner
print("   ✅ PaliGemmaRunner imported successfully", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Using device: {device}", flush=True)

quantization = None
if device == "cuda":
    try:
        import bitsandbytes  # noqa: F401
        quantization = "4bit"
        print("   Using 4-bit quantization", flush=True)
    except:
        print("   Using full precision", flush=True)

runner = PaliGemmaRunner(
    model_id="google/paligemma-3b-pt-224",
    device=device,
    quantization=quantization
)

print(f"   ✅ Model loaded | Layers: {runner.num_layers}", flush=True)

# ==============================================================================
# RUN ABLATION
# ==============================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_FOLDER = os.path.join(OUTPUT_BASE_DIR, f"run_{timestamp}")
os.makedirs(RUN_FOLDER, exist_ok=True)
CHECKPOINT_DIR = os.path.join(RUN_FOLDER, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(RUN_FOLDER, "vision_cutoff_results.json")
CHECKPOINT_INTERVAL = 50

all_results = []
successful = 0
failed = 0
start_time = datetime.now()

runner.model.eval()

print(f"\n   ▶️ Processing {len(data_records)} images with {len(ABLATION_CONFIGS)} configurations each...", flush=True)
print(f"   Total runs: {len(data_records) * len(ABLATION_CONFIGS)}", flush=True)

for idx, record in tqdm(enumerate(data_records), total=len(data_records), desc="Processing images"):
    try:
        # Load image
        if os.path.exists(record['image_url']):
            img = Image.open(record['image_url']).convert("RGB")
        else:
            img_id = record['original_id']
            img_id_str = str(img_id).zfill(12)
            url = f"http://images.cocodataset.org/val2017/COCO_val2017_{img_id_str}.jpg"
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content)).convert("RGB")
            except:
                failed += 1
                continue
        
        question = record.get("question_text", DEFAULT_PROMPT)
        
        # Run each ablation configuration
        for ablation_config in ABLATION_CONFIGS:
            try:
                # Set vision cutoff if needed
                if ablation_config["vision_cutoff_mode"]:
                    runner.set_vision_cutoff(
                        ablation_config["vision_cutoff_mode"],
                        ablation_config["vision_cutoff_layer"]
                    )
                else:
                    # Disable vision cutoff for baseline
                    runner.model.language_model.vision_cutoff["enabled"] = False
                
                # Run inference
                output = runner.run(
                    image=img,
                    prompt=question,
                    prefix=DEFAULT_PREFIX,
                )
                
                # Store results
                result = {
                    "index": idx,
                    "image_url": record["image_url"],
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
                
                if output.layer_evolution and "summary" in output.layer_evolution:
                    result["layer_evolution"] = {
                        "most_diffuse_layer": int(output.layer_evolution["summary"]["most_diffuse_layer"]),
                        "most_focused_layer": int(output.layer_evolution["summary"]["most_focused_layer"]),
                        "total_entropy_change": float(output.layer_evolution["summary"]["total_entropy_change"]),
                    }
                
                all_results.append(result)
                
            except Exception as e:
                print(f"❌ Error for {ablation_config['mode']} on image {idx}: {e}", flush=True)
                continue
        
        successful += 1
        
        # Save checkpoint
        if (idx + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_file = os.path.join(CHECKPOINT_DIR, f"checkpoint_{idx + 1}.json")
            with open(checkpoint_file, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  💾 Checkpoint saved: checkpoint_{idx + 1}.json", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    except Exception as e:
        print(f"❌ Image Error {idx}: {e}", flush=True)
        failed += 1
        continue

# ==============================================================================
# SAVE RESULTS
# ==============================================================================
print("\n💾 SAVING RESULTS...", flush=True)

output_data = {
    "metadata": {
        "total_images": len(data_records),
        "successful": successful,
        "failed": failed,
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "model": "google/paligemma-3b-pt-224",
        "target_layers": TARGET_LAYERS,
        "ablation_configs": ABLATION_CONFIGS,
        "num_layers": runner.num_layers,
        "prompt": DEFAULT_PROMPT,
        "prefix": DEFAULT_PREFIX,
    },
    "results": all_results
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(output_data, f, indent=2)

df_results = pd.json_normalize(all_results)
csv_file = os.path.join(RUN_FOLDER, "vision_cutoff_results.csv")
df_results.to_csv(csv_file, index=False)

elapsed = datetime.now() - start_time

print("\n" + "="*70)
print("✅ ALL RESULTS SAVED")
print("="*70)
print(f"📁 Run folder: {RUN_FOLDER}")
print(f"📄 Results: vision_cutoff_results.json")
print(f"📊 CSV: vision_cutoff_results.csv")
print("="*70)
print(f"Total images: {successful}/{len(data_records)}")
print(f"Failed: {failed}")
print(f"Total runs: {len(all_results)}")
print(f"Time: {elapsed}")
if successful > 0:
    print(f"Avg time/image: {elapsed.total_seconds()/successful:.2f}s")
print("="*70)

del runner
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n🎉 COMPLETE!")
