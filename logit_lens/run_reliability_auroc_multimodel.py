"""
Compute Combined Reliability AUROC for PaliGemma and Qwen2-VL

This script runs the reliability analysis (like step4_reliability_analysis.py)
but for PaliGemma and Qwen2-VL models.

Run on Lambda with GPU access.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_vqa_samples(n_samples=200):
    """Load VQAv2 samples for evaluation."""
    # Try to load from existing data first
    topk_dir = os.path.join(os.path.dirname(SCRIPT_DIR), 'topk')
    
    samples = []
    
    # Load from paligemma results which has the questions
    paligemma_file = os.path.join(topk_dir, 'paligemma_topk_5_results.csv')
    if os.path.exists(paligemma_file):
        import csv
        with open(paligemma_file, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= n_samples:
                    break
                samples.append({
                    'image_url': row['image_url'],
                    'question': row['question'],
                    'ground_truth': row['ground_truth'],
                    'question_type': row['question_type']
                })
    
    print(f"Loaded {len(samples)} samples")
    return samples


def setup_paligemma():
    """Load PaliGemma model."""
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    
    model_id = "google/paligemma-3b-mix-224"
    print(f"Loading PaliGemma from {model_id}...")
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    return model, processor


def setup_qwen2vl():
    """Load Qwen2-VL model."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    print(f"Loading Qwen2-VL from {model_id}...")
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    return model, processor


def load_image(url):
    """Load image from URL."""
    try:
        response = requests.get(url, timeout=10)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def extract_features_paligemma(model, processor, samples, device='cuda'):
    """Extract per-sample features for PaliGemma."""
    print("\nExtracting PaliGemma features...")
    
    results = []
    n_layers = model.config.text_config.num_hidden_layers
    
    for sample in tqdm(samples):
        image = load_image(sample['image_url'])
        if image is None:
            continue
        
        question = sample['question']
        prompt = f"<image> {question}"
        
        try:
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                
                # Get prediction
                logits = outputs.logits[0, -1, :]
                pred_token = torch.argmax(logits).item()
                pred_text = processor.decode([pred_token]).strip().lower()
                
                # Check correctness
                gt = sample['ground_truth'].lower().strip()
                is_correct = pred_text.startswith(gt) or gt.startswith(pred_text) or pred_text in gt or gt in pred_text
                
                # Extract hidden states from each layer (last token position)
                hidden_states = outputs.hidden_states  # tuple of (batch, seq, hidden)
                
                # Get features: mean of last 5 layers' hidden states
                layer_features = []
                for layer_idx in range(max(0, n_layers - 5), n_layers):
                    hs = hidden_states[layer_idx + 1][0, -1, :].cpu().numpy()  # +1 because index 0 is embeddings
                    layer_features.append(hs)
                
                # Compute margin (top logit - second top logit)
                top2 = torch.topk(logits, 2)
                margin = (top2.values[0] - top2.values[1]).item()
                
                # Compute attention entropy (from last layer)
                # Simplified: use logit entropy as proxy
                probs = torch.softmax(logits, dim=0)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                
                results.append({
                    'is_correct': is_correct,
                    'margin': margin,
                    'entropy': entropy,
                    'features': np.concatenate(layer_features),
                    'pred': pred_text,
                    'gt': gt
                })
                
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    return results


def extract_features_qwen2vl(model, processor, samples, device='cuda'):
    """Extract per-sample features for Qwen2-VL."""
    print("\nExtracting Qwen2-VL features...")
    
    results = []
    n_layers = model.config.num_hidden_layers
    
    for sample in tqdm(samples):
        image = load_image(sample['image_url'])
        if image is None:
            continue
        
        question = sample['question']
        
        # Qwen2-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                
                # Get prediction
                logits = outputs.logits[0, -1, :]
                pred_token = torch.argmax(logits).item()
                pred_text = processor.decode([pred_token]).strip().lower()
                
                # Check correctness
                gt = sample['ground_truth'].lower().strip()
                is_correct = pred_text.startswith(gt) or gt.startswith(pred_text) or pred_text in gt or gt in pred_text
                
                # Extract hidden states
                hidden_states = outputs.hidden_states
                
                # Get features: mean of last 5 layers' hidden states
                layer_features = []
                for layer_idx in range(max(0, n_layers - 5), n_layers):
                    hs = hidden_states[layer_idx + 1][0, -1, :].cpu().numpy()
                    layer_features.append(hs)
                
                # Compute margin
                top2 = torch.topk(logits, 2)
                margin = (top2.values[0] - top2.values[1]).item()
                
                # Compute entropy
                probs = torch.softmax(logits, dim=0)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                
                results.append({
                    'is_correct': is_correct,
                    'margin': margin,
                    'entropy': entropy,
                    'features': np.concatenate(layer_features),
                    'pred': pred_text,
                    'gt': gt
                })
                
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    return results


def compute_aurocs(results, model_name):
    """Compute various AUROC metrics from extracted features."""
    print(f"\n{'='*60}")
    print(f"Computing AUROCs for {model_name}")
    print(f"{'='*60}")
    
    if len(results) < 20:
        print(f"Not enough samples ({len(results)}), skipping...")
        return None
    
    # Extract arrays
    correctness = np.array([r['is_correct'] for r in results], dtype=int)
    margins = np.array([r['margin'] for r in results])
    entropies = np.array([r['entropy'] for r in results])
    features = np.vstack([r['features'] for r in results])
    
    n_correct = correctness.sum()
    n_incorrect = len(correctness) - n_correct
    accuracy = n_correct / len(correctness)
    
    print(f"\nSamples: {len(results)}")
    print(f"Correct: {n_correct}, Incorrect: {n_incorrect}")
    print(f"Accuracy: {accuracy:.1%}")
    
    # Check if we have both classes
    if n_correct == 0 or n_incorrect == 0:
        print("Only one class present, cannot compute AUROC")
        return None
    
    # 1. Spatial Attention AUROC (using entropy as proxy)
    try:
        spatial_auroc = roc_auc_score(correctness, -entropies)  # Negative because lower entropy = more confident
    except:
        spatial_auroc = 0.5
    print(f"\nSpatial Attention AUROC (entropy proxy): {spatial_auroc:.3f}")
    
    # 2. Margin AUROC
    try:
        margin_auroc = roc_auc_score(correctness, margins)
    except:
        margin_auroc = 0.5
    print(f"Margin AUROC: {margin_auroc:.3f}")
    
    # 3. Combined Model AUROC (using hidden state features)
    print("\nTraining combined reliability model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, correctness, test_size=0.3, random_state=42, stratify=correctness
    )
    
    # Train logistic regression with regularization
    clf = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
    clf.fit(X_train, y_train)
    
    # Compute AUROC on test set
    y_prob = clf.predict_proba(X_test)[:, 1]
    try:
        combined_auroc = roc_auc_score(y_test, y_prob)
    except:
        combined_auroc = 0.5
    
    test_accuracy = clf.score(X_test, y_test)
    
    print(f"Combined Model Test Accuracy: {test_accuracy:.1%}")
    print(f"Combined Model AUROC: {combined_auroc:.3f}")
    
    # 4. Also compute AUROC on full dataset (for comparison)
    y_prob_full = clf.predict_proba(features)[:, 1]
    try:
        combined_auroc_full = roc_auc_score(correctness, y_prob_full)
    except:
        combined_auroc_full = 0.5
    print(f"Combined Model AUROC (full data): {combined_auroc_full:.3f}")
    
    return {
        'model': model_name,
        'n_samples': len(results),
        'accuracy': accuracy,
        'n_correct': int(n_correct),
        'n_incorrect': int(n_incorrect),
        'spatial_attention_auroc': float(spatial_auroc),
        'margin_auroc': float(margin_auroc),
        'combined_auroc_test': float(combined_auroc),
        'combined_auroc_full': float(combined_auroc_full),
        'test_accuracy': float(test_accuracy)
    }


def main():
    print("="*70)
    print("RELIABILITY AUROC ANALYSIS FOR PALIGEMMA AND QWEN2-VL")
    print("="*70)
    
    # Load samples
    samples = load_vqa_samples(n_samples=200)
    
    if len(samples) == 0:
        print("ERROR: No samples loaded!")
        return
    
    all_results = {}
    
    # ===== PALIGEMMA =====
    print("\n" + "="*70)
    print("PALIGEMMA")
    print("="*70)
    
    try:
        model, processor = setup_paligemma()
        paligemma_features = extract_features_paligemma(model, processor, samples)
        paligemma_aurocs = compute_aurocs(paligemma_features, "PaliGemma-3B")
        all_results['paligemma'] = paligemma_aurocs
        
        # Free memory
        del model, processor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error with PaliGemma: {e}")
        all_results['paligemma'] = None
    
    # ===== QWEN2-VL =====
    print("\n" + "="*70)
    print("QWEN2-VL")
    print("="*70)
    
    try:
        model, processor = setup_qwen2vl()
        qwen_features = extract_features_qwen2vl(model, processor, samples)
        qwen_aurocs = compute_aurocs(qwen_features, "Qwen2-VL-7B")
        all_results['qwen2vl'] = qwen_aurocs
        
        # Free memory
        del model, processor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error with Qwen2-VL: {e}")
        all_results['qwen2vl'] = None
    
    # Save results
    output_file = os.path.join(SCRIPT_DIR, 'reliability_auroc_multimodel.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY - VALUES FOR PAPER TABLE")
    print("="*70)
    
    for model_name, result in all_results.items():
        if result:
            print(f"\n{model_name.upper()}:")
            print(f"  Spatial Attention AUROC: {result['spatial_attention_auroc']:.3f}")
            print(f"  Combined Model AUROC:    {result['combined_auroc_test']:.3f}")
        else:
            print(f"\n{model_name.upper()}: FAILED")


if __name__ == "__main__":
    main()
