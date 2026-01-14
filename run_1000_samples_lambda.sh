#!/bin/bash
# Run 1000-sample experiments for PaliGemma and Qwen2-VL on Lambda
# This replaces the 200-sample experiments with full 1000-sample runs

set -e

cd ~/LLaVAProbe

# Ensure HuggingFace login for gated models (PaliGemma)
echo "Logging into HuggingFace..."
# Use environment variable or pass token via command line:
# export HF_TOKEN=your_token_here
# huggingface-cli login --token $HF_TOKEN
echo "Make sure you're logged in to HuggingFace for PaliGemma access"

# Pull latest code
echo "Pulling latest code..."
git pull

# Verify results.csv exists and has 1000+ samples
echo "Checking dataset..."
wc -l results.csv

# =============================================================================
# Top-K Attention Analysis (1000 samples)
# =============================================================================
echo ""
echo "=========================================="
echo "Running Top-K Attention Analysis (1000 samples)"
echo "=========================================="

cd topk

echo ""
echo "--- PaliGemma Top-K (1000 samples) ---"
python run_topk_multimodel.py --model paligemma --top_k 5 --n_samples 1000

echo ""
echo "--- Qwen2-VL Top-K (1000 samples) ---"
python run_topk_multimodel.py --model qwen2-vl --top_k 5 --n_samples 1000

# =============================================================================
# Causal Intervention Analysis (1000 samples)
# =============================================================================
echo ""
echo "=========================================="
echo "Running Causal Intervention Analysis (1000 samples)"
echo "=========================================="

echo ""
echo "--- PaliGemma Causal Intervention (1000 samples) ---"
python run_causal_intervention_multimodel.py --model paligemma --n_samples 1000

echo ""
echo "--- Qwen2-VL Causal Intervention (1000 samples) ---"
python run_causal_intervention_multimodel.py --model qwen2-vl --n_samples 1000

echo ""
echo "=========================================="
echo "All 1000-sample experiments completed!"
echo "=========================================="
echo ""
echo "Results files:"
ls -la *.csv *.json 2>/dev/null || true

cd ..
