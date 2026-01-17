#!/bin/bash
cd ~/LLaVAProbe/topk

echo "=========================================="
echo "OVERNIGHT EXPERIMENTS - Starting at $(date)"
echo "=========================================="

# 1. PaliGemma Causal Intervention
echo ""
echo "[$(date)] Running PaliGemma Causal Intervention (1000 samples)..."
python run_causal_intervention_multimodel.py --model paligemma --n_samples 1000

# 2. Top-K experiments for PaliGemma (k=1,5,10,25)
for k in 1 5 10 25; do
    echo ""
    echo "[$(date)] Running PaliGemma Top-K k=$k (1000 samples)..."
    python run_topk_multimodel.py --model paligemma --top_k $k --n_samples 1000
done

# 3. Top-K experiments for Qwen2-VL (k=1,5,10,25)
for k in 1 5 10 25; do
    echo ""
    echo "[$(date)] Running Qwen2-VL Top-K k=$k (1000 samples)..."
    python run_topk_multimodel.py --model qwen2-vl --top_k $k --n_samples 1000
done

echo ""
echo "=========================================="
echo "ALL OVERNIGHT EXPERIMENTS COMPLETED at $(date)"
echo "=========================================="
