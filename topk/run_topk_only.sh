#!/bin/bash
cd ~/LLaVAProbe/topk

echo "Starting Top-K experiments at $(date)"
echo "=========================================="

# Top-K experiments for PaliGemma
for k in 1 5 10 25; do
    echo ""
    echo "[$(date)] Running PaliGemma Top-K k=$k..."
    python run_topk_multimodel.py --model paligemma --top_k $k --n_samples 1000
done

# Top-K experiments for Qwen2-VL
for k in 1 5 10 25; do
    echo ""
    echo "[$(date)] Running Qwen2-VL Top-K k=$k..."
    python run_topk_multimodel.py --model qwen2-vl --top_k $k --n_samples 1000
done

echo ""
echo "=========================================="
echo "All Top-K experiments completed at $(date)"
