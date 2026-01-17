#!/bin/bash
cd ~/LLaVAProbe/topk

echo Starting experiments at 01/14/2026 22:38:04
echo ==========================================

# Top-K experiments for PaliGemma
for k in 1 5 10 25; do
    echo "
    echo [01/14/2026 22:38:04] Running PaliGemma Top-K k=...
    python run_topk_multimodel.py --model paligemma --top_k  --n_samples 1000
done

# Top-K experiments for Qwen2-VL
for k in 1 5 10 25; do
    echo "
    echo [01/14/2026 22:38:04] Running Qwen2-VL Top-K k=...
    python run_topk_multimodel.py --model qwen2-vl --top_k  --n_samples 1000
done

# Causal Intervention experiments
echo "
echo [01/14/2026 22:38:04] Running PaliGemma Causal Intervention...
python run_causal_intervention_multimodel.py --model paligemma --n_samples 1000

echo "
echo [01/14/2026 22:38:04] Running Qwen2-VL Causal Intervention...
python run_causal_intervention_multimodel.py --model qwen2-vl --n_samples 1000

echo "
echo ==========================================
echo All experiments completed at 01/14/2026 22:38:04
