#!/bin/bash
cd ~/LLaVAProbe/topk

echo "Starting PaliGemma Causal Intervention at $(date)"
python run_causal_intervention_multimodel.py --model paligemma --n_samples 1000
echo "PaliGemma Causal Intervention completed at $(date)"
