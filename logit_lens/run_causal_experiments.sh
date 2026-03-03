#!/bin/bash
# Run comprehensive causal intervention experiments on Lambda GPU instance
# This validates whether identified neurons are causally involved in reliability

echo "========================================"
echo "CAUSAL NEURON INTERVENTION EXPERIMENTS"
echo "========================================"
echo ""

# Activate environment
source ~/.bashrc
conda activate llava || source venv/bin/activate || echo "Using base environment"

# Install any missing dependencies
pip install -q tqdm pandas matplotlib

# Navigate to project
cd ~/LLaVAProbe/logit_lens

echo ""
echo "Running causal intervention experiments..."
echo "This will test:"
echo "  1. Zero ablation of success neurons (expect accuracy DROP)"
echo "  2. Zero ablation of failure neurons (expect accuracy INCREASE)"
echo "  3. Mean ablation (more interpretable baseline)"
echo "  4. Activation patching (transplant from correct → incorrect)"
echo "  5. Random neuron control (expect minimal effect)"
echo ""

# Run the main experiment
python causal_neuron_intervention.py 2>&1 | tee causal_intervention_log.txt

echo ""
echo "========================================"
echo "Experiments complete!"
echo "Results saved to: causal_intervention_results.json"
echo "Log saved to: causal_intervention_log.txt"
echo "========================================"
