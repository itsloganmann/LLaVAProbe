#!/bin/bash
# Setup and run reliability AUROC analysis on Lambda

echo "======================================"
echo "Setting up environment on Lambda..."
echo "======================================"

# Install required packages
pip install transformers accelerate pillow requests scikit-learn tqdm --quiet

# For Qwen2-VL
pip install qwen-vl-utils --quiet

echo ""
echo "======================================"
echo "Running reliability AUROC analysis..."
echo "======================================"

cd logit_lens
python run_reliability_auroc_multimodel.py

echo ""
echo "======================================"
echo "DONE! Check logit_lens/reliability_auroc_multimodel.json"
echo "======================================"
