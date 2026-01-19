#!/bin/bash
# Setup script for VLM Probe on Google Colab with A100 GPU
# Run this script first to install all dependencies

echo "=========================================="
echo "VLM Probe Setup for Google Colab (A100)"
echo "=========================================="

# Check if running in Colab
if [ -n "$COLAB_GPU" ]; then
    echo "✓ Running in Google Colab environment"
else
    echo "Note: Not running in Colab. Proceeding anyway..."
fi

# Check GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "Installing dependencies..."
echo ""

# Core dependencies
pip install --upgrade pip

# PyTorch (should be pre-installed in Colab, but ensure latest)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Transformers with latest VLM support
pip install transformers>=4.45.0 accelerate>=0.26.0

# For PaliGemma and Qwen-VL
pip install qwen-vl-utils

# Visualization and analysis
pip install matplotlib>=3.7.0 seaborn>=0.12.0
pip install scipy>=1.11.0 scikit-learn>=1.3.0
pip install opencv-python-headless>=4.8.0

# Image processing
pip install Pillow>=10.0.0

# Optional: Flash Attention 2 for faster inference (A100 compatible)
pip install flash-attn --no-build-isolation

# HuggingFace Hub for model downloads
pip install huggingface_hub>=0.20.0

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="

echo ""
echo "IMPORTANT: For PaliGemma2 access, you need to:"
echo "1. Accept the model license at: https://huggingface.co/google/paligemma2-3b-pt-224"
echo "2. Run: huggingface-cli login"
echo "3. Enter your HuggingFace access token"
echo ""

echo "To verify setup, run:"
echo "  python run_analysis.py --check_only"
echo ""
echo "To run analysis:"
echo "  python run_analysis.py --num_samples 100"
