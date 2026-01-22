#!/bin/bash
# Setup script for VLM Probe on Google Colab / Lambda A100 GPU
# Run this script first to install all dependencies

echo "=========================================="
echo "VLM Probe Setup for A100 GPU"
echo "=========================================="

# Check if running in Colab
if [ -n "$COLAB_GPU" ]; then
    echo "✓ Running in Google Colab environment"
elif [ -d "/lambda" ]; then
    echo "✓ Running on Lambda Cloud instance"
else
    echo "Note: Unknown environment. Proceeding anyway..."
fi

# Check GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "Installing dependencies..."
echo ""

# Core dependencies
pip install --upgrade pip

# CRITICAL: Pin numpy to 1.x to avoid conflicts with system packages (scipy, sklearn, tensorflow)
# Lambda has system packages compiled against numpy 1.x
pip install "numpy>=1.24,<2"

# PyTorch stack - pinned versions for compatibility
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Transformers with latest VLM support
pip install "transformers>=4.45.0" "accelerate>=0.26.0"

# For PaliGemma and Qwen-VL
pip install qwen-vl-utils

# Visualization and analysis - install user versions to override incompatible system packages
pip install "matplotlib>=3.7.0" "seaborn>=0.12.0"
pip install "scipy>=1.11.0" "scikit-learn>=1.3.0"
pip install "opencv-python>=4.8.0,<4.10"  # 4.10+ requires numpy 2

# Image processing
pip install "Pillow>=10.0.0"

# Optional: Flash Attention 2 for faster inference (A100 compatible)
pip install flash-attn --no-build-isolation

# HuggingFace Hub for model downloads
pip install "huggingface_hub>=0.20.0"

# Jinja2 - need >=3.1.0 for transformers chat templates
pip install "jinja2>=3.1.0"

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
