#!/bin/bash
# =============================================================================
# LLaVAProbe Setup Script for Google Cloud GPU Instances
# =============================================================================
# This script sets up the complete environment for running LLaVA analysis
# on Google Cloud Platform with GPU support.
#
# Prerequisites:
# - Google Cloud VM with NVIDIA GPU (T4, V100, or A100 recommended)
# - CUDA 11.8+ drivers installed
# - Ubuntu 20.04 LTS or later
# =============================================================================

set -e  # Exit on error

echo "=========================================="
echo "LLaVAProbe Google Cloud Setup"
echo "=========================================="

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ Warning: No NVIDIA GPU detected. This may cause issues."
    echo "  Make sure you're running on a GPU-enabled instance."
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv sees
source sees/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support first
echo ""
echo "Installing PyTorch with CUDA 11.8 support..."
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
echo ""
echo "Installing remaining dependencies..."
pip install transformers==4.37.1 accelerate==0.26.1 bitsandbytes==0.42.0 \
    huggingface-hub==0.20.3 matplotlib==3.7.4 Pillow==10.2.0 ipykernel==6.29.4 ipython==8.12.3 \
    bertviz==1.4.0 opencv-python==4.10.0.84 requests==2.32.3 numpy==1.26.4 scikit-learn==1.4.2 \
    hdbscan==0.8.33 umap-learn==0.5.6 pypdf==6.1.3

# Copy custom model files
echo ""
echo "Installing custom LLaVA model files..."
TRANSFORMERS_PATH=$(python3 -c "import transformers, os; print(os.path.dirname(transformers.__file__))")

if [ -d "$TRANSFORMERS_PATH" ]; then
    echo "✓ Transformers package found at: $TRANSFORMERS_PATH"
    
    # Backup original files
    if [ -f "$TRANSFORMERS_PATH/models/llava/modeling_llava.py" ]; then
        cp "$TRANSFORMERS_PATH/models/llava/modeling_llava.py" "$TRANSFORMERS_PATH/models/llava/modeling_llava.py.backup"
        echo "  Backed up original modeling_llava.py"
    fi
    if [ -f "$TRANSFORMERS_PATH/models/llama/modeling_llama.py" ]; then
        cp "$TRANSFORMERS_PATH/models/llama/modeling_llama.py" "$TRANSFORMERS_PATH/models/llama/modeling_llama.py.backup"
        echo "  Backed up original modeling_llama.py"
    fi
    
    # Copy custom files
    cp modeling_llava.py "$TRANSFORMERS_PATH/models/llava/modeling_llava.py"
    cp modeling_llama.py "$TRANSFORMERS_PATH/models/llama/modeling_llama.py"
    
    echo "✓ Custom model files installed successfully"
else
    echo "✗ Error: Transformers package not found"
    exit 1
fi

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

python3 << EOF
import torch
import transformers
import accelerate
import bitsandbytes

print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"✓ Transformers version: {transformers.__version__}")
print(f"✓ Accelerate version: {accelerate.__version__}")
print(f"✓ BitsAndBytes available: True")
EOF

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  source sees/bin/activate"
echo ""
echo "To run the analysis pipeline:"
echo "  python analysis/pipeline_runner.py --prompts results.csv --output-dir analysis_outputs"
echo ""
echo "For memory-efficient inference with quantization:"
echo "  python analysis/pipeline_runner.py --prompts results.csv --output-dir analysis_outputs --quantization 4bit"
echo ""

