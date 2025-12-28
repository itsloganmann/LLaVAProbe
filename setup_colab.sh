#!/bin/bash
# Setup script for Google Colab
# Run this after cloning the repo to install all dependencies

echo "=========================================="
echo "Setting up LLaVAProbe for Colab"
echo "=========================================="

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data_processing/data/processed
mkdir -p analysis_outputs

echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "To run the layer evolution analysis on 1000 images:"
echo "  python test_layer_evolution.py"
echo ""
echo "Results will be saved to: layer_evolution_results.json"
echo "=========================================="

