# Summary of Changes - Batch Layer Evolution Analysis

## Overview
Modified `test_layer_evolution.py` to process 1000 images and save layer evolution results to a JSON file. The solution is designed to be easily runnable in Google Colab after cloning the repo and installing dependencies.

## Files Created/Modified

### 1. **test_layer_evolution.py** (MODIFIED)
- **Purpose**: Main script that processes 1000 images through LLaVA model
- **Key Features**:
  - Configurable number of images (default: 1000)
  - Automatic checkpoint saving every 50 images
  - Loads images from VQA dataset or falls back to COCO URLs
  - Saves comprehensive layer evolution metrics to JSON
  - Progress tracking and error handling
  - Memory-efficient processing with optional quantization

### 2. **setup_colab.sh** (NEW)
- **Purpose**: One-command setup script for Google Colab
- **What it does**:
  - Installs all dependencies from requirements.txt
  - Creates necessary directories
  - Displays usage instructions

### 3. **Run_Layer_Evolution_Colab.ipynb** (NEW)
- **Purpose**: Interactive Jupyter notebook for Colab
- **Sections**:
  1. Repository cloning
  2. Dependency installation
  3. GPU checking
  4. Running the analysis
  5. Loading and inspecting results
  6. Statistical analysis
  7. Visualization
  8. Downloading results
- **User-friendly**: Step-by-step with explanations

### 4. **LAYER_EVOLUTION_README.md** (NEW)
- **Purpose**: Comprehensive documentation
- **Contents**:
  - Quick start guide (both notebook and script)
  - Configuration options
  - Output format specification
  - Key metrics explained
  - Data sources
  - Checkpointing details
  - Memory optimization tips
  - Time estimates
  - Troubleshooting guide
  - Analysis examples

### 5. **test_quick.py** (NEW)
- **Purpose**: Quick validation script
- **What it does**:
  - Runs analysis on just 5 images
  - Verifies everything works before full run
  - Automatically restores original configuration
  - Saves test results to separate file

### 6. **README.md** (UPDATED)
- Added section about batch processing
- Links to new documentation
- Usage examples for Colab

## How to Use in Google Colab

### Method 1: Using the Notebook (Recommended)
```python
# Upload Run_Layer_Evolution_Colab.ipynb to Colab and follow step-by-step
```

### Method 2: Direct Script Execution
```bash
# Clone repo
!git clone https://github.com/YOUR_USERNAME/LLaVAProbe-1.git
%cd LLaVAProbe-1

# Setup
!bash setup_colab.sh

# Quick test (5 images)
!python test_quick.py

# Full analysis (1000 images)
!python test_layer_evolution.py
```

## Output Format

The script generates `layer_evolution_results.json` with:

```json
{
  "metadata": {
    "total_images": 1000,
    "successful": 980,
    "failed": 20,
    "timestamp": "ISO-8601 timestamp",
    "device": "cuda/cpu",
    "model": "llava-hf/llava-1.5-7b-hf",
    "prompt": "default prompt",
    "prefix": "default prefix"
  },
  "results": [
    {
      "index": 1,
      "image_url": "...",
      "question": "...",
      "predicted_answer": "...",
      "token_confidence": 0.85,
      "head_delta": 0.000123,
      "attention_map_shape": [24, 24],
      "attention_map_mean": 0.042,
      "attention_map_std": 0.032,
      "num_generated_tokens": 5,
      "token_strings": ["token1", "token2", ...],
      "token_ids": [123, 456, ...],
      "attention_map": [[...], [...], ...]
    }
  ]
}
```

## Key Metrics Captured

1. **predicted_answer**: Model's text response
2. **token_confidence**: Confidence score for generated tokens
3. **head_delta**: Layer evolution metric (attention pattern changes)
4. **attention_map**: Full spatial attention data
5. **attention_map_mean/std**: Statistical properties
6. **num_generated_tokens**: Response length
7. **token_strings**: Individual tokens
8. **token_ids**: Token IDs for further analysis

## Features

### Automatic Checkpointing
- Saves progress every 50 images (configurable)
- Prevents data loss from interruptions
- Can resume from checkpoints if needed

### Flexible Data Sources
1. **Primary**: VQA dataset at `data_processing/data/processed/filtered_vqa_with_links.json`
2. **Fallback**: Automatically generates COCO image URLs

### Error Handling
- Skips failed image loads
- Continues processing on inference errors
- Logs all errors for debugging
- Reports success/failure statistics

### Memory Optimization
- Optional quantization (4-bit, 8-bit)
- Configurable to work on different GPU sizes
- Attention maps saved conditionally (for large maps)

### Progress Tracking
- Real-time console output
- Progress counter (N/1000)
- Per-image timing information
- Summary statistics at end

## Performance

### Time Estimates
- **T4 GPU (Colab Free)**: ~3-5 seconds/image → 1.5 hours total
- **V100 GPU (Colab Pro)**: ~1-2 seconds/image → 30-45 minutes total
- **CPU**: Not recommended (10+ hours)

### Memory Usage
- Default (no quantization): ~8-10GB GPU RAM
- 4-bit quantization: ~4-5GB GPU RAM
- 8-bit quantization: ~6-7GB GPU RAM

## Configuration Options

Easy to customize at the top of `test_layer_evolution.py`:

```python
NUM_IMAGES = 1000          # Change to process fewer/more images
OUTPUT_FILE = "..."        # Change output filename
CHECKPOINT_INTERVAL = 50   # Checkpoint frequency
DEFAULT_PROMPT = "..."     # Customize prompt
DEFAULT_PREFIX = "..."     # Customize prefix
```

## Testing

Before running on 1000 images, test with:
```bash
python test_quick.py
```
This runs on just 5 images to verify everything works.

## Requirements

All handled by `setup_colab.sh`:
- transformers
- torch
- PIL
- requests
- numpy
- pandas (for analysis)
- matplotlib (for visualization)

## Summary

This solution provides:
✅ Easy one-command setup in Colab
✅ Robust batch processing of 1000 images
✅ Comprehensive layer evolution metrics
✅ Automatic checkpointing
✅ Error handling and recovery
✅ Multiple interfaces (script + notebook)
✅ Extensive documentation
✅ Quick testing capability
✅ Memory optimization options
✅ Visualization tools
✅ Easy result download

The entire workflow is designed to be run in Google Colab with minimal setup, making it accessible and reproducible.

