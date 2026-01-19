# VLM Probe: Comparative Attention Analysis for Qwen3-VL and PaliGemma2

This module provides tools for analyzing and comparing visual attention patterns in Vision-Language Models (VLMs), specifically **Qwen2.5-VL / Qwen3-VL** and **PaliGemma2**.

Optimized for **Google Colab with A100 GPU** (80GB VRAM, 160GB DRAM).

## 🚀 Quick Start for Google Colab

### Step 1: Setup Environment

```python
# Clone the repository
!git clone https://github.com/itsloganmann/LLaVAProbe.git
%cd LLaVAProbe

# Run setup script
!bash setup_colab.sh
```

### Step 2: Authenticate for PaliGemma2 (Gated Model)

PaliGemma2 requires HuggingFace gated access:

```python
# Login to HuggingFace
from huggingface_hub import login
login()  # Enter your access token

# Or use the CLI
!huggingface-cli login
```

> **Note**: Accept the license at https://huggingface.co/google/paligemma2-3b-pt-224 first.

### Step 3: Run Analysis

```python
# Option A: Use the command line
!python run_analysis.py --num_samples 100 --output_dir ./outputs

# Option B: Use Python API
from vlm_probe import QwenVLRunner, PaliGemmaRunner, VLMAnalyzer
from PIL import Image
import requests
from io import BytesIO

# Initialize models
qwen = QwenVLRunner(model_id="Qwen/Qwen2.5-VL-7B-Instruct")
paligemma = PaliGemmaRunner(model_id="google/paligemma2-3b-pt-224")

# Load an image
url = "http://images.cocodataset.org/val2017/000000219578.jpg"
image = Image.open(BytesIO(requests.get(url).content))

# Run inference
qwen_output = qwen.run(image, "What color is the dog?")
paligemma_output = paligemma.run(image, "What color is the dog?")

print(f"Qwen: {qwen_output.predicted_answer}")
print(f"PaliGemma: {paligemma_output.predicted_answer}")
```

## 📊 Features

### 1. Attention Pattern Extraction
- Per-layer and per-head attention maps
- Aggregated attention visualization
- Attention entropy analysis

### 2. Cross-Model Comparison
- Attention correlation between models
- Answer agreement analysis
- Confidence calibration comparison

### 3. Logit Lens Analysis
- Layer-wise prediction evolution
- Information flow visualization
- Visual contribution analysis

### 4. Novel Insights for Publication
- Automated insight extraction
- Statistical significance testing
- Publication-ready figures

## 🔬 Generating ICLR 2026-Quality Insights

The analysis framework automatically extracts novel insights:

```python
from vlm_probe import run_comparative_analysis

# Prepare samples
samples = [
    {"image": "path/to/image.jpg", "question": "What is this?", "ground_truth": "dog"},
    # ... more samples
]

# Run analysis
results = run_comparative_analysis(
    samples=samples,
    qwen_model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    paligemma_model_id="google/paligemma2-3b-pt-224",
    output_dir="./analysis_outputs"
)

# Insights are automatically extracted
for insight in results['insights']:
    print(f"📊 {insight['title']}")
    print(f"   {insight['description']}")
```

### Key Research Questions Addressed

1. **Attention Distribution**: How do different VLM architectures distribute attention across image regions?

2. **Layer Emergence**: At which layer does visual information significantly influence predictions?

3. **Spatial Coherence**: Do models focus on coherent object regions or distributed features?

4. **Convergent Reasoning**: Do different attention patterns lead to the same answers?

5. **Confidence Calibration**: How well-calibrated are confidence estimates across architectures?

## 📁 Output Structure

```
outputs/
├── analysis_results.json      # Full analysis data
├── entropy_analysis.png       # Attention entropy comparison
├── comparison_*.png           # Per-sample cross-model comparisons
├── qwen_layers_*.png         # Qwen layer attribution
├── paligemma_layers_*.png    # PaliGemma layer attribution
└── publication_figure.png    # Publication-ready summary figure
```

## 🛠️ Model Options

### Qwen-VL Models
| Model ID | Parameters | Notes |
|----------|------------|-------|
| `Qwen/Qwen2.5-VL-7B-Instruct` | 7B | Recommended for A100 |
| `Qwen/Qwen2.5-VL-72B-Instruct` | 72B | Requires multi-GPU |

### PaliGemma2 Models
| Model ID | Parameters | Resolution | Notes |
|----------|------------|------------|-------|
| `google/paligemma2-3b-pt-224` | 3B | 224px | Fast, good for testing |
| `google/paligemma2-3b-pt-448` | 3B | 448px | Higher resolution |
| `google/paligemma2-10b-pt-224` | 10B | 224px | Better quality |
| `google/paligemma2-28b-pt-224` | 28B | 224px | Best quality |

## 🎨 Visualization Examples

### Attention Heatmap
```python
from vlm_probe import plot_attention_heatmap

plot_attention_heatmap(
    image=image,
    attention_map=qwen_output.aggregated_attention,
    title="Qwen-VL Attention",
    save_path="attention.png"
)
```

### Cross-Model Comparison
```python
from vlm_probe import plot_cross_model_comparison

plot_cross_model_comparison(
    image=image,
    qwen_attention=qwen_output.aggregated_attention,
    paligemma_attention=paligemma_output.aggregated_attention,
    qwen_answer=qwen_output.predicted_answer,
    paligemma_answer=paligemma_output.predicted_answer,
    question="What color is the dog?",
    save_path="comparison.png"
)
```

## 📈 Expected Results

Based on preliminary analysis, expect to find:

1. **Entropy Differences**: Qwen-VL typically shows higher attention entropy (more distributed), while PaliGemma2 shows more focused attention patterns.

2. **Layer Emergence**: PaliGemma2 (with SigLIP) tends to process visual features in earlier layers compared to Qwen-VL.

3. **Attention Correlation**: Despite different architectures, attention correlation is often moderate (0.3-0.5), suggesting partially shared visual reasoning.

4. **Confidence Gaps**: The models show different confidence calibration profiles, with implications for uncertainty quantification.

## 📝 Citation

If you use this code for research, please cite:

```bibtex
@misc{vlmprobe2024,
  title={VLM Probe: Comparative Attention Analysis for Vision-Language Models},
  author={Your Name},
  year={2024},
  url={https://github.com/itsloganmann/LLaVAProbe}
}
```

## 🐛 Troubleshooting

### PaliGemma Access Denied
```
Make sure you:
1. Accepted the license at huggingface.co/google/paligemma2-3b-pt-224
2. Logged in with: huggingface-cli login
```

### Out of Memory
```python
# Use smaller models
qwen = QwenVLRunner(model_id="Qwen/Qwen2.5-VL-7B-Instruct")
paligemma = PaliGemmaRunner(model_id="google/paligemma2-3b-pt-224")

# Or enable memory optimization
import torch
torch.cuda.empty_cache()
```

### Flash Attention Errors
```bash
# Disable flash attention
qwen = QwenVLRunner(use_flash_attention=False)
```

## 📄 License

MIT License - See LICENSE file for details.
