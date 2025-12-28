# Layer Evolution Analysis - Batch Processing

This module processes 1000 images through the LLaVA model and saves layer evolution analysis results.

## Quick Start (Google Colab)

### Option 1: Use the Jupyter Notebook

1. Upload `Run_Layer_Evolution_Colab.ipynb` to Google Colab
2. Follow the step-by-step instructions in the notebook
3. Results will be saved and can be downloaded

### Option 2: Run the Python Script Directly

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/LLaVAProbe-1.git
cd LLaVAProbe-1
```

2. **Install dependencies:**
```bash
bash setup_colab.sh
```

3. **Run the analysis:**
```bash
python test_layer_evolution.py
```

## Configuration

You can modify these parameters at the top of `test_layer_evolution.py`:

```python
NUM_IMAGES = 1000                # Number of images to process
OUTPUT_FILE = "layer_evolution_results.json"  # Output file name
CHECKPOINT_INTERVAL = 50         # Save checkpoint every N images
DEFAULT_PROMPT = "What do you see in this image?"
DEFAULT_PREFIX = "This image shows"
```

## Output Format

The script generates a JSON file with the following structure:

```json
{
  "metadata": {
    "total_images": 1000,
    "successful": 980,
    "failed": 20,
    "timestamp": "2025-12-28T10:30:00",
    "device": "cuda",
    "model": "llava-hf/llava-1.5-7b-hf",
    "prompt": "What do you see in this image?",
    "prefix": "This image shows"
  },
  "results": [
    {
      "index": 1,
      "image_url": "http://...",
      "question": "What do you see in this image?",
      "predicted_answer": "a scenic mountain view",
      "token_confidence": 0.8543,
      "head_delta": 0.000123,
      "attention_map_shape": [24, 24],
      "attention_map_mean": 0.0417,
      "attention_map_std": 0.0321,
      "num_generated_tokens": 5,
      "token_strings": ["a", "scenic", "mountain", "view"],
      "token_ids": [1234, 5678, ...],
      "attention_map": [[...], [...], ...]
    },
    ...
  ]
}
```

## Key Metrics Captured

For each image, the script captures:

- **predicted_answer**: Model's text response
- **token_confidence**: Confidence score for generated tokens
- **head_delta**: Layer evolution metric (change in attention patterns across layers)
- **attention_map**: Full attention map visualization data
- **attention_map_mean/std**: Statistical properties of attention
- **num_generated_tokens**: Length of generated response
- **token_strings**: Individual tokens in the response
- **token_ids**: Token IDs for analysis

## Data Sources

The script can load images from two sources:

1. **VQA Dataset** (preferred): Place your VQA dataset at:
   ```
   data_processing/data/processed/filtered_vqa_with_links.json
   ```

2. **Fallback - COCO Images**: If VQA dataset is not found, it automatically generates COCO image URLs

## Checkpointing

The script automatically saves checkpoints every 50 images (configurable) to prevent data loss:

- Checkpoint files: `checkpoint_50.json`, `checkpoint_100.json`, etc.
- Final results: `layer_evolution_results.json`

## Memory Optimization

If you run into CUDA out-of-memory errors, you can enable quantization:

In `test_layer_evolution.py`, change:
```python
runner = LlavaRunner(
    model_id="llava-hf/llava-1.5-7b-hf",
    device=device,
    quantization="4bit"  # or "8bit"
)
```

## Time Estimate

Processing time varies by hardware:
- **GPU (T4)**: ~3-5 seconds per image → ~1.5 hours for 1000 images
- **GPU (V100)**: ~1-2 seconds per image → ~30-45 minutes for 1000 images
- **CPU**: Not recommended (would take 10+ hours)

## Troubleshooting

### Image Loading Errors
Some URLs may be invalid or inaccessible. The script will:
- Skip failed images
- Continue processing remaining images
- Log errors for debugging

### Out of Memory
- Enable quantization (`quantization="4bit"`)
- Reduce batch processing (process fewer images at a time)
- Use a larger GPU in Colab (Runtime → Change runtime type → GPU → High-RAM)

### Dependencies Missing
Run the setup script again:
```bash
bash setup_colab.sh
```

## Analysis Examples

After running, you can analyze results:

```python
import json
import pandas as pd

# Load results
with open('layer_evolution_results.json') as f:
    data = json.load(f)

df = pd.DataFrame(data['results'])

# Find high-confidence predictions
high_conf = df[df['token_confidence'] > 0.9]

# Analyze head delta distribution
print(df['head_delta'].describe())

# Find images with unusual attention patterns
outliers = df[df['head_delta'] > df['head_delta'].quantile(0.95)]
```

## Files Created

- `layer_evolution_results.json` - Main output file
- `checkpoint_N.json` - Intermediate checkpoints
- `layer_evolution_analysis.png` - Visualization (if using notebook)

## License

Same as parent repository.

