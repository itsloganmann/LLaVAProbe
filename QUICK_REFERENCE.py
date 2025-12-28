#!/usr/bin/env python3
"""
Quick Reference - Layer Evolution Analysis

This file contains copy-paste commands for common tasks.
"""

# ============================================================================
# GOOGLE COLAB SETUP (Run in separate cells)
# ============================================================================

# Cell 1: Clone and setup
"""
!git clone https://github.com/YOUR_USERNAME/LLaVAProbe-1.git
%cd LLaVAProbe-1
!bash setup_colab.sh
"""

# Cell 2: Check GPU
"""
import torch
print(f"GPU: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
"""

# Cell 3: Quick test (5 images)
"""
!python test_quick.py
"""

# Cell 4: Full analysis (1000 images)
"""
!python test_layer_evolution.py
"""

# Cell 5: Download results
"""
from google.colab import files
files.download('layer_evolution_results.json')
"""

# ============================================================================
# LOAD AND ANALYZE RESULTS
# ============================================================================

"""
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load results
with open('layer_evolution_results.json', 'r') as f:
    data = json.load(f)

# Show metadata
print(json.dumps(data['metadata'], indent=2))

# Convert to DataFrame
df = pd.DataFrame(data['results'])

# Basic statistics
print(df['token_confidence'].describe())
print(df['head_delta'].describe())

# Find high-confidence predictions
high_conf = df[df['token_confidence'] > 0.9]
print(f"High confidence samples: {len(high_conf)}")

# Find outliers in head_delta
outliers = df[df['head_delta'] > df['head_delta'].quantile(0.95)]
print(f"Head delta outliers: {len(outliers)}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(df['token_confidence'], bins=50)
axes[0].set_title('Token Confidence Distribution')
axes[1].hist(df['head_delta'], bins=50)
axes[1].set_title('Head Delta Distribution')
plt.tight_layout()
plt.savefig('analysis.png')
plt.show()
"""

# ============================================================================
# CUSTOMIZE CONFIGURATION
# ============================================================================

"""
# Edit test_layer_evolution.py and change:

NUM_IMAGES = 500              # Process fewer images
OUTPUT_FILE = "my_results.json"  # Custom output name
CHECKPOINT_INTERVAL = 25      # Save more frequently
DEFAULT_PROMPT = "Describe this image in detail"  # Custom prompt
"""

# ============================================================================
# MEMORY OPTIMIZATION (if running out of CUDA memory)
# ============================================================================

"""
# In test_layer_evolution.py, change the runner initialization:

runner = LlavaRunner(
    model_id="llava-hf/llava-1.5-7b-hf",
    device=device,
    quantization="4bit"  # Add this line for 4-bit quantization
)

# Or use "8bit" for 8-bit quantization
"""

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
# Problem: CUDA out of memory
# Solution: Enable quantization (see above)

# Problem: Some images fail to load
# Solution: This is normal, the script will skip them and continue

# Problem: Process interrupted
# Solution: Check for checkpoint files (checkpoint_N.json)
#           You can restart from these if needed

# Problem: Dependencies missing
# Solution: Re-run setup
!bash setup_colab.sh

# Problem: Model download failing
# Solution: Check internet connection, HuggingFace may be down
#           Try again later or use cached model
"""

# ============================================================================
# FILE LOCATIONS
# ============================================================================

"""
Input:
  - data_processing/data/processed/filtered_vqa_with_links.json (optional)

Output:
  - layer_evolution_results.json (main results)
  - checkpoint_N.json (intermediate checkpoints)
  - layer_evolution_analysis.png (if using notebook)

Documentation:
  - LAYER_EVOLUTION_README.md (detailed docs)
  - IMPLEMENTATION_SUMMARY.md (technical overview)
  - Run_Layer_Evolution_Colab.ipynb (interactive notebook)
"""

# ============================================================================
# COMMON ANALYSES
# ============================================================================

"""
# Find images where model is uncertain
uncertain = df[df['token_confidence'] < 0.5]

# Find images with unusual attention patterns
unusual_attention = df[df['head_delta'] > df['head_delta'].mean() + 2*df['head_delta'].std()]

# Correlate confidence with attention
correlation = df['token_confidence'].corr(df['head_delta'])
print(f"Confidence-HeadDelta correlation: {correlation:.4f}")

# Group by response length
df.groupby('num_generated_tokens')['token_confidence'].mean()

# Export subset for further analysis
interesting = df[df['token_confidence'] > 0.8]
interesting.to_csv('high_confidence_samples.csv', index=False)
"""

print(__doc__)

