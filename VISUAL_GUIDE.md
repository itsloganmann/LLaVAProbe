# Layer Evolution Analysis - Visual Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    GOOGLE COLAB WORKFLOW                         │
└─────────────────────────────────────────────────────────────────┘

Step 1: Setup
┌──────────────────────────────────────┐
│  git clone repo                      │
│  cd LLaVAProbe-1                     │
│  bash setup_colab.sh                 │
└──────────────────────────────────────┘
                  ↓
Step 2: Quick Test (Optional but Recommended)
┌──────────────────────────────────────┐
│  python test_quick.py                │
│  → Tests with 5 images               │
│  → Verifies everything works         │
│  → Takes ~30 seconds                 │
└──────────────────────────────────────┘
                  ↓
Step 3: Run Full Analysis
┌──────────────────────────────────────┐
│  python test_layer_evolution.py      │
│  → Processes 1000 images             │
│  → Saves checkpoints every 50        │
│  → Takes ~1-2 hours on T4            │
└──────────────────────────────────────┘
                  ↓
Step 4: Analyze Results
┌──────────────────────────────────────┐
│  Load JSON → pandas DataFrame        │
│  Calculate statistics                │
│  Create visualizations               │
│  Export findings                     │
└──────────────────────────────────────┘
                  ↓
Step 5: Download
┌──────────────────────────────────────┐
│  files.download('results.json')      │
│  files.download('analysis.png')      │
└──────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                    DATA FLOW DIAGRAM                             │
└─────────────────────────────────────────────────────────────────┘

Input Sources:
┌────────────────────┐        ┌────────────────────┐
│  VQA Dataset       │   OR   │  COCO Image URLs   │
│  (Primary)         │        │  (Fallback)        │
└────────────────────┘        └────────────────────┘
         │                             │
         └──────────┬──────────────────┘
                    ↓
         ┌─────────────────────┐
         │  Load 1000 Images   │
         └─────────────────────┘
                    ↓
         ┌─────────────────────┐
         │  LLaVA Model        │
         │  - Load Image       │
         │  - Run Inference    │
         │  - Extract Metrics  │
         └─────────────────────┘
                    ↓
    ┌────────────────────────────────┐
    │  For Each Image, Capture:      │
    │  • predicted_answer            │
    │  • token_confidence            │
    │  • head_delta                  │
    │  • attention_map               │
    │  • attention statistics        │
    │  • token_strings & IDs         │
    └────────────────────────────────┘
                    ↓
         ┌─────────────────────┐
         │  Save Every 50:     │
         │  checkpoint_N.json  │
         └─────────────────────┘
                    ↓
         ┌─────────────────────┐
         │  Final Output:      │
         │  layer_evolution_   │
         │  results.json       │
         └─────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                    FILE STRUCTURE                                │
└─────────────────────────────────────────────────────────────────┘

LLaVAProbe-1/
│
├── test_layer_evolution.py          ⭐ MAIN SCRIPT (run this!)
├── test_quick.py                    🧪 Quick 5-image test
├── setup_colab.sh                   🔧 One-command setup
│
├── Run_Layer_Evolution_Colab.ipynb  📓 Interactive notebook
│
├── LAYER_EVOLUTION_README.md        📚 Detailed documentation
├── IMPLEMENTATION_SUMMARY.md        📋 Technical overview
├── QUICK_REFERENCE.py               ⚡ Copy-paste commands
│
├── requirements.txt                 📦 Dependencies
│
├── analysis/
│   ├── llava_runner.py             🤖 Model interface
│   └── ...
│
└── data_processing/
    └── data/processed/
        └── filtered_vqa_with_links.json  💾 (optional) VQA data


┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT STRUCTURE                              │
└─────────────────────────────────────────────────────────────────┘

layer_evolution_results.json
│
├── metadata                    📊 Summary statistics
│   ├── total_images           → 1000
│   ├── successful             → 980
│   ├── failed                 → 20
│   ├── timestamp              → ISO-8601
│   ├── device                 → "cuda"/"cpu"
│   ├── model                  → "llava-hf/llava-1.5-7b-hf"
│   ├── prompt                 → default prompt
│   └── prefix                 → default prefix
│
└── results[]                   📝 Per-image results
    ├── [0]
    │   ├── index              → 1
    │   ├── image_url          → "http://..."
    │   ├── question           → "What do you see..."
    │   ├── predicted_answer   → "a scenic mountain..."
    │   ├── token_confidence   → 0.854
    │   ├── head_delta         → 0.000123
    │   ├── attention_map_shape → [24, 24]
    │   ├── attention_map_mean → 0.042
    │   ├── attention_map_std  → 0.032
    │   ├── num_generated_tokens → 5
    │   ├── token_strings      → ["a", "scenic", ...]
    │   ├── token_ids          → [1234, 5678, ...]
    │   └── attention_map      → [[...], [...], ...]
    │
    ├── [1]
    │   └── ...
    │
    └── [999]
        └── ...


┌─────────────────────────────────────────────────────────────────┐
│                    KEY METRICS EXPLAINED                         │
└─────────────────────────────────────────────────────────────────┘

token_confidence
├─ What: Confidence score for generated tokens
├─ Range: 0.0 to 1.0
├─ High (>0.8): Model is confident
└─ Low (<0.5): Model is uncertain

head_delta
├─ What: Change in attention patterns across layers
├─ Range: Small positive values (typically <0.001)
├─ High: Significant attention evolution
└─ Low: Stable attention across layers

attention_map
├─ What: Spatial attention weights [24x24]
├─ Shows: Where model focuses in image
├─ Use: Visualize attention patterns
└─ Stats: mean & std summarize distribution

num_generated_tokens
├─ What: Length of model's response
├─ Range: Typically 1-50 tokens
├─ Short (<5): Brief answers
└─ Long (>20): Detailed descriptions


┌─────────────────────────────────────────────────────────────────┐
│                    PERFORMANCE ESTIMATES                         │
└─────────────────────────────────────────────────────────────────┘

GPU Type         Time/Image    Total Time (1000)    Memory
─────────────────────────────────────────────────────────────────
T4 (Colab Free)    3-5 sec      1.5 hours           8-10 GB
V100 (Colab Pro)   1-2 sec      30-45 min           8-10 GB
A100               0.5-1 sec    10-20 min           8-10 GB
CPU (not rec.)     30-60 sec    10+ hours           4-6 GB

With 4-bit quantization: Memory usage halved (~4-5 GB)


┌─────────────────────────────────────────────────────────────────┐
│                    QUICK COMMANDS CHEATSHEET                     │
└─────────────────────────────────────────────────────────────────┘

Setup:
  bash setup_colab.sh

Test:
  python test_quick.py

Run:
  python test_layer_evolution.py

Check GPU:
  python -c "import torch; print(torch.cuda.is_available())"

Load results:
  python -c "import json; print(json.load(open('layer_evolution_results.json'))['metadata'])"

Convert to CSV:
  python -c "import pandas as pd, json; pd.DataFrame(json.load(open('layer_evolution_results.json'))['results']).to_csv('results.csv')"


┌─────────────────────────────────────────────────────────────────┐
│                    TROUBLESHOOTING GUIDE                         │
└─────────────────────────────────────────────────────────────────┘

Problem: CUDA out of memory
Solution: Enable quantization in test_layer_evolution.py
          Change: quantization=None → quantization="4bit"

Problem: Some images fail
Solution: Normal! Script skips and continues
          Check final metadata for success rate

Problem: Process interrupted
Solution: Check for checkpoint_N.json files
          Can manually merge or restart

Problem: Model download slow
Solution: First run downloads ~13GB
          Subsequent runs use cache

Problem: Dependencies error
Solution: Re-run: bash setup_colab.sh
          Or: pip install -r requirements.txt


┌─────────────────────────────────────────────────────────────────┐
│                    NEXT STEPS                                    │
└─────────────────────────────────────────────────────────────────┘

After getting results:

1. Analyze distributions
   → Look for patterns in confidence/head_delta

2. Find outliers
   → Images with unusual metrics may be interesting

3. Correlate metrics
   → Does confidence relate to attention evolution?

4. Compare categories
   → Do different image types show different patterns?

5. Visualize attention
   → Use attention_map data to create heatmaps

6. Export findings
   → Create visualizations, write up insights

7. Iterate
   → Adjust prompts, test different configurations


Happy analyzing! 🚀
```

