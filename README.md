## Attention Confidence Analysis Pipeline

This repository now provides a reproducible pipeline for analysing spatial attention clusters and confidence metrics
for LLaVA models. The workflow automates the ideas outlined in `Ideas.pdf`, including weighted clustering sweeps,
language-only and visual ablations, confidence calibration, and rich structured reporting.

### Quick start

1. **Environment setup**

	**🚀 Option A: Google Colab (Easiest - Recommended for Testing)**
	
	Perfect for quick testing with free GPU access! LLaVA automatically downloads from HuggingFace.
	
	**Step 1:** Open [Google Colab](https://colab.research.google.com/) and enable GPU
	- Go to: `Runtime` → `Change runtime type` → `Hardware accelerator` → `GPU (T4)`
	
	**Step 2:** Run this in a Colab cell to setup everything:
	
	```python
	# Clone repository and checkout Idea #3 branch
	!git clone https://github.com/itsloganmann/LLaVAProbe.git
	%cd LLaVAProbe
	!git checkout "Idea-#3----Attention-Evolution-Tracking"
	
	# Install dependencies (takes ~2-3 minutes)
	!pip install -q -r requirements.txt
	
	# Copy custom model files to enable attention tracking
	import transformers, os, shutil
	trans_path = os.path.dirname(transformers.__file__)
	shutil.copy("modeling_llava.py", f"{trans_path}/models/llava/modeling_llava.py")
	shutil.copy("modeling_llama.py", f"{trans_path}/models/llama/modeling_llama.py")
	print("✅ Setup complete! LLaVA will auto-download on first run.")
	```
	
	**Step 3:** Test the attention tracking:
	
	```python
	# Run the test script (LLaVA downloads automatically ~13GB, takes 3-5 min first time)
	!python test_layer_evolution.py
	```
	
	**Step 4 (Optional):** Run full analysis pipeline:
	
	```python
	# Create sample prompts file or use existing results.csv
	!python analysis/pipeline_runner.py \
	    --prompts results.csv \
	    --output-dir analysis_outputs \
	    --quantization 4bit  # Use 4bit for free Colab T4 GPU
	```
	
	**💡 Colab Tips:**
	- Free T4 GPU has ~15GB VRAM (enough for LLaVA with 4-bit quantization)
	- Session persists for ~12 hours or until idle timeout (~90 min)
	- Model downloads are cached during session (redownloads if session restarts)
	- To save outputs: Mount Google Drive or download via `files.download()`
	
	**📥 Working with Local Model (Advanced):**
	
	If you've pre-downloaded LLaVA locally in Colab:
	
	```python
	from analysis.llava_runner import LlavaRunner
	
	# Point to your local model path in Colab
	runner = LlavaRunner(
	    model_id="/content/llava-1.5-7b-hf",  # Your local path
	    device="cuda",
	    quantization="4bit"
	)
	```

	---

	**Option B: Google Cloud GPU Instance (recommended for production)**
	
	For Google Cloud VM with GPU (T4, V100, or A100):
	
	```bash
	# Clone and setup in one command
	git clone https://github.com/itsloganmann/LLaVAProbe.git && \
	cd LLaVAProbe && \
	git checkout "Idea-#3----Attention-Evolution-Tracking" && \
	bash setup_gcloud.sh
	```
	
	This script automatically:
	- Detects GPU and verifies CUDA installation
	- Creates virtual environment
	- Installs PyTorch with CUDA 11.8 support
	- Installs all dependencies optimized for cloud GPUs
	- Copies custom model files with automatic backup
	- Verifies the complete installation

	**Option C: Using setup.sh (local development)**
	
	```bash
	bash setup.sh
	```
	
	This creates a virtual environment, installs all required packages (PyTorch, Transformers, HDBSCAN, etc.), and copies the custom model files to the transformers package directory.

	**Option D: Manual installation with requirements.txt**
	
	```bash
	python3 -m venv sees
	source sees/bin/activate
	
	# Install PyTorch with CUDA support (for GPU)
	pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
	
	# Install remaining dependencies
	pip install -r requirements.txt
	
	# Copy custom model files
	TRANSFORMERS_PATH=$(python3 -c "import transformers, os; print(os.path.dirname(transformers.__file__))")
	cp modeling_llava.py "$TRANSFORMERS_PATH/models/llava/modeling_llava.py"
	cp modeling_llama.py "$TRANSFORMERS_PATH/models/llama/modeling_llama.py"
	```
	
	**Note:** The custom model files enable attention tracking and intermediate value extraction required for the analysis pipeline.

2. **Generate prompts (optional)**

	Use `tester.py` to refresh `results.csv` if you need new prompt/image pairs.

3. **Run the analysis pipeline**

	```bash
	python analysis/pipeline_runner.py --prompts results.csv --output-dir analysis_outputs
	```

	Key options:

	- `--quantization {none,4bit,8bit}`: load LLaVA with optional bitsandbytes quantisation.
	- `--log-level`: adjust verbosity (default `INFO`).

4. **Outputs**

	The pipeline produces the following artefacts inside the chosen output directory:

	- `analysis_records.json` / `analysis_records.csv`: per-example metrics including token probabilities, margins,
	  entropies, clustering stats, head ablation deltas, and ablation summaries.
	- `clustering_summary.json`: correlation statistics for every DBSCAN/HDBSCAN/GMM configuration plus null-model
	  baselines.
	- `pipeline_execution.log`: full execution log (timestamps, errors, timing).

### What the pipeline covers

- Weighted 3D DBSCAN sweeps with sample-weight exponents and sensitivity analyses.
- Null-model permutation tests, HDBSCAN, and Gaussian Mixture alternatives.
- Normalised attention entropy and token entropy fixes.
- Ensemble confidence metrics (top-k margin, logit margin, per-token log-probs, sequence probability).
- Confidence calibration (ECE & Brier) for yes/no and short-answer subsets.
- Answer-token vs ground-truth-token head deltas.
- Language-only, visual dropout, head-ablation, and prefix-control experiments.
- Image resolution sweeps (224, 336, 448) with structured exports for downstream analysis.

### Google Cloud Deployment

**Recommended Instance Configuration:**

| Component | Recommendation | Notes |
|-----------|---------------|-------|
| **Instance Type** | `n1-standard-8` or `n1-highmem-8` | 8 vCPUs, 30-52 GB RAM |
| **GPU** | NVIDIA T4, V100, or A100 | T4 for cost-efficiency, A100 for speed |
| **OS** | Ubuntu 20.04 LTS or later | With CUDA 11.8+ pre-installed |
| **Boot Disk** | 100GB+ SSD | Models + data require significant space |
| **Region** | `us-central1` or `us-west1` | Lower latency, good GPU availability |

**Memory Requirements:**
- **Full precision (FP16)**: ~14GB VRAM for LLaVA-1.5-7B
- **8-bit quantization**: ~8GB VRAM (recommended for T4)
- **4-bit quantization**: ~5GB VRAM (fastest setup time)

**Quick Start Command for Google Cloud SSH:**

```bash
git clone https://github.com/itsloganmann/LLaVAProbe.git && \
cd LLaVAProbe && \
git checkout "Idea-#3----Attention-Evolution-Tracking" && \
bash setup_gcloud.sh
```

**Running with Quantization (recommended for T4 GPUs):**

```bash
# After setup
source sees/bin/activate
python analysis/pipeline_runner.py \
    --prompts results.csv \
    --output-dir analysis_outputs \
    --quantization 4bit
```

### Dependencies

The project requires the following key dependencies (see `requirements.txt` for complete list):

- **Core ML**: PyTorch 2.1.2 (with CUDA 11.8), Transformers 4.37.1, Accelerate 0.26.1
- **Image Processing**: Pillow 10.2.0, OpenCV 4.10.0.84
- **Data Analysis**: NumPy 1.26.4, pandas, scikit-learn 1.4.2
- **Clustering**: HDBSCAN 0.8.33, UMAP-learn 0.5.6
- **Quantization**: bitsandbytes 0.42.0 (required for memory-efficient GPU inference)
- **Visualization**: matplotlib 3.7.4, bertviz 1.4.0
- **Jupyter**: jupyter, ipykernel 6.29.4, ipython 8.12.3

All dependencies are pinned to specific versions for reproducibility and optimized for Google Cloud GPU instances.

### Legacy scripts

`main.py` and `main_batching.py` now delegate to the new pipeline but keep their previous implementations below the
entry-point guard for reference.
