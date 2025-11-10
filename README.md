## Attention Confidence Analysis Pipeline

This repository now provides a reproducible pipeline for analysing spatial attention clusters and confidence metrics
for LLaVA models. The workflow automates the ideas outlined in `Ideas.pdf`, including weighted clustering sweeps,
language-only and visual ablations, confidence calibration, and rich structured reporting.

### Quick start

1. **Environment setup**

	**Option A: Using setup.sh (recommended)**
	
	```bash
	bash setup.sh
	```
	
	This creates a virtual environment, installs all required packages (PyTorch, Transformers, HDBSCAN, etc.), and copies the custom model files to the transformers package directory.

	**Option B: Using requirements.txt**
	
	```bash
	python3.10 -m venv sees
	source sees/bin/activate
	pip install -r requirements.txt
	```
	
	**Note:** After installing via requirements.txt, you must manually copy the custom model files:
	
	```bash
	TRANSFORMERS_PATH=$(python3 -c "import transformers, os; print(os.path.dirname(transformers.__file__))")
	cp modeling_llava.py "$TRANSFORMERS_PATH/models/llava/modeling_llava.py"
	cp modeling_llama.py "$TRANSFORMERS_PATH/models/llama/modeling_llama.py"
	```
	
	The custom model files enable attention tracking and intermediate value extraction required for the analysis pipeline.

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

### Dependencies

The project requires the following key dependencies (see `requirements.txt` for complete list):

- **Core ML**: PyTorch 2.1.2, Transformers 4.37.1, Accelerate 0.26.1
- **Image Processing**: Pillow 10.2.0, OpenCV 4.10.0.84
- **Data Analysis**: NumPy 1.26.4, pandas, scikit-learn 1.4.2
- **Clustering**: HDBSCAN 0.8.33, UMAP-learn 0.5.6
- **Quantization**: bitsandbytes 0.42.0 (optional, for 4-bit/8-bit model loading)
- **Visualization**: matplotlib 3.7.4, bertviz 1.4.0
- **Jupyter**: jupyter, ipykernel 6.29.4, ipython 8.12.3

All dependencies are pinned to specific versions for reproducibility.

### Legacy scripts

`main.py` and `main_batching.py` now delegate to the new pipeline but keep their previous implementations below the
entry-point guard for reference.
