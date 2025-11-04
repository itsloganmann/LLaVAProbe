## Attention Confidence Analysis Pipeline

This repository now provides a reproducible pipeline for analysing spatial attention clusters and confidence metrics
for LLaVA models. The workflow automates the ideas outlined in `Ideas.pdf`, including weighted clustering sweeps,
language-only and visual ablations, confidence calibration, and rich structured reporting.

### Quick start

1. **Environment setup**

	```bash
	bash setup.sh
	```

	This creates a virtual environment and installs all required packages (PyTorch, Transformers, HDBSCAN, etc.).

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

### Legacy scripts

`main.py` and `main_batching.py` now delegate to the new pipeline but keep their previous implementations below the
entry-point guard for reference.
