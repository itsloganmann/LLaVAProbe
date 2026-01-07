# Unified Logit Lens Methodology

## Overview

This document describes the unified methodology for logit lens analysis on LLaVA, combining approaches from Yi Xia and Emily Huang.

## 1. Experimental Setup

### 1.1 Model Configuration

- **Model**: LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`)
- **Quantization**: 4-bit (BitsAndBytes NF4)
- **Device**: CUDA (GPU)
- **Precision**: float16 compute dtype

### 1.2 Dataset

- **Source**: VQAv2 validation set
- **Sample size**: 1000 samples
- **Question types**: object identification, counting, color, yes/no, other

### 1.3 Prompt Format

```
USER: <image>
{question}
Answer with a single word or phrase.
ASSISTANT:
```

## 2. Correctness Definition

### Lenient Matching (Unified)

A prediction is considered correct if ANY of these conditions hold:

```python
pred_lower = predicted_token.lower().strip()
gt_lower = ground_truth.lower().strip()

is_correct = (
    pred_lower == gt_lower or           # Exact match
    gt_lower.startswith(pred_lower) or  # GT starts with pred ("ski" → "skiing")
    pred_lower.startswith(gt_lower) or  # Pred starts with GT
    gt_lower in pred_lower or           # GT contained in pred
    pred_lower in gt_lower              # Pred contained in GT
)
```

**Rationale**: Single-token generation often produces prefixes (e.g., "Ele" for "elephant"). This matching captures semantic correctness despite tokenization artifacts.

## 3. Margin Definition

### 3.1 Per-Layer Margin

For each layer L, the margin is defined as:

```
margin(L) = logit(correct_token) - max(logit(other_tokens))
```

Where:

- `correct_token`: The ground truth answer token (or predicted token if checking model behavior)
- `other_tokens`: All vocabulary tokens except the correct one

### 3.2 Visual Attribution Margin

To measure visual contribution:

```
Δmargin(L) = margin(L, with_image) - margin(L, blank_image)
```

## 4. Attribution Decomposition

### 4.1 Residual Stream Decomposition

At each layer L, the residual stream update is:

```
r_out(L) = r_in(L) + Δr_attn(L) + Δr_mlp(L)
```

### 4.2 Contribution Calculation

Since the LM head is linear, contributions are additive:

```
Δlogit_attn(L) = Δr_attn(L) · W_U[:, token_id]
Δlogit_mlp(L) = Δr_mlp(L) · W_U[:, token_id]
```

### 4.3 Percentage Attribution

```
MLP% = sum(|Δlogit_mlp|) / (sum(|Δlogit_mlp|) + sum(|Δlogit_attn|)) × 100
Attn% = 100 - MLP%
```

## 5. Layer Groupings

### 5.1 Functional Phases

| Phase              | Layers | Primary Function        |
| ------------------ | ------ | ----------------------- |
| Feature Extraction | 0-17   | Build representations   |
| Early Signal       | 17     | First predictive signal |
| Suppression        | 21-28  | Wrong answer filtering  |
| Commitment         | 29     | Neuron "decision" point |
| Selection          | 30-31  | Final answer boosting   |

### 5.2 Key Layers for Reliability

Layers used in combined AUROC model: **17, 21, 26, 29, 30, 31**

## 6. Neuron Analysis

### 6.1 Activation Collection

For each sample at key layers (17, 21, 26, 27, 29, 30, 31):

- Extract post-activation MLP outputs
- Shape: (n_neurons,) = (11008,) for Llama-7B

### 6.2 Sparse Probing

- **Method**: L1-regularized logistic regression
- **Target**: Binary correctness label
- **Metric**: Test set accuracy, AUROC

### 6.3 Neuron Importance

Neurons ranked by:

1. Absolute coefficient magnitude (probe weight)
2. Activation difference (correct - incorrect mean)

## 7. Reliability Metrics

### 7.1 Individual Signals

- Total MLP contribution: `S_MLP = Σ Δlogit_mlp(L)`
- Total Attention contribution: `S_Attn = Σ Δlogit_attn(L)`
- Final margin: `margin(L=31)`

### 7.2 Combined Model

Logistic regression on features:

- MLP contributions at layers 17, 21, 26, 29, 30, 31
- Attention contributions at layers 17, 21, 26, 29, 30, 31

**Metric**: AUROC (Area Under ROC Curve)

## 8. Visualization Standards

### 8.1 Trajectory Plots

- X-axis: Layer index (0-31)
- Y-axis: Margin value
- Lines: Mean ± std for correct/incorrect

### 8.2 Attribution Bar Charts

- Stacked bars: MLP (green) vs Attention (blue)
- Grouped by: Layer or functional phase

## 9. Expected Results (Unified)

Based on both analyses, we expect:

| Metric                 | Expected Range |
| ---------------------- | -------------- |
| Overall MLP%           | 55-70%         |
| MLP% in L30-31         | 72-83%         |
| Attention% in L21-28   | 72-81%         |
| Combined AUROC         | 85-96%         |
| Neuron probe accuracy  | 70-86%         |
| Separation start layer | 18-21          |
| Peak separation layer  | 24-27          |

## 10. Script Inventory

| Script                           | Purpose                     |
| -------------------------------- | --------------------------- |
| `run_all_logit_lens.py`          | Master runner               |
| `logit_lens_image_comparison.py` | Step 1b: with/without image |
| `visual_layer_attribution.py`    | Step 2: MLP vs Attention    |
| `logit_lens_neuron_analysis.py`  | Step 3a: Neuron probing     |
| `step3b_ablation_tests.py`       | Step 3b: Ablation tests     |
| `step4_reliability_analysis.py`  | Step 4: AUROC analysis      |
| `question_type_analysis.py`      | Per-question-type breakdown |

## 11. Citation

This methodology combines approaches from:

- Yi Xia (n=1000, 4-bit quantization)
- Emily Huang (n=672, detailed layer analysis)
- Shikhar Shiromani (experimental design framework)

Based on the logit lens technique from:

- nostalgebraist, "interpreting GPT: the logit lens", LessWrong, 2020
