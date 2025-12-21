# Logit Lens Experiment Plan for LLaVAProbe

## Overview

This experiment investigates **where in the language model the VQA answer decision is made**,
and whether **MLP layers contribute more than attention** to the final answer.

This directly addresses the question: "If visual attention doesn't predict reliability,
what does?"

## Hypotheses to Test

### H1: Layer Location

> The logit for the correct answer becomes large mainly in **middle/late LM layers**,
> not in the visual encoder or early fusion layers.

**Metric**: Track `margin(layer) = logit_correct - max(logit_other)` across all 32 layers.

**Expected**: Sigmoid-ish curve where margin is small in early layers, grows in later ones.

### H2: Component Type

> Within those layers, **MLP sublayers contribute more** to the answer logit than attention sublayers.

**Metric**: Decompose each layer's contribution:

- `Δlogit_attn = Δr_attn · W_U[:, token]`
- `Δlogit_mlp = Δr_mlp · W_U[:, token]`

**Expected**: MLP fraction > 60% of total positive logit contribution.

### H3: Reliability Signal

> MLP-derived scores show **higher correlation with correctness** than attention metrics.

**Metric**: Compare `R²(total_MLP, correctness)` vs `R²(attention_entropy, correctness)`.

**Expected**: MLP R² > 0.1 (vs attention R² ≈ 0.003 from our existing results).

## Implementation

### Files Created

1. **`analysis/logit_lens.py`** - Main analysis class

   - `LogitLensAnalyzer`: Hooks into LLaVA layers
   - `analyze_sample()`: Run logit lens on single VQA sample
   - `summarize_results()`: Aggregate across samples

2. **`analysis/logit_lens_analysis.py`** - Visualization and reporting
   - `plot_margin_trajectory()`: Where does the answer emerge?
   - `plot_component_contributions()`: MLP vs Attention per layer
   - `compute_mlp_reliability_correlation()`: Does MLP predict correctness?

### Running the Experiment

```bash
# Run logit lens on 50 samples
python -m analysis.logit_lens

# Analyze and visualize results
python -m analysis.logit_lens_analysis logit_lens_results.json
```

## Expected Outputs

### 1. Margin Trajectory Plot

Shows where in the model the answer "emerges":

- X-axis: Layer index (0-31)
- Y-axis: Margin = logit_correct - max_other
- Expected: Sigmoid growth in middle/late layers

### 2. Component Contribution Plot

Shows MLP vs Attention contribution per layer:

- Stacked bar chart of Δlogit per component
- Cumulative contribution curves
- Pie chart of total MLP vs Attention fraction

### 3. Reliability Comparison Table

| Metric                     | R² vs Correctness |
| -------------------------- | ----------------- |
| Attention Entropy          | 0.003             |
| Cluster Count              | 0.004             |
| **Total MLP Contribution** | **???**           |
| **Best MLP Layer**         | **???**           |

## How This Strengthens the Paper

### Current Story (Incomplete)

> "Visual attention doesn't predict reliability" (negative finding)

### New Story (Complete)

> "Visual attention doesn't predict reliability, **but MLP computations in the
> language model do**. The answer decision happens in middle/late layers,
> primarily through MLP circuits, not attention."

### Paper Section Updates

1. **Abstract**: Add "We find that MLP layers contribute X% of the answer logit,
   and MLP-derived scores show stronger correlation with correctness (R²=Y)
   than attention metrics (R²=0.003)."

2. **New Section 5.3**: "Language Model Component Analysis"

   - Logit lens trajectory figure
   - MLP vs Attention contribution figure
   - Reliability correlation comparison

3. **Conclusion**: "Our results suggest that VQA reliability is primarily
   determined by MLP computations in the language model backbone,
   not by visual attention patterns."

## Future Extensions (Optional)

### Neuron-Level Analysis

- Identify specific neurons correlated with question types
- Ablation tests: Does zeroing certain neurons tank counting but not color?
- Report task-specific "feature neurons"

### Image vs No-Image Comparison

- Run same question with real image vs blank
- Compute Δmargin(image) - Δmargin(blank) per layer
- Identifies which layers actually use visual information

## Timeline

| Task                          | Time Estimate |
| ----------------------------- | ------------- |
| Run logit lens on 100 samples | ~2 hours      |
| Analyze results               | 30 min        |
| Create figures                | 1 hour        |
| Write paper section           | 2 hours       |
| **Total**                     | **~6 hours**  |

## Connection to Existing Findings

This experiment directly connects to:

1. **Terminal Diffusion (Ishan)**: The attention spike at Layer 31 may correspond
   to MLP taking over for final answer generation.

2. **Top-K Aggregation (Yi Xia)**: Aggregating more attention heads doesn't help
   because the signal is in MLP, not attention.

3. **Classifier (Saad)**: The 72.5% accuracy may come from MLP-related features
   (confidence), not attention clustering.

4. **Causal Intervention (Logan)**: Attention is necessary (masking hurts)
   but MLP transforms the visual features into the answer.
