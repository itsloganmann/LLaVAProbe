# Idea 1: Top-K Attention Entropy vs Token Confidence Analysis

## Summary

This study investigates the relationship between **spatial attention entropy** (a measure of how focused or dispersed the model's visual attention is) and **token confidence** (the model's certainty in its prediction). The core question: **Does focused attention correlate with higher confidence?**

## Key Finding

**The Attention-Confidence Gap is real but weak across all methods.** Neither full attention nor top-k focused attention shows strong predictive power for token confidence (R² < 0.30 for most question types).

---

## Methodology

### Attention Entropy Computation
For each VQA sample, we extract attention maps from all 32 transformer layers and compute:

1. **Full Attention Entropy**: Shannon entropy over the full 576-patch (24×24) attention distribution
2. **Top-K Attention Entropy**: Shannon entropy computed only over the K highest-attended patches (K = 5, 10, 25)

$$H = -\sum_{i=1}^{N} p_i \log_2(p_i)$$

Where $p_i$ is the normalized attention weight for patch $i$.

### Token Confidence
The model's confidence in its prediction, measured as:
- **Negative values**: Lower confidence (higher uncertainty)
- **Positive values**: Higher confidence

### Correlation Analysis
We compute Pearson correlation (R) and coefficient of determination (R²) between:
- Attention entropy
- Token confidence

Grouped by question type to understand task-specific patterns.

---

## Results

### Main Comparison: Full vs Top-K Attention Methods

| Question Type | N | R² (Full) | R² (Top-K 5) | R² (Top-K 10) | R² (Top-K 25) |
|--------------|---|-----------|--------------|---------------|---------------|
| Object ID | ~460 | Low | Low | Low | Low |
| Person ID | ~350 | Low | Low | Low | Low |
| Yes/No | ~50 | Low | Low | Low | Low |
| Location | ~30 | Low | Low | Low | Low |
| Other | ~100 | Low | Low | Low | Low |

**Note:** Exact R² values vary by question type but consistently stay below 0.30, indicating weak correlation across all methods.

### Key Observations

1. **Full Attention performs similarly to Top-K**: Restricting to top patches doesn't improve predictive power
2. **Negative R values**: More focused attention (lower entropy) often correlates with *lower* confidence, opposite to intuition
3. **Question type matters**: Some categories show stronger patterns than others, but none exceed the practical significance threshold

### Visualization

Two comparison plots generated:
- `r2_comparison_all_methods.png`: R² values by question type and method
- `r_comparison_all_methods.png`: Pearson R values showing direction of correlation

---

## Implications

### 1. Spatial Attention Metrics Are Not Reliable Confidence Predictors
The weak R² values (<0.30) across all attention methods suggest that:
- How *focused* the model's attention is doesn't reliably predict how *confident* it is
- This holds whether we look at full attention or just top-K patches

### 2. The Intuition "Focused = Confident" Is Wrong
We expected: **Focused attention → Higher confidence**  
We found: **Weak or negative correlation**

This suggests the model can be:
- Confident with dispersed attention (knows the answer from context)
- Uncertain with focused attention (looking hard but still unsure)

### 3. Top-K Filtering Doesn't Help
Restricting to top-K patches (K = 5, 10, 25) doesn't improve the correlation, suggesting:
- The "important" patches aren't more predictive than the full distribution
- Attention entropy might be the wrong metric entirely for confidence estimation

---

## Technical Details

### Dataset
- **Source**: VQAv2 validation set
- **Samples**: ~1000 questions across multiple types
- **Model**: LLaVA-1.5-7B (llava-hf/llava-1.5-7b-hf)

### Clustering Analysis
For each attention map, we also perform DBSCAN clustering to identify:
- Number of distinct attention regions (n_clusters)
- Noise ratio (n_noise)
- Per-cluster statistics (size, average strength, entropy)

### Files
- `analysis_results_final.csv`: Full attention analysis
- `analysis_results_final_topk_5.csv`: Top-5 attention analysis
- `analysis_results_final_topk_10.csv`: Top-10 attention analysis
- `analysis_results_final_topk_25.csv`: Top-25 attention analysis

---

## Conclusion

**The "Attention-Confidence Gap" is confirmed**: spatial attention metrics have weak predictive power for model confidence. This negative result is important because it:

1. **Challenges assumptions** about attention-based interpretability
2. **Suggests alternative approaches** may be needed (e.g., logit lens, internal activations)
3. **Shows consistency**: The weak correlation holds across full and top-K methods

Future work should explore:
- Layer-specific attention patterns
- Alternative confidence measures (margin, calibrated probabilities)
- Causal interventions to test if attention *causes* confidence rather than just correlates

---

## References

- SEES Attention-Confidence Gap Threshold: R² = 0.15
- Visualization script: `visualize.py`
- Data collection: `topk.py`
