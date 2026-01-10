# Idea 5: Causal Interventions on Attended Regions

## Summary

This study implements **causal intervention analysis** on LLaVA's attention mechanisms to understand which attention heads causally contribute to the model's predictions. We compute per-head **Δ log P** (change in log-probability) when isolating each head's contribution, then aggregate the top-K heads to create interpretable attention maps.

## Core Question

**Can we identify which attention heads *cause* the model's predictions vs. merely correlate with them?**

---

## Methodology

### 1. Per-Head Causal Contribution (Δ log P)

For each transformer layer $\ell$ and head $h$, we compute:

$$\Delta \log P(y | h, \ell) = \log P(y | \text{residual} + \text{head}_h) - \log P(y | \text{residual})$$

Where:
- $y$ is the predicted token
- $\text{residual}$ is the residual stream at layer $\ell$ 
- $\text{head}_h$ is the output from head $h$ alone

This measures **how much each head contributes** to predicting the final token.

### 2. Top-K Head Selection

We select the $K$ heads with the largest positive $\Delta \log P$:
- These are the heads that most strongly *boost* the correct prediction
- Default $K = 5$ (following the paper's methodology)

### 3. Aggregated Attention Maps

For selected heads, we:
1. Extract per-patch attention contributions
2. Weight by normalized head scores (softmax)
3. Aggregate into a 576-dimensional (24×24 grid) attention map

### 4. Spatial Clustering Analysis

The aggregated map is then:
1. Filtered (80th percentile threshold)
2. Weighted by attention strength (duplication)
3. Clustered using DBSCAN (eps=1.3, min_samples=15)

**Metrics computed:**
- Number of clusters
- Per-cluster size and average strength
- Per-cluster entropy
- Global attention entropy

---

## Results

### Dataset
- **Source**: VQAv2 validation
- **Samples**: ~92 images across multiple question types
- **Model**: LLaVA-1.5-7B

### Key Findings

#### 1. Causal Head Identification Works
The Δ log P method successfully identifies heads that causally influence predictions:
- Top-5 heads typically span **multiple layers** (not just final layer)
- Head importance varies significantly by question type

#### 2. Weak Attention-Confidence Correlation Persists
Even with causal (not just correlational) attention selection:

| Metric | Full Attention | Top-K Causal |
|--------|----------------|--------------|
| R² with confidence | Low | Low |
| Correlation direction | Mixed | Mixed |

**The "Attention-Confidence Gap" holds even for causally-identified heads.**

#### 3. Clustering Reveals Spatial Patterns
DBSCAN identifies distinct attention regions:
- Typical samples show 3-8 clusters
- High noise ratio (40-60%) indicates dispersed attention
- Cluster entropy varies by question type

---

## Technical Implementation

### Head Contribution Computation

```python
# For each head in each layer:
head_contribution = attention_weights @ V_head @ o_proj_head
added_state = head_contribution + residual_at_layer
delta_log_p = log_prob(added_state, target_token) - log_prob(residual_only, target_token)
```

### Top-K Aggregation

```python
# Select top-K heads by delta_log_p
top_k_heads = sorted(all_heads, key=lambda x: x.delta_log_p)[-K:]

# Compute softmax weights
weights = softmax([h.delta_log_p for h in top_k_heads])

# Aggregate per-patch contributions
aggregated_map = sum(w * head.per_patch_contribution for w, head in zip(weights, top_k_heads))
```

### Entropy Computation

$$H = -\sum_{i=1}^{576} p_i \log_2(p_i) / \log_2(576)$$

Normalized to [0, 1] where 0 = completely focused, 1 = completely uniform.

---

## Implications

### 1. Causal ≠ Predictive
Even when we **causally identify** important heads (not just correlated ones), their attention patterns don't reliably predict confidence. This strengthens the "Attention-Confidence Gap" finding.

### 2. Multi-Layer Importance
Important heads are distributed across layers:
- Early layers: Feature extraction
- Middle layers: Cross-modal integration  
- Final layers: Answer selection

This suggests **no single layer** is responsible for visual grounding.

### 3. Question-Type Variability
Different question types (counting, identification, yes/no) engage different head patterns:
- Spatial questions → more localized attention
- Semantic questions → more distributed attention

---

## Comparison with Idea 1 (Top-K Attention Entropy)

| Aspect | Idea 1 | Idea 5 |
|--------|--------|--------|
| Head selection | All heads equally | Causal Δ log P ranking |
| Attention aggregation | Simple average | Weighted by causal importance |
| Dataset size | ~1000 samples | ~92 samples |
| Core insight | Same | Same (weak correlation) |

**Conclusion**: Whether using simple attention or causally-weighted attention, the Attention-Confidence Gap persists.

---

## Files

- `topk.py`: Main analysis script with causal head computation
- `results/analysis_results_final.csv`: Full attention baseline
- `results/analysis_results_final_topk.csv`: Top-K causal attention results
- `results/visualize.py`: R² comparison plots
- `results/r2_comparison.png`: Visual comparison of methods

---

## Conclusion

**Causal interventions confirm the negative result**: Attention patterns—even when selected for causal importance—do not reliably predict model confidence. This suggests:

1. **Confidence emerges from deeper computations** beyond attention patterns
2. **Alternative interpretability methods** (e.g., logit lens, activation probing) may be more informative
3. **The VLM decision process** is more distributed than attention maps suggest

This work provides causal evidence that attention-based interpretability has fundamental limitations for confidence estimation in VLMs.

---

## References

- Section 3.3: Per-head contribution analysis
- Section 3.4: DBSCAN clustering pipeline
- Section 3.5: Entropy-based metrics
- Equation (4): Δ log P computation
