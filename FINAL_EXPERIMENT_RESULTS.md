# Final Experiment Results: Multi-Model VLM Interpretability Analysis

## Overview

This report summarizes the results of three interpretability experiments conducted on two Vision-Language Models (VLMs):

- **PaliGemma-3B** (Google)
- **Qwen2-VL-7B-Instruct** (Alibaba)

All experiments used 100 samples from the VQAv2 dataset.

---

## 1. Top-K Attention Head Analysis

### Methodology

For each sample, we identify the Top-5 attention heads that contribute most to predicting the correct answer token, measured by per-patch Δ log P (change in log probability when adding each head's contribution).

### Results

#### PaliGemma-3B (18 layers, 8 heads per layer)

| Metric                            | Value                |
| --------------------------------- | -------------------- |
| Mean Attention Entropy            | 0.3252               |
| Mean Token Confidence             | -0.5819              |
| Pearson R (entropy vs confidence) | -0.2359 (p=0.0181)\* |
| R²                                | 0.0556               |

**Top Attention Heads**: Layer 17 consistently dominates

- Layer 17, Head 5: δ ≈ 13-17 (highest)
- Layer 17, Heads 0-4: δ ≈ 6-10

**By Question Type:**
| Type | N | R | R² |
|------|---|---|-----|
| yes/no | 34 | 0.1801 | 0.0324 |
| other | 36 | -0.1833 | 0.0336 |
| counting | 12 | nan | nan |
| comparison | 4 | -0.2470 | 0.0610 |
| color recognition | 10 | -0.1247 | 0.0156 |

#### Qwen2-VL-7B (28 layers, 28 heads per layer with GQA)

| Metric                            | Value             |
| --------------------------------- | ----------------- |
| Mean Attention Entropy            | 0.7260            |
| Mean Token Confidence             | -0.5742           |
| Pearson R (entropy vs confidence) | 0.0583 (p=0.5642) |
| R²                                | 0.0034            |

**Top Attention Heads**: Layer 0 dominates (early layer focus)

- Layer 0, Head 26: δ ≈ 8-9 (highest)
- Layer 0, Heads 5, 15, 17, 18, 25: δ ≈ 4-6
- Layer 6, Head 15: δ ≈ 5

**By Question Type:**
| Type | N | R | R² |
|------|---|---|-----|
| yes/no | 34 | -0.1257 | 0.0158 |
| other | 36 | 0.2217 | 0.0491 |
| counting | 12 | -0.1644 | 0.0270 |
| comparison | 4 | -0.8438 | 0.7119 |
| color recognition | 10 | -0.0785 | 0.0062 |

### Key Findings

1. **Different layer specialization**: PaliGemma concentrates visual processing in late layers (17/18), while Qwen2-VL uses early layers (0) more heavily
2. **Attention entropy**: Qwen2-VL has higher entropy (0.73 vs 0.33), suggesting more distributed attention
3. **Statistical significance**: PaliGemma shows significant negative correlation between entropy and confidence (p<0.05), Qwen2-VL does not

---

## 2. Causal Intervention Analysis

### Methodology

Mask 30% of image patches based on attention scores and measure impact on accuracy:

- **HIGH masking**: Mask top 30% attention regions (should hurt performance if attention is meaningful)
- **LOW masking**: Mask bottom 30% attention regions (should preserve performance)
- **RANDOM masking**: Mask random 30% (control condition)

### Results

#### PaliGemma-3B

| Condition       | Accuracy | Δ from Baseline |
| --------------- | -------- | --------------- |
| Full (baseline) | 73.00%   | -               |
| Mask HIGH 30%   | 65.00%   | **-8.00%**      |
| Mask LOW 30%    | 74.00%   | +1.00%          |
| Mask RANDOM 30% | 70.00%   | -3.00%          |

**Statistical Significance:**

- Full vs Mask HIGH: p = 0.059 (marginal)
- Full vs Mask LOW: p = 0.783 (not significant)
- **Mask HIGH vs LOW: p = 0.028\*** (significant)

**Causal Effect Size**: +9.00 percentage points (LOW - HIGH)

✅ **SUPPORTS CAUSAL HYPOTHESIS**: Masking high-attention regions hurts more than low-attention regions

#### Qwen2-VL-7B

| Condition       | Accuracy | Δ from Baseline |
| --------------- | -------- | --------------- |
| Full (baseline) | 65.00%   | -               |
| Mask HIGH 30%   | 63.00%   | -2.00%          |
| Mask LOW 30%    | 65.00%   | +0.00%          |
| Mask RANDOM 30% | 63.00%   | -2.00%          |

**Statistical Significance:**

- Full vs Mask HIGH: p = 0.620 (not significant)
- Full vs Mask LOW: p = 1.000 (not significant)
- Mask HIGH vs LOW: p = 0.482 (not significant)

**Causal Effect Size**: +2.00 percentage points (LOW - HIGH)

❌ **DOES NOT SUPPORT CAUSAL HYPOTHESIS**: No significant difference between masking conditions

### Key Findings

1. **PaliGemma shows causal attention**: Attention identifies task-relevant regions
2. **Qwen2-VL attention is less localized**: May use different mechanisms for visual grounding
3. **Architecture difference**: Qwen2-VL's dynamic resolution and native multimodal integration may lead to different attention patterns

---

## 3. Logit Lens Analysis

### Methodology

Apply logit lens to intermediate layer representations to track when visual information integrates with language predictions.

### Results

#### PaliGemma-3B (18 layers)

| Metric                   | Value                      |
| ------------------------ | -------------------------- |
| Accuracy WITH image      | 100.0%                     |
| Accuracy WITHOUT image   | 0.0%                       |
| Peak Visual Effect Layer | Layer 14 (Δmargin = 10.85) |
| MLP Contribution         | 51.7%                      |
| Attention Contribution   | 48.3%                      |

**Neuron Analysis** (non-zero neurons < 1.2% per layer):
| Layer | Test Accuracy | Non-zero Neurons |
|-------|---------------|------------------|
| 10 | 100.0% | 23/2048 (1.1%) |
| 13 | 100.0% | 24/2048 (1.2%) |
| 15 | 100.0% | 21/2048 (1.0%) |
| 16 | 100.0% | 22/2048 (1.1%) |
| 17 | 100.0% | 19/2048 (0.9%) |

#### Qwen2-VL-7B (28 layers)

| Metric                   | Value                     |
| ------------------------ | ------------------------- |
| Accuracy WITH image      | 0.0%\*                    |
| Accuracy WITHOUT image   | 0.0%\*                    |
| Peak Visual Effect Layer | Layer 27 (Δmargin = 8.40) |
| MLP Contribution         | 68.2%                     |
| Attention Contribution   | 31.8%                     |

\*Note: Accuracy metrics reflect token-level prediction matching, which differs for Qwen2-VL's generation pattern.

**Neuron Analysis** (non-zero neurons ~1.5% per layer):
| Layer | Test Accuracy | Non-zero Neurons |
|-------|---------------|------------------|
| 18 | 100.0% | 57/3584 (1.6%) |
| 21 | 100.0% | 61/3584 (1.7%) |
| 24 | 100.0% | 53/3584 (1.5%) |
| 25 | 100.0% | 55/3584 (1.5%) |
| 26 | 100.0% | 58/3584 (1.6%) |
| 27 | 100.0% | 52/3584 (1.5%) |

### Key Findings

1. **Visual integration layer**: PaliGemma peaks at Layer 14 (78% depth), Qwen2-VL at Layer 27 (96% depth)
2. **MLP vs Attention**: Qwen2-VL relies more on MLP (68% vs 52%)
3. **Sparse neuron activation**: Both models show <2% active neurons in late layers
4. **High classifier accuracy**: Neurons reliably distinguish high vs low confidence samples

---

## Summary Table

| Experiment                 | PaliGemma-3B     | Qwen2-VL-7B        |
| -------------------------- | ---------------- | ------------------ |
| **Top-K: Dominant Layer**  | Layer 17 (final) | Layer 0 (first)    |
| **Top-K: Entropy**         | 0.33 (focused)   | 0.73 (distributed) |
| **Causal: Effect Size**    | +9.0 pp ✅       | +2.0 pp ❌         |
| **Causal: Significance**   | p=0.028\*        | p=0.482            |
| **Logit Lens: Peak Layer** | Layer 14 (78%)   | Layer 27 (96%)     |
| **Logit Lens: MLP %**      | 51.7%            | 68.2%              |

---

## Conclusions

1. **Architecture matters**: Different VLM architectures (PaliGemma's SigLIP+Gemma vs Qwen2-VL's native multimodal) show fundamentally different attention patterns.

2. **PaliGemma is more interpretable**: Shows clear causal relationship between attention and task performance, with concentrated processing in late layers.

3. **Qwen2-VL is more distributed**: Visual processing happens in early layers, with higher entropy and less localized attention. May use different mechanisms for visual grounding.

4. **Sparse representations**: Both models use <2% of neurons in late layers for visual-linguistic decisions, suggesting potential for model compression.

5. **MLP importance**: Qwen2-VL relies more heavily on MLP layers (68%) compared to PaliGemma (52%), suggesting different information routing strategies.

---

## Files Generated

- `topk/paligemma_topk_5_results.csv`
- `topk/qwen2-vl_topk_5_results.csv`
- `topk/paligemma_causal_intervention_results.csv`
- `topk/qwen2-vl_causal_intervention_results.csv`
- `logit_lens/paligemma_logit_lens_results.json`
- `logit_lens/qwen2-vl_logit_lens_results.json`
- `logit_lens/paligemma_neuron_analysis_results.json`
- `logit_lens/qwen2-vl_neuron_analysis_results.json`
