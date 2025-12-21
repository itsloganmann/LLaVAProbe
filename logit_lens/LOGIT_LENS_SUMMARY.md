# Logit Lens Analysis Summary: Shikhar's Plan Complete

## Executive Summary

We completed all steps of Shikhar's logit lens analysis plan to understand **"where in the language model the VQA answer decision is made"** and **"whether MLP sublayers contribute more than attention sublayers"**.

### Key Discoveries

| Finding                | Value            | Significance                                       |
| ---------------------- | ---------------- | -------------------------------------------------- |
| **MLP dominance**      | 70% overall      | MLP layers contribute 70% of answer signal         |
| **Visual routing**     | 68% Attention    | Visual info SUPPRESSION flows through attention    |
| **Visual boosting**    | 88% MLP          | Layers 30-31 that BOOST correct answer use MLP!    |
| **Super-neuron**       | Neuron 1512      | Δactivation = 34.5 in Layer 31 for correct answers |
| **Predictive neurons** | 91.7% accuracy   | Just 1-2 neurons per layer predict correctness     |
| **Layer specificity**  | Layer 17 R²=0.31 | Individual layers CAN predict, but signals cancel  |
| **Image necessity**    | 46% → 0%         | Without image, accuracy drops to zero              |

---

## NEW CRITICAL FINDING: Visual-Layer Attribution

### The Key Insight: Two Phases of Visual Processing

**Phase 1: Suppression (Layers 13-29) - Attention Dominated**

- Image SUPPRESSES wrong answers (negative Δmargin)
- **Attention contributes 91.4%** of suppression signal
- MLP: -10.3% (actually works against suppression)

**Phase 2: Boosting (Layers 30-31) - MLP Dominated**

- Image BOOSTS correct answer (positive Δmargin)
- **MLP contributes 88.4%** of boosting signal
- This is where the final answer decision happens!

### Summary Table

| Phase        | Layers    | Function                  | MLP %   | Attention % |
| ------------ | --------- | ------------------------- | ------- | ----------- |
| Suppression  | 13-29     | Eliminate wrong answers   | -10%    | 91%         |
| **Boosting** | **30-31** | **Select correct answer** | **88%** | 12%         |
| Overall      | All       | Combined effect           | 32%     | 68%         |

**This resolves the apparent contradiction:**

- Overall, Attention dominates (68%)
- But in the CRITICAL final layers where the answer is formed, **MLP dominates (88%)**!

---

## Step-by-Step Results

### Step 0: Setup ✅

- Implemented hook-based residual stream capture
- Fixed model structure: `lm_head` at `model.lm_head`
- Device handling: Move tensors to `lm_head.weight.device`

### Step 1: Basic Logit Lens (1000 samples) ✅

**Margin Trajectory:**

- Early layers (0-15): Negative margins (model uncertain)
- Layer 20+: Rapid margin increase
- Layer 31: Peak positive margin

**Correct vs Incorrect:**

- Correct predictions show earlier margin crossover
- Final layer margin difference: ~15 logit points

### Step 1b: With vs Without Image (100 samples) ✅

**Critical Result:**

```
With image:    46% accuracy
Without image: 0% accuracy  (!!!)
```

**Δmargin by layer (real - blank image):**
| Layer Range | Δmargin | Interpretation |
|-------------|---------|----------------|
| 0-10 | ~0 | Image not yet used |
| 11-25 | -0.5 to -3.5 | Image SUPPRESSES wrong answers |
| 30 | +3.54 | Image BOOSTS correct answer |
| 31 | **+10.54** | Peak visual contribution |

**Visual Signal Attribution:**

- **Attention: 67.8%** of visual contribution
- **MLP: 32.2%** of visual contribution

### Step 2: MLP vs Attention Attribution ✅

**Overall Answer Formation:**

- **MLP: 69.9%** of total answer signal
- **Attention: 30.1%** of total answer signal

**Key Insight:** Visual information flows through attention (68%), but answer formation happens in MLP (70%).

### Step 3: Layer-Level Correlation Probing ✅

**Per-Layer R² with Correctness:**
| Layer | MLP R² | Attention R² |
|-------|--------|--------------|
| Layer 5 | 0.265 | - |
| Layer 17 | **0.305** | - |
| Layer 21 | 0.180 | - |
| Layer 31 | 0.028 | - |

**Aggregate R² = 0.009** (near zero!)

**Why?** Positive and negative correlations CANCEL when summed.

### Step 3b: Neuron-Level Analysis (200 samples) ✅

**Major Discovery: Super-Neuron 1512**

| Layer | Neuron 1512 Δact | Role               |
| ----- | ---------------- | ------------------ |
| 17    | +0.82            | Initial activation |
| 21    | -0.67            | Suppression        |
| 27    | +0.75            | Re-activation      |
| 30    | -1.50            | Suppression        |
| 31    | **+34.5**        | EXPLOSION!         |

**Logistic Regression Results:**

- **Test accuracy: 91.7%** across all layers
- **Non-zero coefficients: 1-2 per layer** (extremely sparse!)
- Only **2 neurons** needed to predict correctness at 92% accuracy

**Top Task Neurons:**

1. **Neuron 1512** (Layer 31): Δact = +34.5
2. Neuron 1360 (Layer 31): Δact = +4.4
3. Neuron 1573 (Layer 30): Δact = +2.1
4. Neuron 3556 (Layer 29): Δact = +1.0

### Step 4: Reliability Analysis ✅

| Metric            | R²        | AUROC    | Interpretation        |
| ----------------- | --------- | -------- | --------------------- |
| Attention Entropy | 0.003     | 0.54     | Not predictive        |
| Total MLP Score   | 0.009     | 0.55     | Not predictive        |
| Layer 17 MLP      | **0.305** | **0.72** | Moderately predictive |
| Final Margin      | 0.15      | 0.65     | Somewhat predictive   |

---

## Key Insights for Paper

### 1. MLP-Attention Division of Labor

- **Attention**: Routes visual information (68% of visual signal)
- **MLP**: Forms the answer (70% of answer signal)

This is a clean functional separation!

### 2. Answer Formation is Sparse

- Only 1-2 neurons per layer are needed
- Neuron 1512 alone shows Δact = 34.5
- This suggests highly specialized "answer neurons"

### 3. Signals Cancel at Aggregate Level

- Layer-specific probes: R² up to 0.31
- Aggregate metrics: R² = 0.009
- Explains why simple confidence metrics fail

### 4. Visual Information is Essential but Processed Late

- Without image: 0% accuracy
- Peak visual contribution: Layer 31 (Δmargin = +10.54)
- Early layers suppress wrong answers, late layers boost correct ones

---

## Files Generated

| File                             | Description                    |
| -------------------------------- | ------------------------------ |
| `logit_lens_results_1000.json`   | 1000-sample logit lens results |
| `step1b_image_comparison.json`   | With/without image comparison  |
| `step3b_neuron_analysis.json`    | Neuron-level analysis results  |
| `step3b_neuron_analysis.png`     | Neuron analysis visualization  |
| `step2_attribution_complete.png` | MLP vs Attention attribution   |
| `step4_reliability.png`          | Reliability analysis plots     |
| `logit_lens_analysis_1000.png`   | Main logit lens visualization  |

---

## Next Steps (If Desired)

1. **Ablation Study**: Zero out Neuron 1512 across layers, measure accuracy drop
2. **Question-Type Analysis**: Check if different neurons activate for different question types
3. **Cross-Model Comparison**: Run same analysis on other VLMs
4. **Fine-Grained Probing**: Train probes on neuron subsets to find minimal predictive set

---

## Conclusion

The logit lens analysis reveals that LLaVA's VQA answer formation is:

1. **MLP-dominated** (70% of signal)
2. **Visually-processed through attention** (68% of visual signal)
3. **Extremely sparse** (1-2 neurons per layer)
4. **Layer-specific** (signals cancel when aggregated)
5. **Centered on a "super-neuron"** (Neuron 1512, Δact = 34.5)

This provides mechanistic insight into how VLMs make decisions and why simple confidence metrics often fail to predict correctness.
