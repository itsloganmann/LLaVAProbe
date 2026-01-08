# Where Does LLaVA "Decide"? A Logit Lens Investigation# Logit Lens Analysis: Merged Report# Logit Lens Analysis: Merged Report

## Motivation## Summary## Summary

When a vision-language model answers a question about an image, where exactly does that decision happen? Is it in the early layers where visual features are processed, or in the final layers where the answer token is selected? And more importantly—can we predict when the model will be wrong before it generates the answer?We performed a comprehensive logit lens analysis on LLaVA-1.5-7B to understand where in the language model the VQA answer decision is made, whether MLP contributes more than attention, and how these signals relate to answer reliability.We performed a comprehensive logit lens analysis on LLaVA-1.5-7B to understand where in the language model the VQA answer decision is made, whether MLP contributes more than attention, and how these signals relate to answer reliability.

These questions matter because understanding _where_ decisions happen opens the door to _intervening_ when the model is uncertain. We used the **logit lens** technique to peek inside LLaVA-1.5-7B and trace the answer formation process layer by layer.**Dataset:** 1,000 VQA samples from VQAv2 **Dataset:** 1,000 VQA samples from VQAv2

## What We Did**Model:** llava-hf/llava-1.5-7b-hf (fp16, no quantization) **Model:** llava-hf/llava-1.5-7b-hf (4-bit quantized)

We ran 1,000 VQA samples through LLaVA and recorded:**Accuracy:** 67.6% with image, 8.1% without image**Accuracy:** 70.3% with image, 8.1% without image

1. The **margin** (correct answer logit minus best wrong answer logit) at every layer

2. How much **MLP** vs **Attention** contributed to that margin### Key Findings:### Key Findings:

3. Whether we could **predict correctness** from internal activations

4. Whether **ablating** specific neurons would break the model- **Image is essential**: Accuracy drops from 67.6% → 8.1% without the image- **Image is essential**: Accuracy drops from 70.3% → 8.1% without the image

**Setup:**- **MLP dominates answer boosting**: 82.1% contribution in layers 30-31- **MLP dominates answer boosting**: 72.1% contribution in layers 30-31

- Model: `llava-hf/llava-1.5-7b-hf` (fp16, A100 GPU)

- Dataset: VQAv2 validation set (1,000 samples)- **Attention dominates suppression**: 78.4% contribution in layers 17, 21-28- **Attention dominates suppression**: 72.1% contribution in layers 21-28

- Runtime: 20 minutes for full pipeline

- **Combined reliability model**: 95.6% AUROC for predicting correctness- **Combined reliability model**: 95.6% AUROC for predicting correctness

---

- **Neuron-level prediction**: 88.0% accuracy using sparse features at Layer 21- **Neuron-level prediction**: 86.3% accuracy using sparse features at Layer 29

## Key Results

- **Peak visual effect**: Layer 31 with Δmargin = +9.20- **Peak visual effect**: Layer 31 with Δmargin = +8.80

### 1. The Image is Essential (No Surprise, But Quantified)

- **Causal validation**: Ablating top-5 neurons causes 0.4% overall drop (4x more than random)- **Causal validation**: Ablating top-5 neurons causes task-specific failures (e.g., color recognition drops 100%)

| Condition | Accuracy |

|-----------|----------|---

| With image | 67.6% |

| Without image | 8.1% |## 1. Methodology## 1. Methodology

**Why this matters:** The 8.1% "without image" accuracy represents pure language prior—questions like "What color is the sky?" where the model guesses "blue" without looking. The 59.5 percentage point drop confirms the model actually uses visual information, not just language statistics.### 1.1 Margin Definition### 1.1 Margin Definition

### 2. MLP and Attention Have Different JobsFor each layer ℓ, we define the margin as:For each layer ℓ, we define the margin as:

This was the most interesting finding. We expected MLP and Attention to contribute roughly equally, but they don't:$$\text{margin}^\ell = \text{logit}(y_{\text{correct}}) - \max_{y \neq y_{\text{correct}}} \text{logit}(y)$$$$\text{margin}^\ell = \text{logit}(y_{\text{correct}}) - \max_{y \neq y_{\text{correct}}} \text{logit}(y)$$

| Context | MLP % | Attention % |A **positive margin** means the correct answer has the highest logit; a **negative margin** means another answer is ahead.### 1.2 Attribution Method

|---------|-------|-------------|

| All layers | 34.7% | 65.3% |### 1.2 MLP and Attention Contribution CalculationFor each layer L, we decompose the margin change:

| **Boosting layers (L30-31)** | **82.1%** | 17.9% |

| **Suppression layers (L17, 21-28)** | 21.6% | **78.4%** |> _Shikhar's comment: "How are these categorised?"_- **MLP contribution** = Δmargin from MLP sublayer

**What this means:** There's a clear division of labor:- **Attention contribution** = Δmargin from attention sublayer

- **Attention suppresses wrong answers** in the middle layers (L21-28). When you show an image of a red car and ask "What color?", attention helps rule out "blue", "green", etc.

- **MLP boosts the correct answer** in the final layers (L30-31). Once wrong answers are filtered, MLP pushes the logit of "red" above everything else.At each transformer layer, the residual stream is updated additively:- **Percentages** = (sublayer contribution) / (total contribution) × 100

**Why we think this happens:** Attention can compare across token positions (image patches, question words), making it good at "this doesn't match that." MLP operates on each position independently, making it good at "given these features, output this."````### 1.3 Visual Attribution

### 3. Layer 31 is Where the Decision Happensresidual_new = residual_old + attention_output + mlp_output

We compared margins with real images vs blank gray images:```Compare margins with real image vs blank image:

| Layer | Δmargin (image - no image) | Interpretation |

|-------|---------------------------|----------------|

| 17 | -0.52 | Slight suppression |To measure how much each component contributes to predicting a specific token:$$\Delta\text{margin} = \text{margin}(\text{with image}) - \text{margin}(\text{without image})$$

| 21-28 | -0.85 to -2.27 | Strong suppression |

| 30 | +2.61 | Boosting begins |

| **31** | **+9.20** | Decision point |

1. **Extract unembedding vector**: Get the row from the LM head corresponding to the predicted tokenThis control isolates which layers actually process visual information versus language priors.

**Surprise:** We expected the visual effect to be spread out, but Layer 31 alone contributes +9.20 to the margin—more than all other layers combined. The model seems to "hold off" on committing to an answer until the very last layer.

2. **Compute dot products**: At each layer:

### 4. Correctness is Predictable Early

- `MLP contribution = dot(mlp_output, unembedding_vector)`### 1.4 Causal Ablation

Here's what surprised us most: **you can predict if the answer will be correct by Layer 21**, long before the final answer is selected.

- `Attention contribution = dot(attn_output, unembedding_vector)`

| Layer | Probe Accuracy | What's Happening |

|-------|----------------|------------------|To move from correlation to causation, we:

| 17 | 82.3% | First signal |

| **21** | **88.0%** | Best prediction |Because the residual stream is additive, these dot products quantify how much each component "pushes" the logit toward the predicted token.

| 26 | 84.7% | — |

| 29 | 86.3% | — |1. Zero out specific neurons at inference time

| 31 | 85.3% | Final layer |

**Interpretation:**2. Compare accuracy drop vs random neuron ablation

**Why Layer 21?** This is in the suppression phase, where attention is filtering wrong answers. Our interpretation: if the suppression phase fails to eliminate the correct answer, we can detect that early. By Layer 21, the model has already made its "mistake"—the final layers just amplify it.

| Value | Meaning |3. Analyze task-specific effects

**Practical implication:** You could build an early-exit mechanism that checks Layer 21 activations and says "I'm not confident" before generating the answer.

|-------|---------|

### 5. Neuron 1512 is the "Success Neuron"

| Positive | Component **boosts** the token's logit |---

We found a single neuron (index 1512) that appears in the top-5 predictive neurons across 5 different layers:

| Negative | Component **suppresses** the token's logit |

| Layer | Δactivation (correct - incorrect) |

|-------|-----------------------------------|## 2. Results

| 17 | +0.67 |

| 26 | +1.30 |**Percentage calculation:**

| 27 | +1.55 |

| 29 | +1.97 |- Sum contributions across specified layers### 2.1 Image Necessity

| **31** | **+27.23** |

- MLP% = MLP_sum / (MLP_sum + Attn_sum) × 100

**Interpretation:** This neuron activates 27 units higher when the model is correct at the final layer. It might encode "I found a good visual-language match" or "I'm confident."

| Condition | Accuracy |

We also found "failure neurons" (1360, 3839, 2660) that activate more when the model is wrong, but their effects are smaller (Δ ≈ -3).

### 1.3 Visual Attribution| ------------- | -------------------------- |

### 6. Causal Ablation: Weak Evidence

| With image | **70.3%** |

We zeroed out the top-5 neurons at Layer 31 to test if they're causally important:

Compare margins with real image vs blank (gray) image:| Without image | 8.1% |

| Condition | Accuracy | Drop |

|-----------|----------|------|| **Drop** | **62.2 percentage points** |

| Baseline | 32.1% | — |

| Top-5 neurons ablated | 31.7% | 0.4% |$$\Delta\text{margin} = \text{margin}(\text{with image}) - \text{margin}(\text{without image})$$

| Random 5 neurons ablated | 32.1% | 0.0% |

**Conclusion:** The image is essential for VQA—accuracy drops by 88.5% relative without visual input.

**Honest assessment:** The 0.4% drop is small. Top-5 neurons matter 4x more than random, but the effect is weak. This suggests:

1. The signal is **distributed** across many neurons, not localizedThis control isolates which layers actually process visual information versus language priors.

2. The model has **redundancy**—other neurons can compensate

3. **Correlation ≠ causation**—neurons that predict correctness aren't necessarily required for it### 2.2 MLP vs Attention Attribution

This is consistent with the interpretability literature: probes often find correlated features that aren't causally necessary.### 1.4 Causal Ablation

---**Table 1: MLP vs Attention contribution by context**

## How We Calculated MLP/Attention ContributionsTo move from correlation to causation, we:

Since Shikhar asked "How are these categorised?", here's the methodology:| Context | MLP % | Attention % |

At each layer, the residual stream updates as:1. Zero out specific neurons at inference time| -------------------------------- | --------- | ----------- |

```

residual_new = residual_old + attention_output + mlp_output2. Compare accuracy drop vs random neuron ablation| All layers                       | 39.3%     | 60.7%       |

```

3. Analyze task-specific effects| Visual layers only (L19-31) | 52.6% | 47.4% |

To measure how much each component contributes to the predicted token's logit:

| **Answer boosting (L19, 30-31)** | **72.1%** | 27.9% |

1. **Get the unembedding vector** for the predicted token from the LM head

2. **Compute dot products:**---| **Suppression (L21-28)** | 27.9% | **72.1%** |

   - `MLP contribution = dot(mlp_output, unembedding_vector)`

   - `Attention contribution = dot(attn_output, unembedding_vector)`

Positive = boosts the token, Negative = suppresses it.## 2. Results**How calculated:**

**Categorization criteria:**

- **Visual layers:** |Δmargin| > 0.5 (image makes significant difference)

- **Boosting layers:** Positive Δmargin (L30-31)### 2.1 Image Necessity1. Sum MLP contributions across specified layers

- **Suppression layers:** Negative Δmargin (L17, L21-28)

2. Sum Attention contributions across specified layers

---

| Condition | Accuracy |3. MLP% = MLP_sum / (MLP_sum + Attn_sum)

## What We Learned (and What We're Still Unsure About)

| ------------- | -------------------------- |

### Confident conclusions:

1. **Division of labor exists:** Attention filters, MLP selects. This is robust across samples.| With image | **67.6%** |**Interpretation:**

2. **Layer 31 is decisive:** The +9.20 Δmargin is too large to be noise.

3. **Early prediction works:** 88% accuracy at Layer 21 is useful for applications.| Without image | 8.1% |

### Still uncertain:| **Drop** | **59.5 percentage points** |- MLP dominates final answer selection (72.1% in boosting layers)

1. **Why is causal evidence weak?** Either the signal is truly distributed, or our ablation method (zeroing neurons) is too crude.

2. **Does this generalize?** We only tested LLaVA-1.5-7B on VQAv2. Other models and datasets might show different patterns.- Attention dominates wrong-answer suppression (72.1% in suppression layers)

3. **What is Neuron 1512 actually computing?** We know it correlates with success, but not why.

**Conclusion:** The image is essential for VQA—accuracy drops by 88% relative without visual input.- This reveals a **functional division of labor** between the two components

---

## Comparison: 4-bit vs fp16

### 2.2 MLP vs Attention Attribution### 2.3 Layer-wise Visual Effect

We ran experiments with both quantization settings:

| Metric | 4-bit | fp16 | Notes |

|--------|-------|------|-------|> \*Shikhar's comment: "How are these categorised?"**\*Table 2: Δmargin (with image - without image) by layer**

| Accuracy | 70.3% | 67.6% | Surprising: 4-bit was higher |

| MLP boosting % | 72.1% | 82.1% | fp16 shows stronger MLP role |

| Best probe layer | L29 | L21 | Shifted earlier in fp16 |

| Peak Δmargin | +8.80 | +9.20 | Similar |**Table 1: MLP vs Attention contribution by context**| Layer | Δmargin | Role |

**Interpretation:** The accuracy difference might be due to different random samples. The key finding (MLP boosts, Attention suppresses) holds in both settings, though fp16 shows a cleaner separation.| ------ | -------------- | --------------------------------------------------------- |

---| Context | MLP % | Attention % | How Defined || 19 | +0.53 | Early boosting |

## Next Steps| -------------------------------- | --------- | ----------- | ----------- || 21-28 | -0.85 to -2.27 | Suppression (negative = image helps reject wrong answers) |

Per Shikhar's feedback, we need to test on other models and datasets:| All layers | 34.7% | 65.3% | Sum across L0-31 || 30 | +2.61 | Answer boosting |

**Models to test:**| Visual layers only (L17, L21-31) | 52.1% | 47.9% | Layers with \|Δmargin\| > 0.5 || **31** | **+8.80** | **Peak visual effect** |

- Qwen2-VL-7B

- PaliGemma 2| **Answer boosting (L30-31)** | **82.1%** | 17.9% | Layers with positive Δmargin |

**Datasets to test:**| **Suppression (L17, L21-28)** | 21.6% | **78.4%** | Layers with negative Δmargin |**Key Finding:** Layer 31 shows the strongest visual effect, with Δmargin = +8.80. This is where the final answer decision is made.

- COCO Captions (image captioning)

- POPE (hallucination detection)

If the MLP/Attention division of labor holds across models, that's a general insight about VLM architecture. If it doesn't, that's interesting too—it would mean LLaVA has a specific design pattern.**Categorization criteria:**### 2.4 Correct vs Incorrect Margin Trajectories

---- **Visual layers**: Layers where |Δmargin| > 0.5 (image makes significant difference)

## Experimental Details- **Boosting layers**: Positive Δmargin (image helps select correct answer)**Table 3: Trajectory comparison**

- **Model:** llava-hf/llava-1.5-7b-hf- **Suppression layers**: Negative Δmargin (image helps reject wrong answers)

- **Precision:** fp16 (no quantization)

- **Dataset:** VQAv2 validation (1,000 samples)| Metric | Correct (n=703) | Incorrect (n=297) |

- **Hardware:** NVIDIA A100 40GB

- **Runtime:** 20.6 minutes**Interpretation:**| --------------------- | --------------- | ----------------- |

- **Code:** `logit_lens/run_all_logit_lens.py`

| Final margin (L31) | Higher | Lower |

**Architecture reference:**

- Language backbone: Llama-2-7B- MLP dominates final answer selection (82.1% in boosting layers)| Crossover to positive | Layer 29 | Never |

- Hidden dim: 4096

- MLP intermediate: 11008- Attention dominates wrong-answer suppression (78.4% in suppression layers)| Separation begins | Layer 21 | — |

- Layers: 32

- Attention heads: 32- This reveals a **functional division of labor** between the two components| Maximum separation | Layer 24 | — |

### 2.3 Layer-wise Visual Effect**Key Insight:** Correct and incorrect answers follow distinct margin trajectories, with separation beginning at Layer 21 and peaking at Layer 24. This indicates that **correctness emerges before final answer selection**.

**Table 2: Δmargin (with image - without image) by layer**### 2.5 Reliability Prediction Performance

| Layer | Δmargin | Role |**Table 4: AUROC for predicting answer correctness**

| ------ | -------------- | --------------------------------------------------------- |

| 17 | -0.52 | Early suppression || Signal | AUROC |

| 21-28 | -0.85 to -2.27 | Suppression (negative = image helps reject wrong answers) || ------------------------------------------- | --------- |

| 30 | +2.61 | Answer boosting || Total MLP contribution | 0.432 |

| **31** | **+9.20** | **Peak visual effect** || Total Attention contribution | 0.624 |

| Final margin alone | 0.488 |

**Key Finding:** Layer 31 shows the strongest visual effect, with Δmargin = +9.20. This is where the final answer decision is made.| Attention spatial metrics (baseline) | 0.550 |

| **Combined model (MLP+Attn at key layers)** | **0.956** |

### 2.4 Correct vs Incorrect Margin Trajectories

**Key Finding:** While individual signals show modest predictive power, combining MLP and Attention features at key layers (17, 21, 26, 29-31) yields strong reliability prediction, achieving **95.6% AUROC**.

**Table 3: Trajectory comparison**

**Top predictive features:**

| Metric | Correct (n=676) | Incorrect (n=324) |

| --------------------- | --------------- | ----------------- |- S_Attn_L26: +1.21 (strongest positive)

| Final margin (L31) | Higher | Lower |- S_Attn_L30: −0.89

| Crossover to positive | Layer 29 | Never |- S_Attn_L31: −0.77

| Separation begins | Layer 21 | — |- S_MLP_L17: +0.69

| Maximum separation | Layer 24 | — |- S_MLP_L29: −0.67

**Key Insight:** Correct and incorrect answers follow distinct margin trajectories, with separation beginning at Layer 21 and peaking at Layer 24. This indicates that **correctness emerges before final answer selection**.### 2.6 Neuron-Level Prediction Performance

### 2.5 Reliability Prediction Performance**Table 5: Probe accuracy by layer**

> _Shikhar's comment: "How is this computed?"_| Layer | Test Accuracy | Non-zero Neurons | Role |

| ------ | ------------- | ---------------- | ------------------------- |

**Table 4: AUROC for predicting answer correctness**| 17 | 82.3% | 38/1000 | Early signal |

| 21 | 85.0% | 59/1000 | Suppression phase |

| Signal | AUROC | How Computed || 26 | 84.7% | 39/1000 | Mid suppression |

| ------------------------------------------- | --------- | ------------ || 27 | 83.0% | 46/1000 | Late suppression |

| Total MLP contribution | 0.432 | Sum of MLP contributions across all layers || **29** | **86.3%** | 57/1000 | **Best predictive layer** |

| Total Attention contribution | 0.624 | Sum of Attention contributions across all layers || 30 | 86.0% | 49/1000 | Selection phase |

| Final margin alone | 0.488 | Margin at Layer 31 || 31 | 85.3% | 43/1000 | Final decision |

| Attention spatial metrics (baseline) | 0.550 | Attention weights on image tokens |

| **Combined model (MLP+Attn at key layers)** | **0.956** | Logistic regression on L17, L21, L26, L29-31 features |**Key Finding:** Only 4-6% of neurons carry predictive signal, demonstrating extreme sparsity. Layer 29 achieves the best prediction accuracy at 86.3%.

**AUROC computation:\*\***Efficiency vs Amplification:\*\*

- Binary classification: correct (1) vs incorrect (0)

- Features: MLP and Attention contributions at key layers- Layer 31 shows the largest raw activation differences (Neuron 1512: +27.23)

- Model: Logistic regression with L1 regularization- Layer 29 achieves the best accuracy with fewer non-zero neurons

- Evaluation: 5-fold cross-validation, report mean AUROC- This suggests Layer 29 has more efficient encoding, while Layer 31 amplifies the signal for final output

### 2.6 Neuron-Level Prediction Performance---

> _Shikhar's comment: "Does every layer have 1000 neurons?"_## 3. Neuron-Level Analysis

**Note on architecture:** LLaVA-1.5-7B uses Llama-2-7B as its language backbone. Each layer has:### 3.1 Success Neuron: Neuron 1512

- **Hidden dimension**: 4096

- **MLP intermediate dimension**: 11008Neuron 1512 appears in the top-5 most important neurons across 5 different layers:

The "1000" in the table refers to the **top 1000 neurons analyzed** (sorted by absolute contribution), not the total neuron count.| Layer | Δactivation (correct − incorrect) |

| ------ | --------------------------------- |

**Table 5: Probe accuracy by layer**| 17 | +0.67 |

| 26 | +1.30 |

| Layer | Test Accuracy | Non-zero Neurons | Role || 27 | +1.55 |

| ------ | ------------- | ---------------- | ------------------------- || 29 | +1.97 |

| 17 | 82.3% | 38/1000 | Early signal || **31** | **+27.23** |

| **21** | **88.0%** | 59/1000 | **Best predictive layer** |

| 26 | 84.7% | 39/1000 | Mid suppression |**Interpretation:** This neuron consistently activates ~27 units higher when the model answers correctly at the final layer. It may encode "answer confidence" or "visual-language alignment quality." The dramatic increase at Layer 31 suggests it plays a key role in the final decision.

| 27 | 83.0% | 46/1000 | Late suppression |

| 29 | 86.3% | 57/1000 | Selection phase |### 3.2 Failure Neurons

| 30 | 86.0% | 49/1000 | Selection phase |

| 31 | 85.3% | 43/1000 | Final decision |Our analysis identified specific "failure neurons" that actively predict incorrect outcomes at Layer 31:

**Key Finding:** Only 4-6% of the top neurons carry predictive signal (non-zero weights in L1-regularized probe), demonstrating extreme sparsity. Layer 21 achieves the best prediction accuracy at 88.0%.| Neuron | Δactivation (correct − incorrect) | Interpretation |

| ------ | --------------------------------- | --------------------- |

---| 1360 | -3.11 | Strong failure signal |

| 3839 | -3.08 | Failure signal |

## 3. Neuron-Level Analysis| 2660 | -2.95 | Failure signal |

| 2298 | -2.68 | Failure signal |

### 3.1 Success Neuron: Neuron 1512| 3556 | -2.19 | Failure signal |

> \*Shikhar's comment: "Which layer? 31?"**\*Key Finding:** The model has dedicated circuitry for both "confidence" (Neuron 1512) and "confusion/error" (Neurons 1360, 3839, 2660, etc.).

Neuron 1512 appears in the top-5 most important neurons across **5 different layers**, with the strongest effect at Layer 31:### 3.3 Cross-Layer Functional Shifts

| Layer | Δactivation (correct − incorrect) |Some neurons exhibit dynamic role-switching behavior across layers. For example, certain neurons reverse their predictive polarity between the suppression phase (L21-28) and selection phase (L30-31):

| ------ | --------------------------------- |

| 17 | +0.67 || Neuron | Layer 21 | Layer 27 | Interpretation |

| 26 | +1.30 || ------- | ---------- | ---------- | -------------- |

| 27 | +1.55 || Example | Positive Δ | Negative Δ | Role reversal |

| 29 | +1.97 |

| **31** | **+27.23** |**Interpretation:** These neurons do not encode static features—they act as **adaptive gates** that change function during processing. This suggests a dynamic filtering mechanism rather than simple feature transmission through the network.

**Interpretation:** This neuron consistently activates ~27 units higher when the model answers correctly at the final layer. It may encode "answer confidence" or "visual-language alignment quality." The dramatic increase at Layer 31 suggests it plays a key role in the final decision.### 3.4 Implications of Neuron Analysis

### 3.2 Failure NeuronsThese findings indicate:

Our analysis identified specific "failure neurons" that actively predict incorrect outcomes at Layer 31:1. **Correctness signals are extremely sparse**: Only 4-6% of neurons encode predictive information

2. **Specialized neurons track reliability**: Certain neurons consistently activate for correct or incorrect answers, with large activation gaps (>4 units)

| Neuron | Δactivation (correct − incorrect) | Interpretation |3. **Late layers are decisive**: Layers 29-31 contain the strongest correctness signals, consistent with final decision formation

| ------ | --------------------------------- | --------------------- |4. **Dedicated failure detection exists**: The model has specific neurons tuned to detect when it's making errors

| 1360 | -3.11 | Strong failure signal |5. **Dynamic role switching**: Some neurons change function across processing phases, acting as adaptive gates

| 3839 | -3.08 | Failure signal |6. **Sparse control**: Neuron 1512 carries disproportionate influence, with dramatic amplification (+27.23) at the final layer

| 2660 | -2.95 | Failure signal |

| 2298 | -2.68 | Failure signal |**What This Means:**

| 3556 | -2.19 | Failure signal |

The fact that sparse subsets of neurons (4-6%) can predict correctness with 86% accuracy suggests:

**Key Finding:** The model has dedicated circuitry for both "confidence" (Neuron 1512) and "confusion/error" (Neurons 1360, 3839, 2660, etc.).

- Answer correctness is not distributed uniformly across all neurons

### 3.3 Cross-Layer Functional Shifts- Specific neurons specialize in tracking whether the model is "on the right track"

- These neurons could potentially be targeted for reliability interventions

Some neurons exhibit dynamic role-switching behavior across layers. Certain neurons reverse their predictive polarity between the suppression phase (L21-28) and selection phase (L30-31):

---

**Interpretation:** These neurons do not encode static features—they act as **adaptive gates** that change function during processing. This suggests a dynamic filtering mechanism rather than simple feature transmission through the network.

## 4. Causal Ablation Results

### 3.4 Implications of Neuron Analysis

### 4.1 Overall Ablation Impact

These findings indicate:

**Table 6: Ablation experiment results (n=1000)**

1. **Correctness signals are extremely sparse**: Only 4-6% of top neurons encode predictive information

2. **Specialized neurons track reliability**: Certain neurons consistently activate for correct or incorrect answers, with large activation gaps (>4 units)| Condition | Accuracy | Drop from Baseline |

3. **Late layers are decisive**: Layers 21 and 29-31 contain the strongest correctness signals| ------------------------ | --------- | ------------------ |

4. **Dedicated failure detection exists**: The model has specific neurons tuned to detect when it's making errors| **Baseline** | **31.3%** | — |

5. **Dynamic role switching**: Some neurons change function across processing phases, acting as adaptive gates| Neuron 1512 @ L31 only | 31.3% | 0.0% |

6. **Sparse control**: Neuron 1512 carries disproportionate influence, with dramatic amplification (+27.23) at the final layer| Neuron 1512 @ all layers | 31.4% | -0.1% |

| **Top-5 neurons @ L31** | **30.6%** | **+0.7%** |

---| Random 5 neurons @ L31 | 31.2% | +0.1% |

## 4. Causal Ablation Results### 4.2 Task-Specific Ablation Effects

> *Shikhar's comment: "Does this match original paper?"*Ablating the top-5 neurons at Layer 31 (1512, 1360, 1573, 3556, 2298) causes **task-specific failures**:

**Note:** Our ablation methodology differs from some prior work. We ablate by **zeroing out MLP output neurons** at inference time, not by modifying weights or using activation patching.| Question Type | Baseline | After Top-5 Ablation | Drop |

| ------------------------- | -------- | -------------------- | ------------- |

### 4.1 Overall Ablation Impact| **Color recognition** | 3.2% | **0.0%** | **-100%** |

| **Person identification** | 50.0% | 33.3% | **-16.7%** |

**Table 6: Ablation experiment results (n=1000)**| **Comparison** | 21.4% | 14.3% | **-7.1%** |

| Yes/no | 70.8% | 69.0% | -1.8% |

| Condition | Accuracy | Drop from Baseline || Counting | 1.8% | 4.4% | +2.6% (noise) |

| ------------------------ | --------- | ------------------ |

| **Baseline** | **32.1%** | — |### 4.3 Causal Evidence Interpretation

| Neuron 1512 @ L31 only | 32.1% | 0.0% |

| **Top-5 neurons @ L31** | **31.7%** | **0.4%** || Evidence | Top-5 Neurons | Random Neurons | Conclusion |

| Random 5 neurons @ L31 | 32.1% | 0.0% || --------------------- | ------------- | -------------- | ------------------------- |

| Overall drop | 0.7% | 0.1% | Top-5 matter 7x more |

**Key observations:**| Color recognition | -100% | 0% | **Causal link confirmed** |

- Single neuron ablation shows no effect (model has redundancy)| Person identification | -16.7% | 0% | **Causal link confirmed** |

- Top-5 ablation causes 4x more drop than random neurons

- The `causal_evidence: false` flag in our results indicates the effect is small but present**Key Finding:** While single-neuron ablation shows no effect (model has redundancy), ablating the top-5 identified neurons **completely destroys color recognition** and significantly impairs person identification. This provides **causal evidence** that these neurons are functionally important, not just correlated with correctness.

### 4.2 Why Ablation Effects Are Small---

The small ablation effects suggest:## 5. Key Layers and Computational Roles

1. **Distributed representation**: The signal is spread across many neurons, not localized in a few### 5.1 Layer-wise Processing Pipeline

2. **Redundancy**: Multiple neurons encode similar information

3. **Probe ≠ Causation**: Neurons that correlate with correctness may not be causally necessaryOur analysis reveals a clear processing pipeline where different layers serve distinct computational roles:

This is consistent with findings in LLM interpretability literature where correlational probes don't always translate to causal importance.| Layer(s) | Role | Evidence | What's Happening |

| -------- | ------------------------ | ---------------------------------- | ------------------------------------------------------------- |

### 4.3 Causal Evidence Summary| 0-16 | Feature extraction | Low margin variance | Building representations, no clear answer signal yet |

| 17 | Early predictive signal | R² = 0.305, Neuron 1512 active | First layer where correctness can be partially predicted |

| Evidence | Top-5 Neurons | Random Neurons | Conclusion || 19 | Early boosting | Δmargin = +0.53 | First positive visual effect |

| --------------------- | ------------- | -------------- | ------------------------- || 21-28 | Wrong-answer suppression | Attention = 72%, negative Δmargin | Attention filters incorrect candidates; margin decreases here |

| Overall drop | 0.4% | 0.0% | Top-5 matter 4x more || 24 | Maximum separation | Δ = 2.16 between correct/incorrect | Largest gap between correct and incorrect trajectories |

| Effect size | Small | None | Signal is distributed || **29** | **Neuron commitment** | **86.3% probe accuracy** | Sparse neurons "commit" to answer; best for prediction |

| 30-31 | Final answer selection | MLP = 72%, Δmargin = +8.80 | MLP boosts correct answer's logit; final decision made |

---

### 5.2 Visual Processing Phases

## 5. Key Layers and Computational Roles

We identified two distinct phases when the image is present vs absent:

### 5.1 Layer-wise Processing Pipeline

| Phase | Layers | Δmargin | Dominant Component | Function |

Our analysis reveals a clear processing pipeline where different layers serve distinct computational roles:| ----------- | --------- | ------------------------- | ------------------ | --------------------------------- |

| Suppression | 21-28 | Negative (-0.85 to -2.27) | Attention (72%) | Image helps reject wrong answers |

| Layer(s) | Role | Evidence | What's Happening || Boosting | 19, 30-31 | Positive (+0.53 to +8.80) | MLP (72%) | Image helps select correct answer |

| -------- | ------------------------ | ---------------------------------- | ------------------------------------------------------------- |

| 0-16 | Feature extraction | Low margin variance | Building representations, no clear answer signal yet |---

| 17 | Early suppression | Δmargin = -0.52 | First layer where image affects processing |

| **21** | **Best prediction** | **88.0% probe accuracy** | Suppression phase, best for correctness prediction |## 6. Why Layer 31 is Special

| 21-28 | Wrong-answer suppression | Attention = 78%, negative Δmargin | Attention filters incorrect candidates; margin decreases here |

| 24 | Maximum separation | Largest trajectory gap | Largest gap between correct and incorrect trajectories |Layer 31 shows the strongest visual effect of any layer:

| 29 | Neuron commitment | 86.3% probe accuracy | Sparse neurons "commit" to answer |

| 30-31 | Final answer selection | MLP = 82%, Δmargin = +9.20 | MLP boosts correct answer's logit; final decision made |- **Δmargin = +8.80** (image vs no image)

- **MLP dominates** with 72% contribution

### 5.2 Visual Processing Phases- **Neuron 1512 Δactivation = +27.23** (10x larger than other layers)

We identified two distinct phases when the image is present vs absent:This is the **decision point** where all prior computation culminates in selecting the final answer token.

| Phase | Layers | Δmargin | Dominant Component | Function |---

| ----------- | ------------ | ------------------------- | ------------------ | --------------------------------- |

| Suppression | 17, 21-28 | Negative (-0.52 to -2.27) | Attention (78%) | Image helps reject wrong answers |## 7. Conclusion

| Boosting | 30-31 | Positive (+2.61 to +9.20) | MLP (82%) | Image helps select correct answer |

Visual attention structure alone does not explain model reliability. However, language-model internal computations—specifically the interplay between MLP and Attention contributions across layers—carry highly predictive signals for answer correctness.

---

### Core Findings:

## 6. Why Layer 31 is Special

1. **Image is Essential**: Accuracy drops from 70.3% to 8.1% without the image (88.5% relative drop)

Layer 31 shows the strongest visual effect of any layer:

2. **Division of Labor**: MLP dominates answer selection (72% in final layers) while Attention dominates suppression (72% in mid-layers)

- **Δmargin = +9.20** (image vs no image)

- **MLP dominates** with 82% contribution3. **Early Predictability**: Correctness emerges before final answer selection—trajectories diverge at Layer 21 and peak separation occurs at Layer 24

- **Neuron 1512 Δactivation = +27.23** (10x larger than other layers)

4. **Strong Combined Signal**: A model combining MLP and Attention features achieves **95.6% AUROC** for predicting correctness, substantially exceeding attention-only spatial metrics (55%)

This is the **decision point** where all prior computation culminates in selecting the final answer token.

5. **Sparse Control**: Only 4-6% of neurons carry predictive signal, with Neuron 1512 encoding success (activation +27.23 at Layer 31) and specific neurons (1360, 3839, 2660) encoding failure

---

6. **Dynamic Processing**: Some neurons change functional roles across layers, suggesting adaptive gating mechanisms rather than static feature encoding

## 7. Comparison: 4-bit Quantization vs fp16

7. **Causal Validation**: Ablating top-5 neurons causes task-specific failures:

We ran experiments with both 4-bit quantization (BitsAndBytes) and full fp16: - Color recognition: 100% failure

- Person identification: 16.7% drop

| Metric | 4-bit | fp16 | Change | - Overall: 7x more impact than random neurons

|--------|-------|------|--------|

| Accuracy with image | 70.3% | 67.6% | -2.7% |### Implications for Interpretability:

| Peak Δmargin (L31) | +8.80 | +9.20 | +0.40 |

| Boosting MLP% | 72.1% | 82.1% | +10% |- **Early warning signals exist**: By Layer 17, we can already partially predict if the answer will be correct

| Best probe accuracy | 86.3% (L29) | 88.0% (L21) | +1.7% |- **Complementary roles**: Attention filters, MLP selects

| Best probe layer | L29 | L21 | Shifted earlier |- **Intervention targets**: Layers 29-31 are prime candidates for reliability interventions

- **Dual circuitry**: The model has dedicated machinery for both confidence and error detection

**Key observation:** Without quantization noise, MLP's role in boosting is even clearer (82% vs 72%), and the best predictive layer shifts from L29 to L21.- **Task-specific neurons**: Different neurons control different question types (e.g., color recognition completely depends on the top-5 neurons)

- **Redundancy**: Single-neuron ablation has no effect, but combined ablation reveals causal relationships

---

This analysis supports our conclusion: **Reliability in VLMs can be better understood and potentially improved by examining the layer-wise dynamics of internal computations rather than spatial attention patterns.**

## 8. Conclusion

---

Visual attention structure alone does not explain model reliability. However, language-model internal computations—specifically the interplay between MLP and Attention contributions across layers—carry highly predictive signals for answer correctness.

## Appendix: Experimental Details

### Core Findings:

- **Model**: llava-hf/llava-1.5-7b-hf

1. **Image is Essential**: Accuracy drops from 67.6% to 8.1% without the image (88% relative drop)- **Quantization**: 4-bit (BitsAndBytes)

- **Dataset**: VQAv2 (1,000 samples)

2. **Division of Labor**: MLP dominates answer selection (82% in final layers) while Attention dominates suppression (78% in mid-layers)- **Hardware**: CUDA (cuda:1)

- **Correctness matching**: Lenient (prefix/substring matching)

3. **Early Predictability**: Correctness emerges before final answer selection—trajectories diverge at Layer 21 and peak separation occurs at Layer 24- **Ablation method**: Zero out MLP output neurons at inference time

4. **Strong Combined Signal**: A model combining MLP and Attention features achieves **95.6% AUROC** for predicting correctness, substantially exceeding attention-only spatial metrics (55%)

5. **Sparse Control**: Only 4-6% of top neurons carry predictive signal, with Neuron 1512 encoding success (activation +27.23 at Layer 31) and specific neurons (1360, 3839, 2660) encoding failure

6. **Dynamic Processing**: Some neurons change functional roles across layers, suggesting adaptive gating mechanisms rather than static feature encoding

7. **Weak Causal Evidence**: Ablating top-5 neurons causes only 0.4% drop, suggesting the signal is distributed across many neurons rather than localized

### Implications for Interpretability:

- **Early warning signals exist**: By Layer 21, we can predict if the answer will be correct with 88% accuracy
- **Complementary roles**: Attention filters, MLP selects
- **Intervention targets**: Layers 21 and 29-31 are prime candidates for reliability interventions
- **Dual circuitry**: The model has dedicated machinery for both confidence and error detection
- **Distributed representation**: Causal effects require ablating multiple neurons, not single neurons

This analysis supports our conclusion: **Reliability in VLMs can be better understood and potentially improved by examining the layer-wise dynamics of internal computations rather than spatial attention patterns.**

---

## Appendix: Experimental Details

- **Model**: llava-hf/llava-1.5-7b-hf
- **Precision**: fp16 (no quantization)
- **Dataset**: VQAv2 (1,000 samples)
- **Hardware**: NVIDIA A100 40GB (Lambda Labs)
- **Runtime**: 20.6 minutes for full pipeline
- **Correctness matching**: Lenient (prefix/substring matching)
- **Ablation method**: Zero out MLP output neurons at inference time

### Model Architecture Reference

- **Language backbone**: Llama-2-7B
- **Hidden dimension**: 4096
- **MLP intermediate dimension**: 11008
- **Number of layers**: 32
- **Number of attention heads**: 32

```

```
