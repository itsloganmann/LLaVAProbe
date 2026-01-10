# Cross-Model Logit Lens Analysis - Full Analysis

## Summary

This document presents a **comprehensive cross-model comparison** of logit lens analysis across three Vision-Language Models (VLMs). We replicate the full LLaVA-7B analysis pipeline for Qwen2-VL and PaliGemma, including:
- **Step 1**: With/without image comparison (1000 samples)
- **Step 2**: MLP vs Attention attribution (200 samples)
- **Step 3**: Question type breakdown

## Models Analyzed

| Model | Architecture | Total Layers | Parameters | Language Backbone |
|-------|-------------|--------------|------------|-------------------|
| LLaVA-1.5-7B | LLaMA-2 based | 32 | 7B | Llama-2-7B |
| Qwen2-VL-7B | Qwen2 based | 28 | 7B | Qwen2-7B |
| PaliGemma-3B | Gemma based | 18 | 3B | Gemma-2B |

---

## Step 1: Image Necessity Results (1000 samples)

| Model | Acc WITH Image | Acc WITHOUT Image | Absolute Drop | Relative Drop | Peak Δmargin Layer |
|-------|----------------|-------------------|---------------|---------------|-------------------|
| LLaVA-1.5-7B | 67.6% | 8.1% | 59.5% | 88.0% | 31 (of 32) = 97% |
| Qwen2-VL-7B | 81.1% | 13.5% | 67.6% | 83.4% | 27 (of 28) = 96% |
| PaliGemma-3B | 67.5% | 0.0% | 67.5% | 100.0% | 14 (of 18) = 78% |

### Key Insight: Image Dependency Spectrum

```
Complete Dependency         Partial Text Robustness         Some Text Capability
       ←────────────────────────────────────────────────────────────────→
    PaliGemma (0%)              LLaVA (8.1%)                 Qwen2-VL (13.5%)
```

**PaliGemma's complete image dependency (0% without)** is remarkable—it literally cannot answer any VQA questions correctly without visual input. This suggests its training prioritized vision-language alignment over text-only reasoning.

---

## Step 2: MLP vs Attention Attribution (200 samples)

| Model | MLP % | Attention % | Dominant in Boosting | Dominant in Suppression |
|-------|-------|-------------|---------------------|-------------------------|
| LLaVA-1.5-7B | 57.7% | 42.3% | MLP | Attention |
| Qwen2-VL-7B | 48.1% | 51.9% | Balanced | Balanced |
| PaliGemma-3B | 47.6% | 52.4% | Balanced | Balanced |

### Cross-Model Pattern

LLaVA shows a clear **MLP-dominant** pattern (57.7%) for answer boosting, while Qwen2-VL and PaliGemma show **balanced** MLP/Attention contributions (~48-52%).

This suggests:
- **LLaVA**: Follows the "Attention filters, MLP selects" pattern most strongly
- **Qwen2-VL/PaliGemma**: More distributed processing where both components contribute equally

---

## Step 3: Question Type Analysis

### LLaVA-1.5-7B (Reference)

| Question Type | Count | Accuracy | Crossover Layer | Crossover % |
|---------------|-------|----------|-----------------|-------------|
| Object ID | 460 | 62.4% | 29 | 91% |
| Person ID | 351 | 66.4% | 29 | 91% |
| Yes/No | 54 | 85.2% | 29 | 91% |
| Location | 27 | 70.4% | 29 | 91% |
| Other | 108 | 72.2% | 29 | 91% |

### Qwen2-VL (New Results)

| Question Type | Count | Accuracy | Crossover Layer | Crossover % |
|---------------|-------|----------|-----------------|-------------|
| Object ID | 460 | 77.8% | 25 | 89% |
| Person ID | 351 | 81.2% | 25 | 89% |
| Yes/No | 54 | 96.3% | 26 | 93% |
| Location | 27 | 100.0% | 26 | 93% |
| Other | 108 | 79.6% | 25 | 89% |

### PaliGemma (New Results)

| Question Type | Count | Accuracy | Crossover Layer | Crossover % |
|---------------|-------|----------|-----------------|-------------|
| Object ID | 460 | 64.6% | 17 | 94% |
| Person ID | 351 | 69.2% | 17 | 94% |
| Yes/No | 54 | **100.0%** | 16 | 89% |
| Location | 27 | **100.0%** | 16 | 89% |
| Other | 108 | 50.0% | 17 | 94% |

### Question Type Patterns Across Models

**Yes/No Questions:**
- LLaVA: 85.2% → Qwen2-VL: 96.3% → PaliGemma: **100%**
- Binary questions are easiest across all models

**Location Questions:**
- LLaVA: 70.4% → Qwen2-VL: **100%** → PaliGemma: **100%**
- Both newer models achieve perfect location accuracy

**Object Identification:**
- LLaVA: 62.4% → Qwen2-VL: 77.8% → PaliGemma: 64.6%
- Most challenging category across all models

---

## Unified Main Results (1000 samples)

## Key Findings

### 1. Universal Final-Layer Decision Pattern
All three models show the same pattern: **answer decisions are made in the final 1-3 layers**.

- **LLaVA-7B**: Peak at layer 31 (second-to-last), with layer 30-31 showing dramatic margin increase
- **Qwen2-VL**: Peak at layer 27 (second-to-last), consistent steep climb in final layers
- **PaliGemma**: Peak at layer 14 (layer 15 of 18), earlier but still in final 25%

**Normalized Depth Analysis:**
| Model | Peak Layer Depth | Decision Phase |
|-------|-----------------|----------------|
| LLaVA-7B | 97% (31/32) | Final layer |
| Qwen2-VL | 96% (27/28) | Final layer |
| PaliGemma | 78% (14/18) | Late layers |

### 2. Image Dependency: A Spectrum
- **PaliGemma**: 0% accuracy without images - **complete visual dependency**
- **Qwen2-VL**: 13.5% without images - **some text reasoning capability**
- **LLaVA-7B**: 8.1% without images - **minimal text-only capability**

### 3. Performance Rankings

**Overall Accuracy (with image):**
1. Qwen2-VL: 81.1% ← **Best overall**
2. LLaVA-7B: 67.6%
3. PaliGemma: 67.5%

**Text Robustness (without image):**
1. Qwen2-VL: 13.5% ← **Most text-capable**
2. LLaVA-7B: 8.1%
3. PaliGemma: 0.0% ← **Pure vision model**

### 4. MLP vs Attention Division of Labor

| Model | MLP % | Attention % | Pattern |
|-------|-------|-------------|---------|
| LLaVA-7B | 57.7% | 42.3% | MLP-dominant |
| Qwen2-VL | 48.1% | 51.9% | Balanced |
| PaliGemma | 47.6% | 52.4% | Balanced |

**Interpretation:**
- LLaVA maintains the "Attention filters, MLP selects" pattern most strongly
- Qwen2-VL and PaliGemma have more distributed processing
- Despite architectural differences, all models stay within 48-58% MLP range

### 5. Layer Depth vs Pattern Consistency
Despite varying from 18 to 32 layers, all models show:
- Negative margins in early/middle layers (answer not yet determined)
- Sharp positive transition in final layers (answer crystallizes)
- Peak improvement within final 5-20% of layers

## Margin Progression Analysis

### LLaVA-1.5-7B (32 layers)
```
Early layers (0-20): Negative margins, fluctuating around -8 to -12
Middle layers (21-28): Still negative, averaging -10
Final layers (29-31): Dramatic rise from -8 → -3 → +6 → +9
Peak Δmargin: +9.57 at Layer 31
```

### Qwen2-VL (28 layers)  
```
Early layers (0-15): Negative margins, -8 to -16
Middle layers (16-24): Still negative but improving, -13 to -7
Final layers (25-27): Sharp rise from -7 → -3 → +3
Peak Δmargin: +2.16 at Layer 27
```

### PaliGemma (18 layers)
```
Early layers (0-10): Very negative margins, -700 → -40 (normalizing)
Middle layers (11-14): Improving, -45 to -30
Final layers (15-17): Rise from -17 → -0.9 → +1
Peak Δmargin: +9.57 at Layer 14
```

---

## Visual Layers Identified

Layers where |Δmargin| > 0.5 (image makes significant difference):

| Model | Visual Layers | Count | % of Model |
|-------|--------------|-------|------------|
| LLaVA-7B | 17, 21-31 | 12 | 38% |
| Qwen2-VL | 0-5, 7-27 | 27 | 96% |
| PaliGemma | 0-2, 4-6, 8-9, 11, 13-17 | 14 | 78% |

**Key Insight:** Qwen2-VL shows visual processing across nearly ALL layers (96%), while LLaVA concentrates visual effects in later layers (38%). PaliGemma falls in between.

## Implications

### 1. Architecture-Independent Patterns
The late-layer decision phenomenon is **universal across architectures**:
- All models make final decisions in the last 3-22% of layers
- All show negative→positive margin transitions
- MLP/Attention ratios remain in 48-58% range regardless of architecture

### 2. Visual Dependency as Design Choice
- **PaliGemma's 0% text-only accuracy** suggests design optimization for visual grounding over text reasoning
- **Qwen2-VL's 13.5%** indicates intentional text capability preservation
- Tradeoff: More text capability may mean less tight vision-language coupling

### 3. Intervention Targets
For reliability interventions across VLMs:

| Model | Target Layers | % Depth |
|-------|--------------|---------|
| LLaVA-7B | 29-31 | 91-97% |
| Qwen2-VL | 25-27 | 89-96% |
| PaliGemma | 14-17 | 78-94% |

### 4. Question Type Universals
Across all models:
- **Yes/No questions**: Highest accuracy (85-100%)
- **Location questions**: High accuracy in newer models (70-100%)
- **Object ID questions**: Consistently most challenging (62-78%)

---

## Data Files

**Full Analysis Results:**
- `qwen2_vl_full_results.json` - Qwen2-VL Steps 1-3 (1000 + 200 samples)
- `paligemma_full_results.json` - PaliGemma Steps 1-3 (1000 + 200 samples)

**Original Results:**
- `compiled_results_1000.json` - LLaVA-7B detailed 1000-sample results
- `qwen2_vl_results.json` - Qwen2-VL 1000-sample results (basic)
- `paligemma_results.json` - PaliGemma 1000-sample results (basic)

## Methodology

Each model was tested on the same 1000 VQA samples:

### Step 1: Image Necessity (1000 samples)
- **With image**: Full multimodal processing
- **Without image**: Text-only (question + answer choices)

For each layer, we computed:
- Logit margin = logit(correct answer) - max(logit(incorrect answers))
- Positive margin = model would predict correctly at that layer
- Delta margin = margin_with_image - margin_without_image (measures visual contribution)

### Step 2: MLP vs Attention Attribution (200 samples)
For each layer, decompose the margin contribution:
- **MLP contribution** = Δmargin from MLP sublayer  
- **Attention contribution** = Δmargin from attention sublayer
- **Percentages** = (sublayer contribution) / (total contribution) × 100

### Step 3: Question Type Analysis
Categorize questions and compute per-type:
- Accuracy
- Crossover layer (where margin becomes positive)
- Layer-wise margin trajectories

---

## Conclusion

This cross-model analysis reveals **universal patterns** in how VLMs process visual-language tasks:

1. **Final-layer decisions are universal** - All architectures crystallize answers in the last ~5% of layers
2. **MLP/Attention balance is consistent** - 48-58% MLP across all models
3. **Image dependency varies by design** - From 0% (PaliGemma) to 13.5% (Qwen2-VL) text-only capability
4. **Question types matter** - Binary and location questions are universally easier than object identification

These findings suggest that **interpretability techniques developed for one VLM architecture may transfer** to others, as the underlying computational patterns are remarkably similar despite different training procedures and model sizes.
