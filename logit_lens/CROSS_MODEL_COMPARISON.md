# Cross-Model Logit Lens Analysis - 1000 Samples

## Summary

This document presents a cross-model comparison of logit lens analysis across three Vision-Language Models (VLMs), each with different architectures and numbers of transformer layers.

## Models Analyzed

| Model | Architecture | Total Layers | Parameters |
|-------|-------------|--------------|------------|
| LLaVA-1.5-7B | LLaMA-2 based | 32 | 7B |
| Qwen2-VL-7B | Qwen2 based | 28 | 7B |
| PaliGemma-3B | Gemma based | 18 | 3B |

## Main Results (1000 samples)

| Model | Acc WITH Image | Acc WITHOUT Image | Delta | Peak Layer | Peak Layer % |
|-------|----------------|-------------------|-------|------------|--------------|
| LLaVA-1.5-7B | 67.6% | 8.1% | 59.5% | 31 (of 32) | 97% |
| Qwen2-VL-7B | 81.1% | 13.5% | 67.6% | 27 (of 28) | 96% |
| PaliGemma-3B | 67.5% | 0.0% | 67.5% | 14 (of 18) | 78% |

## Key Findings

### 1. Universal Final-Layer Decision Pattern
All three models show the same pattern: **answer decisions are made in the final 1-3 layers**.

- **LLaVA-7B**: Peak at layer 31 (second-to-last), with layer 30-31 showing dramatic margin increase
- **Qwen2-VL**: Peak at layer 27 (second-to-last), consistent steep climb in final layers
- **PaliGemma**: Peak at layer 14 (layer 15 of 18), slightly earlier but still in final 25%

### 2. Image Importance Varies by Architecture
- **PaliGemma**: 0% accuracy without images - completely dependent on visual input
- **Qwen2-VL**: 13.5% without images - some text-based reasoning ability
- **LLaVA-7B**: 8.1% without images - minimal text-only capability

### 3. Qwen2-VL Shows Highest Performance
- 81.1% accuracy is notably higher than both LLaVA (67.6%) and PaliGemma (67.5%)
- Suggests better visual-language integration despite similar layer count patterns

### 4. Layer Depth vs Pattern Consistency
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
```

### Qwen2-VL (28 layers)  
```
Early layers (0-15): Negative margins, -8 to -16
Middle layers (16-24): Still negative but improving, -13 to -7
Final layers (25-27): Sharp rise from -7 → -3 → +3
```

### PaliGemma (18 layers)
```
Early layers (0-10): Very negative margins, -700 → -40 (normalizing)
Middle layers (11-14): Improving, -45 to -30
Final layers (15-17): Rise from -17 → -0.9 → +1
```

## Implications

1. **Architecture-Independent Pattern**: The late-layer decision phenomenon is not specific to LLaVA's architecture—it's a fundamental property of how VLMs process multimodal information.

2. **Intervention Target**: For any VLM intervention research, the final 1-3 layers are the key targets regardless of model architecture.

3. **Visual Information Integration**: The visual information remains in a "distributed" state until the final layers, where it's integrated with language processing to produce the answer.

## Data Files

- `compiled_results_1000.json` - LLaVA-7B detailed 1000-sample results
- `qwen2_vl_results.json` - Qwen2-VL 1000-sample results  
- `paligemma_results.json` - PaliGemma 1000-sample results

## Methodology

Each model was tested on the same 1000 VQA samples:
1. **With image**: Full multimodal processing
2. **Without image**: Text-only (question + answer choices)

For each layer, we computed:
- Logit margin = logit(correct answer) - max(logit(incorrect answers))
- Positive margin = model would predict correctly at that layer
- Delta margin = margin_with_image - margin_without_image (measures visual contribution)
