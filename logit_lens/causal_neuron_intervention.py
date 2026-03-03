"""
CAUSAL NEURON INTERVENTION EXPERIMENTS

This script performs rigorous causal intervention experiments to test whether
the neurons identified by our sparse probes are *causally* involved in reliability,
not just correlated.

Experiments:
1. Ablation: Zero out top-k probe neurons → measure accuracy drop
2. Activation Patching: Replace neuron activations from correct→incorrect samples
3. Interchange Intervention: Swap activations between correct/incorrect pairs
4. Mean Ablation: Replace with dataset mean (more interpretable than zeroing)

If neurons are causally important, we should see:
- Ablating success neurons → accuracy drops
- Ablating failure neurons → accuracy increases (removes error signal)
- Patching from correct→incorrect should flip predictions

Author: LLaVA Probe Team
"""

import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import requests
from io import BytesIO
import os
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pandas as pd

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)


@dataclass
class InterventionResult:
    """Result of a single intervention experiment."""
    intervention_type: str  # 'ablate', 'patch', 'mean_ablate'
    target_neurons: List[int]
    target_layers: List[int]
    baseline_accuracy: float
    intervention_accuracy: float
    accuracy_delta: float
    n_flipped_correct_to_wrong: int
    n_flipped_wrong_to_correct: int
    n_samples: int


class CausalNeuronIntervenor:
    """Performs causal intervention experiments on identified neurons."""
    
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.device = next(model.parameters()).device
        
        # Get model structure
        self.lm = model.language_model
        if hasattr(self.lm, 'model') and hasattr(self.lm.model, 'layers'):
            self.layers = self.lm.model.layers
        elif hasattr(self.lm, 'layers'):
            self.layers = self.lm.layers
        else:
            raise ValueError("Cannot find transformer layers")
        
        self.n_layers = len(self.layers)
        self.hooks = []
        self.intervention_config = {}
        self.collected_activations = {}
        
    def _create_collection_hook(self, layer_idx: int):
        """Create hook to collect MLP activations."""
        def hook(module, input, output):
            # Store the activation for this layer
            self.collected_activations[layer_idx] = output.detach().clone()
            return output
        return hook
    
    def _create_ablation_hook(self, layer_idx: int, neurons: List[int], 
                               replacement_value: Optional[torch.Tensor] = None):
        """Create hook to ablate (zero or replace) specified neurons."""
        def hook(module, input, output):
            with torch.no_grad():
                for neuron_idx in neurons:
                    if neuron_idx < output.shape[-1]:
                        if replacement_value is not None:
                            # Mean ablation: replace with dataset mean
                            output[:, :, neuron_idx] = replacement_value[neuron_idx]
                        else:
                            # Zero ablation
                            output[:, :, neuron_idx] = 0
            return output
        return hook
    
    def _create_patch_hook(self, layer_idx: int, neurons: List[int],
                           source_activation: torch.Tensor):
        """Create hook to patch activations from a source sample."""
        def hook(module, input, output):
            with torch.no_grad():
                # Get the last token position (where prediction happens)
                last_pos = output.shape[1] - 1
                for neuron_idx in neurons:
                    if neuron_idx < output.shape[-1]:
                        # Patch from source activation
                        src_val = source_activation[:, -1, neuron_idx]
                        output[:, last_pos, neuron_idx] = src_val
            return output
        return hook
    
    def clear_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.collected_activations = {}
    
    def collect_activations(self, image, question, layers: List[int]) -> Dict[int, torch.Tensor]:
        """Run forward pass and collect activations at specified layers."""
        self.clear_hooks()
        self.collected_activations = {}
        
        # Set up collection hooks
        for layer_idx in layers:
            hook = self.layers[layer_idx].mlp.register_forward_hook(
                self._create_collection_hook(layer_idx)
            )
            self.hooks.append(hook)
        
        # Run inference
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                output_hidden_states=False
            )
        
        # Get collected activations
        activations = {k: v.cpu() for k, v in self.collected_activations.items()}
        self.clear_hooks()
        
        return activations
    
    def run_with_ablation(self, image, question, layers: List[int], 
                          neurons: List[int], 
                          mean_values: Optional[Dict[int, torch.Tensor]] = None) -> str:
        """Run inference with specified neurons ablated."""
        self.clear_hooks()
        
        # Set up ablation hooks
        for layer_idx in layers:
            replacement = mean_values[layer_idx] if mean_values else None
            hook = self.layers[layer_idx].mlp.register_forward_hook(
                self._create_ablation_hook(layer_idx, neurons, replacement)
            )
            self.hooks.append(hook)
        
        # Run inference
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )
        
        answer = self.processor.decode(outputs[0], skip_special_tokens=True)
        answer = answer.split("ASSISTANT:")[-1].strip().lower()
        
        self.clear_hooks()
        return answer
    
    def run_with_patch(self, image, question, layers: List[int],
                       neurons: List[int], 
                       source_activations: Dict[int, torch.Tensor]) -> str:
        """Run inference with activations patched from source."""
        self.clear_hooks()
        
        # Set up patch hooks
        for layer_idx in layers:
            if layer_idx in source_activations:
                hook = self.layers[layer_idx].mlp.register_forward_hook(
                    self._create_patch_hook(layer_idx, neurons, 
                                           source_activations[layer_idx].to(self.device))
                )
                self.hooks.append(hook)
        
        # Run inference
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )
        
        answer = self.processor.decode(outputs[0], skip_special_tokens=True)
        answer = answer.split("ASSISTANT:")[-1].strip().lower()
        
        self.clear_hooks()
        return answer


def load_probe_neurons(results_path: str, top_k: int = 10) -> Tuple[List[int], List[int]]:
    """Load top success and failure neurons from probe results."""
    # This would load from your actual probe training results
    # For now, using the neurons identified in the paper
    success_neurons = [1512, 2847, 3201, 1089, 2456]  # Top positive weight
    failure_neurons = [1360, 892, 2156, 3045, 1678]   # Top negative weight
    return success_neurons[:top_k], failure_neurons[:top_k]


def run_ablation_experiment(
    model, processor, samples: List[Dict], 
    target_layers: List[int], target_neurons: List[int],
    use_mean_ablation: bool = False
) -> InterventionResult:
    """Run ablation experiment on specified neurons."""
    
    intervenor = CausalNeuronIntervenor(model, processor)
    
    baseline_correct = 0
    intervention_correct = 0
    flipped_correct_to_wrong = 0
    flipped_wrong_to_correct = 0
    
    # First pass: compute mean activations if needed
    mean_values = None
    if use_mean_ablation:
        print("Computing mean activations for mean ablation...")
        all_activations = {layer: [] for layer in target_layers}
        for sample in tqdm(samples[:100], desc="Computing means"):  # Use subset for speed
            image = sample['image']
            question = sample['question']
            acts = intervenor.collect_activations(image, question, target_layers)
            for layer in target_layers:
                if layer in acts:
                    all_activations[layer].append(acts[layer][:, -1, :])  # Last token
        
        mean_values = {}
        for layer in target_layers:
            if all_activations[layer]:
                stacked = torch.cat(all_activations[layer], dim=0)
                mean_values[layer] = stacked.mean(dim=0)
    
    # Main experiment
    for sample in tqdm(samples, desc="Running ablation"):
        image = sample['image']
        question = sample['question']
        ground_truth = sample['ground_truth'].lower()
        
        # Baseline
        baseline_answer = intervenor.run_with_ablation(
            image, question, [], []  # No ablation
        )
        baseline_is_correct = ground_truth in baseline_answer or baseline_answer in ground_truth
        
        # With ablation
        intervention_answer = intervenor.run_with_ablation(
            image, question, target_layers, target_neurons, mean_values
        )
        intervention_is_correct = ground_truth in intervention_answer or intervention_answer in ground_truth
        
        baseline_correct += int(baseline_is_correct)
        intervention_correct += int(intervention_is_correct)
        
        if baseline_is_correct and not intervention_is_correct:
            flipped_correct_to_wrong += 1
        elif not baseline_is_correct and intervention_is_correct:
            flipped_wrong_to_correct += 1
    
    n_samples = len(samples)
    baseline_acc = baseline_correct / n_samples
    intervention_acc = intervention_correct / n_samples
    
    return InterventionResult(
        intervention_type='mean_ablate' if use_mean_ablation else 'zero_ablate',
        target_neurons=target_neurons,
        target_layers=target_layers,
        baseline_accuracy=baseline_acc,
        intervention_accuracy=intervention_acc,
        accuracy_delta=intervention_acc - baseline_acc,
        n_flipped_correct_to_wrong=flipped_correct_to_wrong,
        n_flipped_wrong_to_correct=flipped_wrong_to_correct,
        n_samples=n_samples
    )


def run_activation_patching_experiment(
    model, processor, 
    correct_samples: List[Dict], incorrect_samples: List[Dict],
    target_layers: List[int], target_neurons: List[int]
) -> Dict:
    """
    Run activation patching: take activations from correct samples,
    patch them into incorrect samples, see if we can fix the errors.
    """
    intervenor = CausalNeuronIntervenor(model, processor)
    
    n_fixed = 0
    n_broken = 0
    n_total_patches = 0
    
    results = []
    
    # For each incorrect sample, try patching from a correct sample
    for i, incorrect_sample in enumerate(tqdm(incorrect_samples[:50], desc="Patching")):
        # Get a correct sample to patch from
        correct_sample = correct_samples[i % len(correct_samples)]
        
        # Collect activations from correct sample
        source_acts = intervenor.collect_activations(
            correct_sample['image'], 
            correct_sample['question'],
            target_layers
        )
        
        # Run incorrect sample with patched activations
        patched_answer = intervenor.run_with_patch(
            incorrect_sample['image'],
            incorrect_sample['question'],
            target_layers,
            target_neurons,
            source_acts
        )
        
        # Check if patching fixed the error
        gt = incorrect_sample['ground_truth'].lower()
        is_fixed = gt in patched_answer or patched_answer in gt
        
        if is_fixed:
            n_fixed += 1
        
        n_total_patches += 1
        
        results.append({
            'sample_id': i,
            'original_correct': False,
            'patched_correct': is_fixed,
            'ground_truth': gt,
            'patched_answer': patched_answer
        })
    
    return {
        'n_fixed': n_fixed,
        'n_total': n_total_patches,
        'fix_rate': n_fixed / n_total_patches if n_total_patches > 0 else 0,
        'details': results
    }


def main():
    """Run all causal intervention experiments."""
    
    print("=" * 60)
    print("CAUSAL NEURON INTERVENTION EXPERIMENTS")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        quantization_config=bnb_config,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    
    # Load samples
    print("Loading samples...")
    samples_path = os.path.join(PARENT_DIR, "topk", "paligemma_topk_5_results.csv")
    if os.path.exists(samples_path):
        df = pd.read_csv(samples_path)
    else:
        # Use backup path or generate samples
        print(f"Warning: {samples_path} not found")
        return
    
    # Load probe neurons
    success_neurons, failure_neurons = load_probe_neurons(None, top_k=5)
    
    # Target layers (last 5 layers where probes work best)
    target_layers = [27, 28, 29, 30, 31]
    
    print(f"\nTarget neurons:")
    print(f"  Success: {success_neurons}")
    print(f"  Failure: {failure_neurons}")
    print(f"Target layers: {target_layers}")
    
    # Prepare samples (would need image loading logic)
    # For now, this is a template
    
    results = {
        'experiments': [],
        'summary': {}
    }
    
    # Experiment 1: Ablate success neurons
    print("\n" + "=" * 40)
    print("Experiment 1: Ablate Success Neurons")
    print("Hypothesis: Accuracy should DROP")
    print("=" * 40)
    # result1 = run_ablation_experiment(...)
    
    # Experiment 2: Ablate failure neurons  
    print("\n" + "=" * 40)
    print("Experiment 2: Ablate Failure Neurons")
    print("Hypothesis: Accuracy should INCREASE")
    print("=" * 40)
    # result2 = run_ablation_experiment(...)
    
    # Experiment 3: Activation patching
    print("\n" + "=" * 40)
    print("Experiment 3: Activation Patching")
    print("Hypothesis: Can fix errors by patching from correct samples")
    print("=" * 40)
    # result3 = run_activation_patching_experiment(...)
    
    # Experiment 4: Random neuron baseline
    print("\n" + "=" * 40)
    print("Experiment 4: Random Neuron Ablation (Control)")
    print("Hypothesis: Minimal effect")
    print("=" * 40)
    # result4 = run_ablation_experiment(..., random_neurons)
    
    # Save results
    output_path = os.path.join(SCRIPT_DIR, "causal_intervention_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
