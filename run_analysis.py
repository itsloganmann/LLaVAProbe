"""
Main entry point for VLM Probe analysis.

Run this script to perform comparative analysis of Qwen3-VL and PaliGemma2.
Optimized for Google Colab with A100 GPU (80GB VRAM, 160GB DRAM).

Usage:
    python run_analysis.py --num_samples 100 --output_dir ./outputs
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any

import torch


def check_environment():
    """Check and print environment information."""
    print("=" * 60)
    print("VLM Probe: Environment Check")
    print("=" * 60)
    
    print(f"\nPython version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"Compute capability: {props.major}.{props.minor}")
    
    # Check for required packages
    required = ['transformers', 'PIL', 'numpy', 'matplotlib', 'scipy']
    missing = []
    for pkg in required:
        try:
            __import__(pkg if pkg != 'PIL' else 'PIL')
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"\n⚠️  Missing packages: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("\n✓ All required packages available")
    return True


def load_vqa_samples(csv_path: str, num_samples: int = 100) -> List[Dict[str, Any]]:
    """Load VQA samples from the results.csv file."""
    import csv
    
    samples = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= num_samples:
                break
            samples.append({
                'image': row['image_url'],
                'question': row['question'],
                'ground_truth': row['ground_truth'],
                'question_type': row.get('question_type', 'unknown'),
            })
    
    print(f"Loaded {len(samples)} samples from {csv_path}")
    return samples


def run_full_analysis(
    samples: List[Dict[str, Any]],
    output_dir: str,
    qwen_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    paligemma_model: str = "google/paligemma2-3b-pt-224",
):
    """Run the full comparative analysis pipeline."""
    from vlm_probe import (
        QwenVLRunner,
        PaliGemmaRunner,
        VLMAnalyzer,
        plot_attention_heatmap,
        plot_cross_model_comparison,
        plot_attention_entropy_analysis,
        plot_layer_attribution,
    )
    from vlm_probe.visualizations import create_publication_figure
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize runners
    print("\n" + "=" * 60)
    print("Initializing Models")
    print("=" * 60)
    
    print(f"\nLoading Qwen-VL: {qwen_model}")
    qwen = QwenVLRunner(model_id=qwen_model)
    
    print(f"\nLoading PaliGemma: {paligemma_model}")
    paligemma = PaliGemmaRunner(model_id=paligemma_model)
    
    # Initialize analyzer
    analyzer = VLMAnalyzer(
        qwen_runner=qwen,
        paligemma_runner=paligemma,
        output_dir=output_dir,
    )
    
    # Run batch analysis
    print("\n" + "=" * 60)
    print("Running Comparative Analysis")
    print("=" * 60)
    
    results = analyzer.run_batch_analysis(samples)
    
    # Save results
    analyzer.save_results("analysis_results.json")
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    # Collect data for visualizations
    qwen_entropies = []
    paligemma_entropies = []
    qwen_layer_entropies = {}
    paligemma_layer_entropies = {}
    
    for comp in analyzer.comparisons:
        if comp.qwen_attention:
            qwen_entropies.append(comp.qwen_attention.entropy)
        if comp.paligemma_attention:
            paligemma_entropies.append(comp.paligemma_attention.entropy)
    
    # Entropy analysis plot
    if qwen_entropies and paligemma_entropies:
        plot_attention_entropy_analysis(
            qwen_entropies=qwen_entropies,
            paligemma_entropies=paligemma_entropies,
            qwen_layer_entropies=qwen_layer_entropies,
            paligemma_layer_entropies=paligemma_layer_entropies,
            save_path=os.path.join(output_dir, "entropy_analysis.png"),
        )
    
    # Create sample comparison visualizations
    import requests
    from io import BytesIO
    from PIL import Image
    
    for i, (sample, comp) in enumerate(zip(samples[:5], analyzer.comparisons[:5])):
        try:
            # Load image
            if sample['image'].startswith('http'):
                response = requests.get(sample['image'], timeout=10)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(sample['image']).convert("RGB")
            
            # Get attention maps from comparison
            qwen_output = qwen.run(image, sample['question'])
            paligemma_output = paligemma.run(image, sample['question'])
            
            plot_cross_model_comparison(
                image=image,
                qwen_attention=qwen_output.aggregated_attention,
                paligemma_attention=paligemma_output.aggregated_attention,
                qwen_answer=qwen_output.predicted_answer,
                paligemma_answer=paligemma_output.predicted_answer,
                question=sample['question'],
                ground_truth=sample.get('ground_truth'),
                save_path=os.path.join(output_dir, f"comparison_{i+1}.png"),
            )
            
            # Layer attribution
            plot_layer_attribution(
                layer_contributions=qwen_output.layer_contributions,
                model_name="Qwen-VL",
                save_path=os.path.join(output_dir, f"qwen_layers_{i+1}.png"),
            )
            
            plot_layer_attribution(
                layer_contributions=paligemma_output.layer_contributions,
                model_name="PaliGemma",
                save_path=os.path.join(output_dir, f"paligemma_layers_{i+1}.png"),
            )
            
        except Exception as e:
            print(f"Warning: Failed to create visualization for sample {i+1}: {e}")
    
    # Create publication figure
    comparison_dicts = [
        {
            'qwen_confidence': c.qwen_confidence,
            'paligemma_confidence': c.paligemma_confidence,
            'attention_correlation': c.attention_correlation,
            'answer_agreement': c.answer_agreement,
        }
        for c in analyzer.comparisons
    ]
    
    insight_dicts = [
        {
            'title': ins.title,
            'description': ins.description,
            'significance': ins.statistical_significance,
        }
        for ins in analyzer.insights
    ]
    
    create_publication_figure(
        comparisons=comparison_dicts,
        insights=insight_dicts,
        save_path=os.path.join(output_dir, "publication_figure.png"),
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - analysis_results.json")
    print(f"  - entropy_analysis.png")
    print(f"  - comparison_*.png")
    print(f"  - publication_figure.png")
    
    print("\n" + "=" * 60)
    print("Key Insights")
    print("=" * 60)
    
    for insight in analyzer.insights:
        print(f"\n📊 {insight.title}")
        print(f"   {insight.description}")
        print(f"   Significance: {insight.statistical_significance:.2f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="VLM Probe: Comparative attention analysis for Qwen3-VL and PaliGemma2"
    )
    parser.add_argument(
        "--samples_csv",
        type=str,
        default="results.csv",
        help="Path to CSV file with VQA samples",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to analyze",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vlm_analysis_outputs",
        help="Output directory for results",
    )
    parser.add_argument(
        "--qwen_model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Qwen model ID",
    )
    parser.add_argument(
        "--paligemma_model",
        type=str,
        default="google/paligemma2-3b-pt-224",
        help="PaliGemma model ID",
    )
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Only check environment, don't run analysis",
    )
    
    args = parser.parse_args()
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please install missing packages.")
        sys.exit(1)
    
    if args.check_only:
        print("\n✓ Environment check passed!")
        sys.exit(0)
    
    # Load samples
    if os.path.exists(args.samples_csv):
        samples = load_vqa_samples(args.samples_csv, args.num_samples)
    else:
        print(f"\n⚠️  Samples file not found: {args.samples_csv}")
        print("Using example samples instead...")
        
        samples = [
            {
                'image': 'http://images.cocodataset.org/val2017/000000219578.jpg',
                'question': 'What color is the dog?',
                'ground_truth': 'brown',
            },
            {
                'image': 'http://images.cocodataset.org/val2017/000000397133.jpg',
                'question': 'What is the person doing?',
                'ground_truth': 'surfing',
            },
            {
                'image': 'http://images.cocodataset.org/val2017/000000037777.jpg',
                'question': 'How many people are in the image?',
                'ground_truth': '2',
            },
        ]
    
    # Run analysis
    run_full_analysis(
        samples=samples,
        output_dir=args.output_dir,
        qwen_model=args.qwen_model,
        paligemma_model=args.paligemma_model,
    )


if __name__ == "__main__":
    main()
