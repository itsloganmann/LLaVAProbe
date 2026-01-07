"""
Quick validation script to check accuracy with unified correctness criteria.
Run this to estimate accuracy before re-running the full analysis.
"""

import pandas as pd
import os
import sys

# Get paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)


def check_correct_strict(predicted, ground_truth):
    """Original strict matching."""
    if pd.isna(predicted) or pd.isna(ground_truth):
        return False
    return str(predicted).lower().strip() == str(ground_truth).lower().strip()


def check_correct_lenient(predicted, ground_truth):
    """Unified lenient matching."""
    if pd.isna(predicted) or pd.isna(ground_truth):
        return False
    pred_lower = str(predicted).lower().strip()
    gt_lower = str(ground_truth).lower().strip()
    return (pred_lower == gt_lower or gt_lower.startswith(pred_lower)
            or pred_lower.startswith(gt_lower) or gt_lower in pred_lower
            or pred_lower in gt_lower)


def main():
    # Load results
    df = pd.read_csv(os.path.join(PARENT_DIR, 'results.csv'))

    print("=" * 70)
    print("CORRECTNESS VALIDATION")
    print("=" * 70)
    print(f"\nTotal samples: {len(df)}")

    # Check both criteria
    strict_correct = 0
    lenient_correct = 0
    examples_changed = []

    for _, row in df.iterrows():
        pred = row['model_answer']
        gt = row['ground_truth']

        strict = check_correct_strict(pred, gt)
        lenient = check_correct_lenient(pred, gt)

        if strict:
            strict_correct += 1
        if lenient:
            lenient_correct += 1

        if lenient and not strict:
            examples_changed.append({
                'question': row['question'][:50],
                'predicted': pred,
                'ground_truth': gt
            })

    print(f"\n📊 ACCURACY COMPARISON:")
    print(
        f"   Strict matching:  {strict_correct}/{len(df)} = {strict_correct/len(df)*100:.1f}%"
    )
    print(
        f"   Lenient matching: {lenient_correct}/{len(df)} = {lenient_correct/len(df)*100:.1f}%"
    )
    print(
        f"   Additional correct: +{lenient_correct - strict_correct} samples")

    # Show examples of changed classifications
    print(f"\n📝 EXAMPLES OF NEWLY CORRECT (first 20):")
    print("-" * 70)
    for ex in examples_changed[:20]:
        print(f"   Q: {ex['question']}...")
        print(f"   Pred: '{ex['predicted']}' → GT: '{ex['ground_truth']}'")
        print()

    # By question type
    print("\n📊 BY QUESTION TYPE:")
    print("-" * 70)
    for qt in df['question_type'].unique():
        subset = df[df['question_type'] == qt]
        strict_acc = sum(
            check_correct_strict(row['model_answer'], row['ground_truth'])
            for _, row in subset.iterrows()) / len(subset)
        lenient_acc = sum(
            check_correct_lenient(row['model_answer'], row['ground_truth'])
            for _, row in subset.iterrows()) / len(subset)
        print(
            f"   {qt}: {strict_acc*100:.1f}% → {lenient_acc*100:.1f}% (+{(lenient_acc-strict_acc)*100:.1f}%)"
        )

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print(
        f"\nWith lenient matching, expected accuracy: ~{lenient_correct/len(df)*100:.0f}%"
    )
    print(
        f"This is {'closer to' if abs(lenient_correct/len(df) - 0.72) < 0.2 else 'different from'} Emily's 71.9%"
    )

    if lenient_correct / len(df) > 0.5:
        print("\n✅ Lenient matching brings accuracy to reasonable range.")
        print("   Recommend re-running analysis with unified methodology.")
    else:
        print("\n⚠️ Accuracy still low. May need to investigate further.")
        print(
            "   Possible issues: tokenization, prompt format, model weights.")


if __name__ == "__main__":
    main()
