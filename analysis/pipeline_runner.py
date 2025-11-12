"""Command-line pipeline orchestrating attention and confidence analysis."""

from __future__ import annotations

import argparse
import json
import logging
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests
from PIL import Image

from . import (
    AnalysisConfig,
    AnalysisRecord,
    AnalysisWriter,
    ClusterReport,
    ClusteringPipeline,
    ConfidenceCalibrator,
    StructuredOutputs,
    LlavaRunner,
    RunnerMode,
    compute_attention_entropy,
)


@dataclass
class PromptEntry:
    question_type: str
    question: str
    prefix: str
    prompt: str
    ground_truth: str
    image_url: str

    @property
    def generic_prefix(self) -> str:
        return "Answer briefly"


def load_prompts(csv_path: Path) -> List[PromptEntry]:
    import csv

    prompts: List[PromptEntry] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "question_type", "question", "prefix", "prompt", "ground_truth",
            "image_url"
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV at {csv_path} is missing columns: {sorted(missing)}")
        for row in reader:
            prompts.append(
                PromptEntry(
                    question_type=row["question_type"],
                    question=row["question"],
                    prefix=row["prefix"],
                    prompt=row["prompt"],
                    ground_truth=row["ground_truth"],
                    image_url=row["image_url"],
                ))
    logging.info("Loaded %d prompts", len(prompts))
    return prompts


def fetch_image(url: str, timeout: float = 10.0) -> Image.Image:
    response = requests.get(url, timeout=timeout, verify=False)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def build_cluster_reports(labels: np.ndarray,
                          attention_map: np.ndarray) -> List[ClusterReport]:
    valid_labels = sorted({int(label) for label in labels if label != -1})
    flattened = attention_map.reshape(-1)
    if flattened.sum() <= 0:
        return []
    reports: List[ClusterReport] = []
    entropy_config = AnalysisConfig().entropy
    for label in valid_labels:
        indices = np.where(labels == label)[0]
        if len(indices) == 0:
            continue
        strengths = flattened[indices]
        metrics = compute_attention_entropy(strengths, entropy_config)
        reports.append(
            ClusterReport(
                cluster_id=label,
                size=len(indices),
                mean_strength=float(np.mean(strengths)),
                entropy=float(metrics.normalized_entropy),
            ))
    return reports


def determine_subset(question_type: str) -> str:
    lower = question_type.lower()
    if "yes" in lower:
        return "yes/no"
    return "short_answer"


def is_correct(predicted: str, ground_truth: str) -> bool:
    return predicted.strip().lower() == ground_truth.strip().lower()


def softmax(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    ex = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return ex / np.sum(ex, axis=axis, keepdims=True)


def combine_topk_heads(
        per_head_maps: np.ndarray,  # shape: (n_heads, H, W) or (n_heads, N)
        head_deltas: np.ndarray,  # shape: (n_heads,)
        k: int = 5,
        method: str = "weighted_avg",  # "weighted_avg" | "voting" | "max"
        vote_pct: float = 0.8,  # percentile for voting
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (combined_map (H,W), topk_indices (k,), weights (k,)).
    - per_head_maps: attention maps by head (n_heads, H, W) or (n_heads, N)
    - head_deltas: per-head score indicating importance (n_heads,)
    """
    if per_head_maps.ndim == 3:
        n_heads, H, W = per_head_maps.shape
        per_head_flat = per_head_maps.reshape(n_heads, -1)
    else:
        n_heads, N = per_head_maps.shape
        per_head_flat = per_head_maps

    head_deltas = np.asarray(head_deltas)
    if head_deltas.shape[0] != per_head_flat.shape[0]:
        raise ValueError(
            "head_deltas and per_head_maps have different head counts")

    # top-k indices (highest deltas)
    k = min(k, per_head_flat.shape[0])
    topk_idx = np.argsort(head_deltas)[-k:][::-1]  # descending
    topk_scores = head_deltas[topk_idx]

    if method == "weighted_avg":
        weights = softmax(topk_scores)
        weighted = (weights[:, None] * per_head_flat[topk_idx]).sum(axis=0)
        combined = weighted
    elif method == "voting":
        # threshold each head at vote_pct percentile
        masks = []
        for idx in topk_idx:
            arr = per_head_flat[idx]
            thr = np.percentile(
                arr, 100 * (1.0 - vote_pct))  # vote_pct proportion preserved
            masks.append((arr >= thr).astype(float))
        mask_sum = np.sum(np.stack(masks, axis=0), axis=0)  # counts [0..k]
        combined = mask_sum  # raw vote counts
        weights = np.ones(len(topk_idx)) / len(topk_idx)
    elif method == "max":
        combined = np.max(per_head_flat[topk_idx], axis=0)
        weights = np.ones(len(topk_idx)) / len(topk_idx)
    else:
        raise ValueError("Unknown method: " + method)

    # normalize combined to sum to 1 (avoid zero-sum)
    if combined.sum() <= 0:
        norm_combined = combined
    else:
        norm_combined = combined / float(np.sum(combined))

    # reshape back to H,W if needed
    if per_head_maps.ndim == 3:
        combined_map = norm_combined.reshape(H, W)
    else:
        combined_map = norm_combined.reshape(-1)

    return combined_map, topk_idx, (softmax(topk_scores)
                                    if method == "weighted_avg" else
                                    np.ones(len(topk_idx)) / len(topk_idx))


def run_pipeline(
    *,
    prompts_path: Path,
    output_dir: Path,
    quantization: Optional[str] = None,
    log_level: str = "INFO",
) -> None:
    log_file = output_dir / "pipeline_execution.log"
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = AnalysisConfig()
    runner = LlavaRunner(
        config=config,
        quantization=None if quantization == "none" else quantization)
    clustering = ClusteringPipeline(config)
    calibrator = ConfidenceCalibrator(config.calibration)
    writer = AnalysisWriter(output_dir)

    prompts = load_prompts(prompts_path)
    records: List[AnalysisRecord] = []

    for index, entry in enumerate(prompts, start=1):
        logging.info("[%d/%d] Processing %s", index, len(prompts),
                     entry.image_url)
        try:
            image = fetch_image(entry.image_url)
        except Exception as exc:  # pragma: no cover - network issues
            logging.exception("Failed to fetch %s: %s", entry.image_url, exc)
            continue

        variants = [
            (RunnerMode.FULL, entry.prefix, "baseline", 0.0),
            (RunnerMode.FULL, "", "no_prefix", 0.0),
            (RunnerMode.FULL, entry.generic_prefix, "generic", 0.0),
            (RunnerMode.LANGUAGE_ONLY, entry.prefix, "language_only", 0.0),
            (RunnerMode.VISUAL_DROPOUT, entry.prefix, "dropout_0.2", 0.2),
            (RunnerMode.VISUAL_DROPOUT, entry.prefix, "dropout_0.5", 0.5),
        ]

        for resolution in config.patch_sweep.resolutions:
            resized = image.resize((resolution, resolution))
            for mode, prefix_value, variant_name, dropout_rate in variants:
                variant_label = f"{variant_name}_r{resolution}"
                try:
                    run_output = runner.run(
                        image=resized,
                        prompt=entry.prompt,
                        prefix=prefix_value,
                        mode=mode,
                        dropout_rate=dropout_rate,
                        ground_truth=entry.ground_truth,
                    )
                except Exception as exc:  # pragma: no cover - model failure
                    logging.exception("Model run failed for %s: %s",
                                      entry.image_url, exc)
                    continue

                # ---------- Top-k head analysis ----------
                # Assumption: LlavaRunner returns per-head attention maps as `run_output.per_head_attention`
                # with shape (n_heads, H, W) and head deltas as run_output.head_delta (n_heads,)
                per_head_maps = getattr(run_output, "per_head_attention", None)
                head_deltas = np.asarray(getattr(run_output, "head_delta", []))

                # Default: fallback to original single map if per-head not present
                if per_head_maps is None or per_head_maps.size == 0 or head_deltas.size == 0:
                    # old behavior
                    cluster_result = clustering.evaluate_sample(
                        run_output.attention_map, run_output.token_confidence)
                    cluster_reports = build_cluster_reports(
                        cluster_result.labels, run_output.attention_map)
                    attention_metrics = compute_attention_entropy(
                        run_output.attention_map, config.entropy)
                    combined_topk_info = {
                        "method": "single_head_fallback",
                        "topk": [None],
                        "weights": []
                    }
                else:
                    # choose k and method (you can tune these or make them config params)
                    TOP_K = getattr(config, "analysis_top_k", 5)
                    COMB_METHOD = getattr(
                        config, "analysis_comb_method",
                        "weighted_avg")  # or "voting" / "max"

                    combined_map, topk_idx, topk_weights = combine_topk_heads(
                        per_head_maps=per_head_maps,
                        head_deltas=head_deltas,
                        k=TOP_K,
                        method=COMB_METHOD,
                        vote_pct=0.8,
                    )

                    # clustering & reports use the combined map
                    cluster_result = clustering.evaluate_sample(
                        combined_map, run_output.token_confidence)
                    cluster_reports = build_cluster_reports(
                        cluster_result.labels, combined_map)
                    attention_metrics = compute_attention_entropy(
                        combined_map, config.entropy)

                    # also compute per-head reports for diagnostics (optional)
                    per_head_reports = []
                    for hid in topk_idx:
                        head_map = per_head_maps[hid]
                        hr_cluster_result = clustering.evaluate_sample(
                            head_map, run_output.token_confidence)
                        hr_reports = build_cluster_reports(
                            hr_cluster_result.labels, head_map)
                        hr_entropy = compute_attention_entropy(
                            head_map, config.entropy)
                        per_head_reports.append({
                            "head":
                            int(hid),
                            "entropy":
                            float(hr_entropy.normalized_entropy),
                            "clusters":
                            hr_reports
                        })

                    combined_topk_info = {
                        "method": COMB_METHOD,
                        "topk": [int(h) for h in topk_idx],
                        "weights": [float(w) for w in topk_weights],
                        "per_head": per_head_reports
                    }
                # ---------- End top-k handling ----------
                confidence = run_output.confidence_metrics

                record = AnalysisRecord(
                    question_type=entry.question_type,
                    question=entry.question,
                    image_url=entry.image_url,
                    ground_truth=entry.ground_truth,
                    predicted_answer=run_output.predicted_answer,
                    confidence=confidence,
                    attention_entropy=attention_metrics.normalized_entropy,
                    clusters=cluster_reports,
                    noise_ratio=cluster_result.noise_ratio,
                    cluster_count=cluster_result.n_clusters,
                    token_confidence=run_output.token_confidence,
                    top_margin=confidence.probability_margin,
                    logit_margin=confidence.logit_margin,
                    head_delta=run_output.head_delta,
                    per_head_topk_info=combined_topk_info,
                    ground_truth_delta=run_output.ground_truth_delta,
                    ground_truth_tokens=run_output.ground_truth_tokens,
                    ablations=[{
                        "name": result.experiment.name,
                        "delta": result.confidence_delta,
                        "ablated_confidence": result.ablated_confidence,
                    } for result in run_output.ablations],
                    run_mode=mode.value,
                    prefix_variant=variant_label,
                )
                records.append(record)

                if variant_name == "baseline" and resolution == config.patch_sweep.resolutions[
                        0]:
                    subset = determine_subset(entry.question_type)
                    calibrator.add(
                        subset=subset,
                        probability=run_output.token_confidence,
                        is_correct=is_correct(run_output.predicted_answer,
                                              entry.ground_truth),
                    )

    outputs = StructuredOutputs(records=records,
                                calibration=calibrator.summarize())
    json_path = writer.write_json(outputs, "analysis_records.json")
    csv_path = writer.write_csv(outputs, "analysis_records.csv")

    clustering_summary = clustering.summarize()
    null_payload = []
    for clusterer, results in clustering_summary.null_results.items():
        if not results:
            continue
        avg_noise = float(
            np.mean([res.cluster_result.noise_ratio for res in results]))
        avg_clusters = float(
            np.mean([res.cluster_result.n_clusters for res in results]))
        null_payload.append({
            "clusterer": clusterer.value,
            "average_noise_ratio": avg_noise,
            "average_clusters": avg_clusters,
            "samples": len(results),
        })

    summary_payload = {
        "summaries": [{
            "clusterer": summary.clusterer.value,
            "eps": summary.eps,
            "min_samples": summary.min_samples,
            "weight_exponent": summary.weight_exponent,
            "correlations": summary.correlations,
            "sample_count": summary.sample_count,
        } for summary in clustering_summary.summaries],
        "null_models":
        null_payload,
    }
    (output_dir / "clustering_summary.json").write_text(json.dumps(
        summary_payload, indent=2),
                                                        encoding="utf-8")

    logging.info("JSON output saved to %s", json_path)
    logging.info("CSV output saved to %s", csv_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the attention analysis pipeline")
    parser.add_argument("--prompts",
                        type=str,
                        default="results.csv",
                        help="Path to CSV generated by tester.py")
    parser.add_argument("--output-dir",
                        type=str,
                        default="analysis_outputs",
                        help="Directory for structured outputs")
    parser.add_argument(
        "--quantization",
        choices=["none", "4bit", "8bit"],
        default="none",
        help="Optional quantization mode for loading LLaVA",
    )
    parser.add_argument("--log-level",
                        default="INFO",
                        help="Logging level (e.g., INFO, DEBUG)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_pipeline(
        prompts_path=Path(args.prompts),
        output_dir=Path(args.output_dir),
        quantization=args.quantization,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
