"""Command-line pipeline orchestrating attention and confidence analysis."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

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
        required = {"question_type", "question", "prefix", "prompt", "ground_truth", "image_url"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV at {csv_path} is missing columns: {sorted(missing)}")
        for row in reader:
            prompts.append(
                PromptEntry(
                    question_type=row["question_type"],
                    question=row["question"],
                    prefix=row["prefix"],
                    prompt=row["prompt"],
                    ground_truth=row["ground_truth"],
                    image_url=row["image_url"],
                )
            )
    logging.info("Loaded %d prompts", len(prompts))
    return prompts


def fetch_image(url: str, timeout: float = 10.0) -> Image.Image:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return Image.open(response.raw).convert("RGB")


def build_cluster_reports(labels: np.ndarray, attention_map: np.ndarray) -> List[ClusterReport]:
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
            )
        )
    return reports


def determine_subset(question_type: str) -> str:
    lower = question_type.lower()
    if "yes" in lower:
        return "yes/no"
    return "short_answer"


def is_correct(predicted: str, ground_truth: str) -> bool:
    return predicted.strip().lower() == ground_truth.strip().lower()


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
    runner = LlavaRunner(config=config, quantization=None if quantization == "none" else quantization)
    clustering = ClusteringPipeline(config)
    calibrator = ConfidenceCalibrator(config.calibration)
    writer = AnalysisWriter(output_dir)

    prompts = load_prompts(prompts_path)
    records: List[AnalysisRecord] = []

    for index, entry in enumerate(prompts, start=1):
        logging.info("[%d/%d] Processing %s", index, len(prompts), entry.image_url)
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
                    logging.exception("Model run failed for %s: %s", entry.image_url, exc)
                    continue

                cluster_result = clustering.evaluate_sample(run_output.attention_map, run_output.token_confidence)
                cluster_reports = build_cluster_reports(cluster_result.labels, run_output.attention_map)

                attention_metrics = compute_attention_entropy(run_output.attention_map, config.entropy)
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
                    ground_truth_delta=run_output.ground_truth_delta,
                    ground_truth_tokens=run_output.ground_truth_tokens,
                    ablations=[
                        {
                            "name": result.experiment.name,
                            "delta": result.confidence_delta,
                            "ablated_confidence": result.ablated_confidence,
                        }
                        for result in run_output.ablations
                    ],
                    run_mode=mode.value,
                    prefix_variant=variant_label,
                )
                records.append(record)

                if variant_name == "baseline" and resolution == config.patch_sweep.resolutions[0]:
                    subset = determine_subset(entry.question_type)
                    calibrator.add(
                        subset=subset,
                        probability=run_output.token_confidence,
                        is_correct=is_correct(run_output.predicted_answer, entry.ground_truth),
                    )

    outputs = StructuredOutputs(records=records, calibration=calibrator.summarize())
    json_path = writer.write_json(outputs, "analysis_records.json")
    csv_path = writer.write_csv(outputs, "analysis_records.csv")

    clustering_summary = clustering.summarize()
    null_payload = []
    for clusterer, results in clustering_summary.null_results.items():
        if not results:
            continue
        avg_noise = float(np.mean([res.cluster_result.noise_ratio for res in results]))
        avg_clusters = float(np.mean([res.cluster_result.n_clusters for res in results]))
        null_payload.append(
            {
                "clusterer": clusterer.value,
                "average_noise_ratio": avg_noise,
                "average_clusters": avg_clusters,
                "samples": len(results),
            }
        )

    summary_payload = {
        "summaries": [
            {
                "clusterer": summary.clusterer.value,
                "eps": summary.eps,
                "min_samples": summary.min_samples,
                "weight_exponent": summary.weight_exponent,
                "correlations": summary.correlations,
                "sample_count": summary.sample_count,
            }
            for summary in clustering_summary.summaries
        ],
        "null_models": null_payload,
    }
    (output_dir / "clustering_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    logging.info("JSON output saved to %s", json_path)
    logging.info("CSV output saved to %s", csv_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the attention analysis pipeline")
    parser.add_argument("--prompts", type=str, default="results.csv", help="Path to CSV generated by tester.py")
    parser.add_argument("--output-dir", type=str, default="analysis_outputs", help="Directory for structured outputs")
    parser.add_argument(
        "--quantization",
        choices=["none", "4bit", "8bit"],
        default="none",
        help="Optional quantization mode for loading LLaVA",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (e.g., INFO, DEBUG)")
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
