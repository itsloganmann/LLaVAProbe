"""Utilities for exporting structured analysis outputs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

from .metrics import ConfidenceBreakdown, ConfidenceMetrics


@dataclass
class ClusterReport:
    cluster_id: int
    size: int
    mean_strength: float
    entropy: float


@dataclass
class AnalysisRecord:
    question_type: str
    question: str
    image_url: str
    ground_truth: str
    predicted_answer: str
    confidence: ConfidenceMetrics
    attention_entropy: float
    clusters: Sequence[ClusterReport]
    noise_ratio: float
    cluster_count: int
    token_confidence: float
    top_margin: float
    logit_margin: float
    head_delta: Optional[float] = None
    ground_truth_delta: Optional[float] = None
    ground_truth_tokens: Optional[Sequence[str]] = None
    ablations: Optional[List[Dict[str, Union[float, str]]]] = None
    run_mode: Optional[str] = None
    prefix_variant: Optional[str] = None


@dataclass
class StructuredOutputs:
    records: List[AnalysisRecord]
    calibration: List[ConfidenceBreakdown] = field(default_factory=list)

    def to_json(self) -> str:
        payload = {
            "records": [self._record_to_dict(record) for record in self.records],
            "calibration": [asdict(breakdown) for breakdown in self.calibration],
        }
        return json.dumps(payload, indent=2)

    @staticmethod
    def _record_to_dict(record: AnalysisRecord) -> dict:
        cluster_payload = [asdict(cluster) for cluster in record.clusters]
        return {
            "question_type": record.question_type,
            "question": record.question,
            "image_url": record.image_url,
            "ground_truth": record.ground_truth,
            "predicted_answer": record.predicted_answer,
            "confidence": asdict(record.confidence),
            "attention_entropy": record.attention_entropy,
            "clusters": cluster_payload,
            "noise_ratio": record.noise_ratio,
            "cluster_count": record.cluster_count,
            "token_confidence": record.token_confidence,
            "top_margin": record.top_margin,
            "logit_margin": record.logit_margin,
            "head_delta": record.head_delta,
            "ground_truth_delta": record.ground_truth_delta,
            "ground_truth_tokens": list(record.ground_truth_tokens) if record.ground_truth_tokens else None,
            "ablations": record.ablations,
            "run_mode": record.run_mode,
            "prefix_variant": record.prefix_variant,
        }


class AnalysisWriter:
    """Persists structured outputs to artefact files."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, outputs: StructuredOutputs, filename: str = "analysis_records.json") -> Path:
        path = self.output_dir / filename
        path.write_text(outputs.to_json(), encoding="utf-8")
        return path

    def write_csv(self, outputs: StructuredOutputs, filename: str = "analysis_records.csv") -> Path:
        path = self.output_dir / filename
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "question_type",
                    "question",
                    "image_url",
                    "ground_truth",
                    "predicted_answer",
                    "top1_probability",
                    "probability_margin",
                    "logit_margin",
                    "token_entropy",
                    "sequence_probability",
                    "attention_entropy",
                    "cluster_count",
                    "noise_ratio",
                    "head_delta",
                    "ground_truth_delta",
                    "ablations",
                    "run_mode",
                    "prefix_variant",
                ]
            )
            for record in outputs.records:
                confidence = record.confidence
                writer.writerow(
                    [
                        record.question_type,
                        record.question,
                        record.image_url,
                        record.ground_truth,
                        record.predicted_answer,
                        confidence.top1_probability,
                        confidence.probability_margin,
                        confidence.logit_margin,
                        confidence.token_entropy,
                        confidence.sequence_probability,
                        record.attention_entropy,
                        record.cluster_count,
                        record.noise_ratio,
                        record.head_delta,
                        record.ground_truth_delta,
                        json.dumps(record.ablations or []),
                        record.run_mode,
                        record.prefix_variant,
                    ]
                )
        return path