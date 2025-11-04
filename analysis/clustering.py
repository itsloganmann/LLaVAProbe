"""Clustering utilities for spatial attention analysis."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:  # pragma: no cover - optional dependency
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    hdbscan = None

from .config import AnalysisConfig
from .metrics import AttentionEntropyMetrics, compute_attention_entropy


class ClustererType(str, enum.Enum):
    DBSCAN = "dbscan"
    HDBSCAN = "hdbscan"
    GAUSSIAN_MIXTURE = "gaussian_mixture"


@dataclass
class ClusterResult:
    """Result of a single clustering run for one attention map."""

    clusterer: ClustererType
    labels: np.ndarray
    n_clusters: int
    n_noise: int
    noise_ratio: float
    average_strength: float
    attention_entropy: AttentionEntropyMetrics
    eps: Optional[float] = None
    min_samples: Optional[int] = None
    weight_exponent: Optional[float] = None
    token_confidence: Optional[float] = None
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class NullModelResult:
    """Metrics derived from null-model (permuted) clustering."""

    permutation_index: int
    cluster_result: ClusterResult


@dataclass
class ClusterSweepSummary:
    """Summary statistics for a given hyper-parameter configuration."""

    clusterer: ClustererType
    eps: Optional[float]
    min_samples: Optional[int]
    weight_exponent: Optional[float]
    correlations: Dict[str, Dict[str, float]]
    sample_count: int


@dataclass
class ClusterSweepResult:
    """Aggregate of clustering results across multiple samples."""

    summaries: List[ClusterSweepSummary]
    detailed_results: Mapping[Tuple[ClustererType, Optional[float], Optional[int], Optional[float]], List[ClusterResult]]
    null_results: Mapping[ClustererType, List[NullModelResult]]


def attention_to_points(attention_map: np.ndarray) -> np.ndarray:
    """Convert a 24x24 attention map into (x, y, value) coordinates."""

    h, w = attention_map.shape
    coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1).reshape(-1, 2)
    values = attention_map.reshape(-1, 1)
    y_flipped = (h - 1) - coords[:, 1:2]
    return np.concatenate([coords[:, :1], y_flipped, values], axis=1).astype(np.float64)


LabelArray = Union[Sequence[int], np.ndarray]


class ClusteringPipeline:
    """Pipeline orchestrating weighted clustering and null-model analysis."""

    def __init__(self, config: AnalysisConfig) -> None:
        self._config = config
        self._results: Dict[Tuple[ClustererType, Optional[float], Optional[int], Optional[float]], List[ClusterResult]] = {}
        self._null_results: Dict[ClustererType, List[NullModelResult]] = {}
        self._baseline_params = (1.3, 15, 1.0)

    def evaluate_sample(self, attention_map: np.ndarray, token_confidence: float) -> ClusterResult:
        points = attention_to_points(attention_map)
        scaled_points = self._standardize_points(points)

        baseline_result: Optional[ClusterResult] = None
        last_result: Optional[ClusterResult] = None
        for weight_exp in self._config.iter_weight_exponents():
            weights = np.power(points[:, 2], weight_exp)
            weights = np.clip(weights, a_min=1e-6, a_max=None)
            for eps, min_samples in self._config.iter_eps_min_samples():
                result = self._run_dbscan(
                    scaled_points,
                    eps=eps,
                    min_samples=min_samples,
                    sample_weight=weights,
                    token_confidence=token_confidence,
                    weight_exponent=weight_exp,
                )
                self._store_result(result)
                last_result = result
                if (eps, min_samples, weight_exp) == self._baseline_params:
                    baseline_result = result

        if hdbscan is not None:
            result = self._run_hdbscan(scaled_points, token_confidence=token_confidence)
            self._store_result(result)
            last_result = result
        gmm_result = self._run_gmm(points, token_confidence=token_confidence)
        self._store_result(gmm_result)
        if last_result is None:
            last_result = gmm_result

        self._evaluate_null_models(points, token_confidence)
        return baseline_result or last_result

    def summarize(self) -> ClusterSweepResult:
        summaries: List[ClusterSweepSummary] = []
        for key, records in self._results.items():
            clusterer, eps, min_samples, weight_exp = key
            confidences = np.array([r.token_confidence for r in records if r.token_confidence is not None])
            correlations = {
                "noise_ratio": {
                    "r_squared": self._r_squared(np.array([r.noise_ratio for r in records]), confidences),
                    "spearman": self._spearman(np.array([r.noise_ratio for r in records]), confidences),
                },
                "cluster_count": {
                    "r_squared": self._r_squared(np.array([r.n_clusters for r in records]), confidences),
                    "spearman": self._spearman(np.array([r.n_clusters for r in records]), confidences),
                },
                "mean_strength": {
                    "r_squared": self._r_squared(np.array([r.average_strength for r in records]), confidences),
                    "spearman": self._spearman(np.array([r.average_strength for r in records]), confidences),
                },
                "attention_entropy": {
                    "r_squared": self._r_squared(
                        np.array([r.attention_entropy.normalized_entropy for r in records]), confidences
                    ),
                    "spearman": self._spearman(
                        np.array([r.attention_entropy.normalized_entropy for r in records]), confidences
                    ),
                },
            }
            summaries.append(
                ClusterSweepSummary(
                    clusterer=clusterer,
                    eps=eps,
                    min_samples=min_samples,
                    weight_exponent=weight_exp,
                    correlations=correlations,
                    sample_count=len(records),
                )
            )
        return ClusterSweepResult(
            summaries=sorted(
                summaries,
                key=lambda s: (
                    s.clusterer.value,
                    float("inf") if s.eps is None else s.eps,
                    float("inf") if s.min_samples is None else s.min_samples,
                    float("inf") if s.weight_exponent is None else s.weight_exponent,
                ),
            ),
            detailed_results=self._results,
            null_results=self._null_results,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_dbscan(
        self,
        points: np.ndarray,
        *,
        eps: float,
        min_samples: int,
        sample_weight: Optional[np.ndarray],
        token_confidence: float,
        weight_exponent: float,
    ) -> ClusterResult:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(points, sample_weight=sample_weight)
        labels = db.labels_
        n_clusters, n_noise = _cluster_stats(labels)
        noise_ratio = n_noise / len(labels)
        mean_strength = float(np.average(sample_weight)) if sample_weight is not None else float(points[:, 2].mean())
        entropy_metrics = compute_attention_entropy(points[:, 2], self._config.entropy)
        metadata = {
            "mean_cluster_size": float(_mean_cluster_size(labels)),
            "std_cluster_size": float(_std_cluster_size(labels)),
        }
        return ClusterResult(
            clusterer=ClustererType.DBSCAN,
            labels=labels,
            n_clusters=n_clusters,
            n_noise=n_noise,
            noise_ratio=float(noise_ratio),
            average_strength=mean_strength,
            attention_entropy=entropy_metrics,
            eps=eps,
            min_samples=min_samples,
            weight_exponent=weight_exponent,
            token_confidence=token_confidence,
            metadata=metadata,
        )

    def _run_hdbscan(self, points: np.ndarray, *, token_confidence: float) -> ClusterResult:
        if hdbscan is None:  # pragma: no cover - optional dependency
            raise RuntimeError("hdbscan is not available")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
        labels = clusterer.fit_predict(points)
        n_clusters, n_noise = _cluster_stats(labels)
        entropy_metrics = compute_attention_entropy(points[:, 2], self._config.entropy)
        metadata = {"persistence": float(getattr(clusterer, "cluster_persistence_", np.nan))}
        return ClusterResult(
            clusterer=ClustererType.HDBSCAN,
            labels=labels,
            n_clusters=n_clusters,
            n_noise=n_noise,
            noise_ratio=float(n_noise / len(labels)),
            average_strength=float(points[:, 2].mean()),
            attention_entropy=entropy_metrics,
            token_confidence=token_confidence,
            metadata=metadata,
        )

    def _run_gmm(self, points: np.ndarray, *, token_confidence: float) -> ClusterResult:
        gm = GaussianMixture(n_components=3, covariance_type="full", random_state=self._config.random_seed)
        gm.fit(points[:, :3])
        labels = gm.predict(points[:, :3])
        n_clusters, _ = _cluster_stats(labels)
        responsibilities = gm.predict_proba(points[:, :3])
        noise_ratio = float(1.0 - responsibilities.max(axis=1).mean())
        entropy_metrics = compute_attention_entropy(points[:, 2], self._config.entropy)
        metadata = {
            "bic": float(gm.bic(points[:, :3])),
            "aic": float(gm.aic(points[:, :3])),
        }
        return ClusterResult(
            clusterer=ClustererType.GAUSSIAN_MIXTURE,
            labels=labels,
            n_clusters=n_clusters,
            n_noise=0,
            noise_ratio=noise_ratio,
            average_strength=float(points[:, 2].mean()),
            attention_entropy=entropy_metrics,
            token_confidence=token_confidence,
            metadata=metadata,
        )

    def _evaluate_null_models(self, points: np.ndarray, token_confidence: float) -> None:
        rng = np.random.default_rng(self._config.random_seed)
        for i in range(self._config.null_model_samples):
            permuted = points.copy()
            rng.shuffle(permuted[:, 2])
            rotated = permuted.reshape(24, 24, 3)
            rotated = np.rot90(rotated, k=int(rng.integers(0, 4)), axes=(0, 1)).reshape(-1, 3)
            flipped = rotated[::-1] if rng.random() < 0.5 else rotated
            result = self._run_dbscan(
                flipped,
                eps=1.3,
                min_samples=15,
                sample_weight=flipped[:, 2],
                token_confidence=token_confidence,
                weight_exponent=1.0,
            )
            self._null_results.setdefault(result.clusterer, []).append(
                NullModelResult(permutation_index=i, cluster_result=result)
            )

    def _standardize_points(self, points: np.ndarray) -> np.ndarray:
        scaler = StandardScaler()
        xy = points[:, :2]
        z = points[:, 2:3]
        scaled_xy = scaler.fit_transform(xy)
        normalized_z = (z - z.mean()) / (z.std() + 1e-6)
        return np.concatenate([scaled_xy, normalized_z], axis=1)

    def _store_result(self, result: ClusterResult) -> None:
        key = (result.clusterer, result.eps, result.min_samples, result.weight_exponent)
        self._results.setdefault(key, []).append(result)

    @staticmethod
    def _r_squared(x: np.ndarray, y: np.ndarray) -> float:
        if x.size == 0 or y.size == 0 or x.size != y.size:
            return 0.0
        if np.var(y) == 0:
            return 0.0
        correlation_matrix = np.corrcoef(x, y)
        r = correlation_matrix[0, 1]
        return float(r ** 2)

    @staticmethod
    def _spearman(x: np.ndarray, y: np.ndarray) -> float:
        if x.size == 0 or y.size == 0 or x.size != y.size:
            return 0.0
        x_ranks = _rankdata(x)
        y_ranks = _rankdata(y)
        if np.var(y_ranks) == 0:
            return 0.0
        correlation_matrix = np.corrcoef(x_ranks, y_ranks)
        return float(correlation_matrix[0, 1])


def _cluster_stats(labels: LabelArray) -> Tuple[int, int]:
    array = np.asarray(labels)
    unique = set(int(label) for label in array.tolist())
    n_clusters = len([label for label in unique if label != -1])
    n_noise = int(np.sum(array == -1))
    return n_clusters, n_noise


def _mean_cluster_size(labels: LabelArray) -> float:
    counts = _cluster_counts(labels)
    return float(np.mean(list(counts.values()))) if counts else 0.0


def _std_cluster_size(labels: LabelArray) -> float:
    counts = _cluster_counts(labels)
    return float(np.std(list(counts.values()))) if counts else 0.0


def _cluster_counts(labels: LabelArray) -> Dict[int, int]:
    array = np.asarray(labels)
    counts: Dict[int, int] = {}
    for raw_label in array.tolist():
        label = int(raw_label)
        if label == -1:
            continue
        counts[label] = counts.get(label, 0) + 1
    return counts


def _rankdata(values: np.ndarray) -> np.ndarray:
    temp = values.argsort()
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(len(values))
    return ranks