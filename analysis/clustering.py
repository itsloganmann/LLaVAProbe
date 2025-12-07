"""Clustering utilities for spatial attention analysis."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
from scipy.special import kl_div
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
    detailed_results: Mapping[Tuple[ClustererType, Optional[float],
                                    Optional[int], Optional[float]], List[ClusterResult]]
    null_results: Mapping[ClustererType, List[NullModelResult]]


def attention_to_points(attention_map: np.ndarray) -> np.ndarray:
    """Convert a 24x24 attention map into (x, y, value) coordinates."""

    h, w = attention_map.shape
    coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)),
                      axis=-1).reshape(-1, 2)
    values = attention_map.reshape(-1, 1)
    y_flipped = (h - 1) - coords[:, 1:2]
    return np.concatenate([coords[:, :1], y_flipped, values], axis=1).astype(np.float64)


LabelArray = Union[Sequence[int], np.ndarray]


class ClusteringPipeline:
    """Pipeline orchestrating weighted clustering and null-model analysis."""

    def __init__(self, config: AnalysisConfig) -> None:
        self._config = config
        self._results: Dict[Tuple[ClustererType, Optional[float],
                                  Optional[int], Optional[float]], List[ClusterResult]] = {}
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
            result = self._run_hdbscan(
                scaled_points, token_confidence=token_confidence)
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
            confidences = np.array(
                [r.token_confidence for r in records if r.token_confidence is not None])
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
                        np.array(
                            [r.attention_entropy.normalized_entropy for r in records]), confidences
                    ),
                    "spearman": self._spearman(
                        np.array(
                            [r.attention_entropy.normalized_entropy for r in records]), confidences
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
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(
            points, sample_weight=sample_weight)
        labels = db.labels_
        n_clusters, n_noise = _cluster_stats(labels)
        noise_ratio = n_noise / len(labels)
        mean_strength = float(np.average(
            sample_weight)) if sample_weight is not None else float(points[:, 2].mean())
        entropy_metrics = compute_attention_entropy(
            points[:, 2], self._config.entropy)
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
        entropy_metrics = compute_attention_entropy(
            points[:, 2], self._config.entropy)
        # --- FIX: persistence is an array, so compute a summary statistic ---
        persistence_vals = getattr(clusterer, "cluster_persistence_", None)

        if persistence_vals is None or len(persistence_vals) == 0:
            persistence_scalar = 0.0
        else:
        # You can use mean, max, or sum — mean is most standard
            persistence_scalar = float(np.mean(persistence_vals))

        metadata = {"persistence": persistence_scalar}

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
        gm = GaussianMixture(
            n_components=3, covariance_type="full", random_state=self._config.random_seed)
        gm.fit(points[:, :3])
        labels = gm.predict(points[:, :3])
        n_clusters, _ = _cluster_stats(labels)
        responsibilities = gm.predict_proba(points[:, :3])
        noise_ratio = float(1.0 - responsibilities.max(axis=1).mean())
        entropy_metrics = compute_attention_entropy(
            points[:, 2], self._config.entropy)
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
            rotated = np.rot90(rotated, k=int(
                rng.integers(0, 4)), axes=(0, 1)).reshape(-1, 3)
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
        key = (result.clusterer, result.eps,
               result.min_samples, result.weight_exponent)
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

    def analyze_layer_evolution(self, all_layer_attentions: List[np.ndarray]) -> Dict:
        """
        Complete wrapper to compute detailed evolution across all attention layers.
        """
        results = track_attention_evolution(all_layer_attentions)  # Now uses enhanced version
        print_layer_evolution(results)  # Pretty print the full table
        return results


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


"""Layer-wise attention evolution tracking with full layer details."""


def track_attention_evolution(layer_attentions: List[np.ndarray]) -> Dict:
    """
    Track attention evolution across ALL layers with detailed per-layer metrics.

    Args:
        layer_attentions: List of attention maps, one per layer
                          Each element should be [heads, tokens, tokens] numpy array

    Returns:
        Dict with detailed per-layer metrics and analysis
    """
    if not layer_attentions:
        return {
            "num_layers": 0,
            "per_layer_metrics": [],
            "critical_layers": np.array([]),
            "critical_layers_multimetric": np.array([])
        }

    per_layer_metrics = []
    prev_mean_attn = None

    # Compute metrics for each layer
    for layer_idx, layer_attn in enumerate(layer_attentions):
        mean_attn = np.mean(layer_attn, axis=0)

        layer_data = {
            "layer_index": layer_idx,
            "entropy": _compute_entropy(mean_attn),
            "sparsity": _compute_sparsity(mean_attn),
            "head_diversity": _compute_head_diversity(layer_attn),
        }

        # Compute shifts from previous layer
        if prev_mean_attn is not None:
            prev_metrics = per_layer_metrics[layer_idx - 1]
            layer_data["entropy_shift"] = layer_data["entropy"] - \
                prev_metrics["entropy"]
            layer_data["sparsity_shift"] = layer_data["sparsity"] - \
                prev_metrics["sparsity"]
            layer_data["kl_divergence"] = _compute_kl_divergence(
                prev_mean_attn, mean_attn)
        else:
            # First layer has no previous layer to compare
            layer_data["entropy_shift"] = 0.0
            layer_data["sparsity_shift"] = 0.0
            layer_data["kl_divergence"] = 0.0

        per_layer_metrics.append(layer_data)
        prev_mean_attn = mean_attn

    # Extract arrays for aggregate analysis
    layer_entropies = np.array([m["entropy"] for m in per_layer_metrics])
    entropy_shifts = np.array([m["entropy_shift"] for m in per_layer_metrics])
    sparsity_shifts = np.array([m["sparsity_shift"]
                               for m in per_layer_metrics])
    kl_divergences = np.array([m["kl_divergence"] for m in per_layer_metrics])

    # Detect critical layers using original entropy method
    entropy_transitions = np.abs(entropy_shifts[1:])  # Skip first (always 0)
    if len(entropy_transitions) > 0 and np.std(entropy_transitions) > 0:
        entropy_threshold = np.mean(
            entropy_transitions) + np.std(entropy_transitions)
        critical_layers_entropy = np.where(
            entropy_transitions > entropy_threshold)[0] + 1
    else:
        critical_layers_entropy = np.array([])

    # Detect critical layers using multi-metric voting
    critical_layers_multi = _detect_critical_multimetric(
        kl_divergences[1:],  # Skip first layer
        entropy_shifts[1:],
        sparsity_shifts[1:]
    )
    if len(critical_layers_multi) > 0:
        critical_layers_multi = critical_layers_multi + \
            1  # Adjust for skipped first layer

    # Mark critical layers in per_layer_metrics
    for layer_data in per_layer_metrics:
        layer_data["is_critical_entropy"] = layer_data["layer_index"] in critical_layers_entropy
        layer_data["is_critical_multimetric"] = layer_data["layer_index"] in critical_layers_multi

    # Compute aggregate statistics
    summary = {
        "most_diffuse_layer": int(np.argmax(layer_entropies)),
        "most_focused_layer": int(np.argmax([m["sparsity"] for m in per_layer_metrics])),
        "highest_shift_layer": int(np.argmax(kl_divergences[1:])) + 1 if len(kl_divergences) > 1 else -1,
        "most_diverse_heads_layer": int(np.argmax([m["head_diversity"] for m in per_layer_metrics])),
        "total_entropy_change": float(layer_entropies[-1] - layer_entropies[0]),
        "max_single_shift": float(np.max(np.abs(entropy_shifts[1:]))) if len(entropy_shifts) > 1 else 0.0,
    }

    return {
        "num_layers": len(layer_attentions),
        "per_layer_metrics": per_layer_metrics,  # Complete layer-by-layer data
        "critical_layers_entropy": critical_layers_entropy,
        "critical_layers_multimetric": critical_layers_multi,
        "summary": summary,

        # Also provide raw arrays for plotting/analysis
        "arrays": {
            "layer_entropies": layer_entropies,
            "entropy_shifts": entropy_shifts,
            "kl_divergences": kl_divergences,
            "sparsity_values": np.array([m["sparsity"] for m in per_layer_metrics]),
            "head_diversity": np.array([m["head_diversity"] for m in per_layer_metrics]),
        }
    }


def _compute_entropy(attn: np.ndarray) -> float:
    """Compute Shannon entropy of attention distribution."""
    flat = attn.flatten()
    flat = flat / (flat.sum() + 1e-9)
    flat = flat[flat > 1e-10]
    return float(entropy(flat))


def _compute_sparsity(attn: np.ndarray) -> float:
    """
    Compute sparsity using L1/L2 ratio.
    Higher values indicate more focused attention.
    """
    flat = attn.flatten()
    l1_norm = np.sum(np.abs(flat))
    l2_norm = np.sqrt(np.sum(flat ** 2))
    if l2_norm < 1e-10:
        return 0.0
    return float(l1_norm / (l2_norm * np.sqrt(len(flat))))


def _compute_head_diversity(layer_attn: np.ndarray) -> float:
    """
    Measure how different attention heads are from each other.
    Higher values = more diverse head behaviors.
    """
    if layer_attn.shape[0] <= 1:
        return 0.0

    heads_flat = layer_attn.reshape(layer_attn.shape[0], -1)
    heads_normalized = heads_flat / \
        (np.sum(heads_flat, axis=1, keepdims=True) + 1e-9)
    similarities = np.dot(heads_normalized, heads_normalized.T)

    n_heads = heads_normalized.shape[0]
    avg_similarity = (np.sum(similarities) - n_heads) / \
        (n_heads * (n_heads - 1))

    return float(1.0 - avg_similarity)


def _compute_kl_divergence(prev_attn: np.ndarray, curr_attn: np.ndarray) -> float:
    """
    Compute KL divergence between consecutive attention layers.
    Measures how much the attention distribution has shifted.
    """
    prev_flat = prev_attn.flatten()
    curr_flat = curr_attn.flatten()

    prev_flat = prev_flat / (prev_flat.sum() + 1e-9)
    curr_flat = curr_flat / (curr_flat.sum() + 1e-9)

    prev_flat = np.clip(prev_flat, 1e-10, 1.0)
    curr_flat = np.clip(curr_flat, 1e-10, 1.0)

    kl_forward = np.sum(kl_div(curr_flat, prev_flat))
    kl_backward = np.sum(kl_div(prev_flat, curr_flat))

    return float((kl_forward + kl_backward) / 2.0)


def _detect_critical_multimetric(
    kl_divergences: np.ndarray,
    entropy_shifts: np.ndarray,
    sparsity_shifts: np.ndarray,
    z_threshold: float = 1.5
) -> np.ndarray:
    """
    Detect critical layers using multiple metrics.
    A layer is critical if it shows significant change in multiple metrics.
    """
    if len(kl_divergences) == 0:
        return np.array([])

    critical_votes = np.zeros(len(kl_divergences))

    # Vote 1: High KL divergence
    if np.std(kl_divergences) > 0:
        kl_z_scores = (kl_divergences - np.mean(kl_divergences)
                       ) / np.std(kl_divergences)
        critical_votes += (kl_z_scores > z_threshold).astype(int)

    # Vote 2: High entropy shift (absolute value)
    if len(entropy_shifts) == len(kl_divergences) and np.std(np.abs(entropy_shifts)) > 0:
        entropy_z = (np.abs(entropy_shifts) - np.mean(np.abs(entropy_shifts))
                     ) / np.std(np.abs(entropy_shifts))
        critical_votes += (entropy_z > z_threshold).astype(int)

    # Vote 3: High sparsity shift (absolute value)
    if len(sparsity_shifts) == len(kl_divergences) and np.std(np.abs(sparsity_shifts)) > 0:
        sparsity_z = (np.abs(sparsity_shifts) -
                      np.mean(np.abs(sparsity_shifts))) / np.std(np.abs(sparsity_shifts))
        critical_votes += (sparsity_z > z_threshold).astype(int)

    # Return layers with at least 2 votes
    critical_layers = np.where(critical_votes >= 2)[0]

    return critical_layers


def print_layer_evolution(results: Dict) -> None:
    """
    Pretty print the complete layer evolution results.
    """
    print(f"\n{'='*80}")
    print(f"ATTENTION EVOLUTION ACROSS {results['num_layers']} LAYERS")
    print(f"{'='*80}\n")

    print(f"{'Layer':<6} {'Entropy':<10} {'ΔEntropy':<12} {'Sparsity':<10} {'KL-Div':<10} {'Critical':<10}")
    print(f"{'-'*80}")

    for layer_data in results['per_layer_metrics']:
        idx = layer_data['layer_index']
        entropy = layer_data['entropy']
        entropy_shift = layer_data['entropy_shift']
        sparsity = layer_data['sparsity']
        kl_div = layer_data['kl_divergence']

        critical_mark = ""
        if layer_data['is_critical_entropy']:
            critical_mark += "E"
        if layer_data['is_critical_multimetric']:
            critical_mark += "M"

        print(f"{idx:<6} {entropy:<10.4f} {entropy_shift:>+11.4f} {sparsity:<10.4f} "
              f"{kl_div:<10.4f} {critical_mark:<10}")

    print(f"\n{'-'*80}")
    print(f"Legend: E=Critical (Entropy), M=Critical (Multi-metric)")
    print(f"\nSummary:")
    for key, value in results['summary'].items():
        print(f"  {key}: {value}")
    print(f"{'='*80}\n")
