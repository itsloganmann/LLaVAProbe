"""Adaptive clustering with entropy-based parameter selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan
    from hdbscan.validity import validity_index
except ImportError:
    hdbscan = None
    validity_index = None

from .config import AnalysisConfig
from .clustering import ClusterResult, ClustererType, attention_to_points
from .metrics import compute_attention_entropy


@dataclass
class AdaptiveClusteringMetrics:
    """Metrics for adaptive clustering quality assessment."""
    entropy: float
    variance: float
    selected_eps: float
    silhouette_score: Optional[float]
    dbcv_score: Optional[float]
    parameter_search_results: List[Dict[str, float]]


def compute_attention_entropy_signal(attention_map: np.ndarray) -> float:
    """
    Compute entropy of attention distribution as adaptivity signal.
    
    Args:
        attention_map: 2D spatial attention map (e.g., 24x24)
    
    Returns:
        Normalized entropy value [0, 1]
    """
    # Flatten and normalize to probability distribution
    flat = attention_map.flatten()
    flat = flat / (flat.sum() + 1e-10)
    
    # Compute Shannon entropy
    entropy = -np.sum(flat * np.log(flat + 1e-10))
    
    # Normalize by max possible entropy (uniform distribution)
    max_entropy = np.log(len(flat))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    return float(normalized_entropy)


def compute_attention_variance_signal(attention_map: np.ndarray) -> float:
    """
    Compute variance of attention distribution as diversity signal.
    
    Args:
        attention_map: 2D spatial attention map
    
    Returns:
        Normalized variance [0, 1]
    """
    flat = attention_map.flatten()
    # Normalize to [0, 1]
    if flat.max() > flat.min():
        normalized = (flat - flat.min()) / (flat.max() - flat.min())
    else:
        normalized = flat
    
    variance = float(np.var(normalized))
    return variance


def map_entropy_to_eps(
    entropy: float,
    eps_min: float = 0.5,
    eps_max: float = 2.5,
    entropy_threshold_loose: float = 0.7
) -> float:
    """
    Map entropy signal to eps parameter for DBSCAN.
    
    High entropy → looser clusters (higher eps)
    Low entropy → tighter clusters (lower eps)
    
    Args:
        entropy: Normalized entropy [0, 1]
        eps_min: Minimum eps value
        eps_max: Maximum eps value
        entropy_threshold_loose: Entropy above which to use loose clustering
    
    Returns:
        Adaptive eps value
    """
    # Linear mapping with threshold
    if entropy > entropy_threshold_loose:
        # High entropy: map to upper range
        alpha = (entropy - entropy_threshold_loose) / (1.0 - entropy_threshold_loose)
        eps = eps_min + alpha * (eps_max - eps_min) + 0.5
    else:
        # Low entropy: map to lower range
        alpha = entropy / entropy_threshold_loose
        eps = eps_min + alpha * (eps_max - eps_min) * 0.5
    
    return float(np.clip(eps, eps_min, eps_max))


def adaptive_dbscan_with_quality_metrics(
    attention_map: np.ndarray,
    token_confidence: float,
    config: AnalysisConfig,
    eps_candidates: Optional[List[float]] = None,
    min_samples: int = 15,
) -> Tuple[ClusterResult, AdaptiveClusteringMetrics]:
    """
    Perform adaptive DBSCAN clustering with automatic parameter selection.
    
    Uses entropy and variance as signals to:
    1. Suggest initial eps value
    2. Search around that value
    3. Select best clustering using silhouette score or DBCV
    
    Args:
        attention_map: 2D attention heatmap
        token_confidence: Model confidence for this sample
        config: Analysis configuration
        eps_candidates: Optional list of eps values to try (if None, auto-generated)
        min_samples: DBSCAN min_samples parameter
    
    Returns:
        Tuple of (best ClusterResult, AdaptiveClusteringMetrics)
    """
    # Compute adaptivity signals
    entropy = compute_attention_entropy_signal(attention_map)
    variance = compute_attention_variance_signal(attention_map)
    
    # Convert attention map to points
    points = attention_to_points(attention_map)
    scaled_points = _standardize_points(points)
    
    # Get initial eps suggestion from entropy
    suggested_eps = map_entropy_to_eps(entropy)
    
    # Generate candidate eps values around suggestion
    if eps_candidates is None:
        eps_candidates = [
            suggested_eps * 0.7,
            suggested_eps * 0.85,
            suggested_eps,
            suggested_eps * 1.15,
            suggested_eps * 1.3,
        ]
    
    # Try each eps candidate and compute quality metrics
    best_result: Optional[ClusterResult] = None
    best_score: float = -np.inf
    search_results: List[Dict[str, float]] = []
    
    for eps in eps_candidates:
        # Run DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(scaled_points)
        
        # Skip if all noise or single cluster
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 2:
            search_results.append({
                "eps": eps,
                "n_clusters": n_clusters,
                "quality_score": -1.0,
            })
            continue
        
        # Compute quality score (silhouette)
        try:
            silhouette = silhouette_score(scaled_points, labels)
        except Exception:
            silhouette = -1.0
        
        # Compute DBCV if available
        dbcv = None
        if hdbscan is not None and validity_index is not None:
            try:
                dbcv = validity_index(scaled_points, labels, metric='euclidean')
            except Exception:
                pass
        
        # Use silhouette as primary quality metric
        quality_score = silhouette
        
        search_results.append({
            "eps": eps,
            "n_clusters": n_clusters,
            "silhouette": silhouette,
            "dbcv": dbcv if dbcv is not None else np.nan,
            "quality_score": quality_score,
        })
        
        # Track best result
        if quality_score > best_score:
            best_score = quality_score
            
            # Compute cluster statistics
            n_noise = int(np.sum(labels == -1))
            noise_ratio = n_noise / len(labels)
            mean_strength = float(points[:, 2].mean())
            
            # Compute entropy metrics using existing function
            entropy_metrics = compute_attention_entropy(points[:, 2], config.entropy)
            
            best_result = ClusterResult(
                clusterer=ClustererType.DBSCAN,
                labels=labels,
                n_clusters=n_clusters,
                n_noise=n_noise,
                noise_ratio=noise_ratio,
                average_strength=mean_strength,
                attention_entropy=entropy_metrics,
                eps=eps,
                min_samples=min_samples,
                weight_exponent=1.0,
                token_confidence=token_confidence,
                metadata={
                    "silhouette_score": silhouette,
                    "dbcv_score": dbcv if dbcv is not None else np.nan,
                    "entropy_signal": entropy,
                    "variance_signal": variance,
                    "adaptive_eps": True,
                },
            )
    
    # If no valid clustering found, use default eps
    if best_result is None:
        eps = suggested_eps
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(scaled_points)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        noise_ratio = n_noise / len(labels) if len(labels) > 0 else 0.0
        
        entropy_metrics = compute_attention_entropy(points[:, 2], config.entropy)
        
        best_result = ClusterResult(
            clusterer=ClustererType.DBSCAN,
            labels=labels,
            n_clusters=n_clusters,
            n_noise=n_noise,
            noise_ratio=noise_ratio,
            average_strength=float(points[:, 2].mean()),
            attention_entropy=entropy_metrics,
            eps=eps,
            min_samples=min_samples,
            weight_exponent=1.0,
            token_confidence=token_confidence,
            metadata={
                "entropy_signal": entropy,
                "variance_signal": variance,
                "adaptive_eps": True,
            },
        )
    
    # Build adaptive metrics
    adaptive_metrics = AdaptiveClusteringMetrics(
        entropy=entropy,
        variance=variance,
        selected_eps=best_result.eps,
        silhouette_score=best_result.metadata.get("silhouette_score"),
        dbcv_score=best_result.metadata.get("dbcv_score"),
        parameter_search_results=search_results,
    )
    
    return best_result, adaptive_metrics


def adaptive_hdbscan_with_entropy(
    attention_map: np.ndarray,
    token_confidence: float,
    config: AnalysisConfig,
) -> Tuple[ClusterResult, AdaptiveClusteringMetrics]:
    """
    Perform HDBSCAN with entropy-based parameter adjustment.
    
    Args:
        attention_map: 2D attention heatmap
        token_confidence: Model confidence
        config: Analysis configuration
    
    Returns:
        Tuple of (ClusterResult, AdaptiveClusteringMetrics)
    """
    if hdbscan is None:
        raise RuntimeError("hdbscan is not available")
    
    # Compute signals
    entropy = compute_attention_entropy_signal(attention_map)
    variance = compute_attention_variance_signal(attention_map)
    
    # Convert to points
    points = attention_to_points(attention_map)
    scaled_points = _standardize_points(points)
    
    # Adjust min_cluster_size based on entropy
    # High entropy → smaller clusters allowed
    if entropy > 0.7:
        min_cluster_size = 3
    elif entropy > 0.5:
        min_cluster_size = 5
    else:
        min_cluster_size = 8
    
    # Run HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=3,
        cluster_selection_epsilon=0.0,
    )
    labels = clusterer.fit_predict(scaled_points)
    
    # Compute statistics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    noise_ratio = n_noise / len(labels)
    
    entropy_metrics = compute_attention_entropy(points[:, 2], config.entropy)
    
    # Try to get DBCV score
    dbcv = None
    if validity_index is not None:
        try:
            dbcv = validity_index(scaled_points, labels, metric='euclidean')
        except Exception:
            pass
    
    result = ClusterResult(
        clusterer=ClustererType.HDBSCAN,
        labels=labels,
        n_clusters=n_clusters,
        n_noise=n_noise,
        noise_ratio=noise_ratio,
        average_strength=float(points[:, 2].mean()),
        attention_entropy=entropy_metrics,
        eps=None,
        min_samples=None,
        weight_exponent=None,
        token_confidence=token_confidence,
        metadata={
            "min_cluster_size": min_cluster_size,
            "dbcv_score": dbcv if dbcv is not None else np.nan,
            "entropy_signal": entropy,
            "variance_signal": variance,
            "persistence": float(getattr(clusterer, "cluster_persistence_", np.nan)),
        },
    )
    
    adaptive_metrics = AdaptiveClusteringMetrics(
        entropy=entropy,
        variance=variance,
        selected_eps=0.0,  # HDBSCAN doesn't use eps
        silhouette_score=None,
        dbcv_score=dbcv,
        parameter_search_results=[],
    )
    
    return result, adaptive_metrics


def _standardize_points(points: np.ndarray) -> np.ndarray:
    """Standardize point coordinates for clustering."""
    scaler = StandardScaler()
    xy = points[:, :2]
    z = points[:, 2:3]
    scaled_xy = scaler.fit_transform(xy)
    normalized_z = (z - z.mean()) / (z.std() + 1e-6)
    return np.concatenate([scaled_xy, normalized_z], axis=1)


def compare_adaptive_vs_static(
    attention_map: np.ndarray,
    token_confidence: float,
    config: AnalysisConfig,
    static_eps: float = 1.3,
    static_min_samples: int = 15,
) -> Dict[str, ClusterResult]:
    """
    Compare adaptive vs static clustering on the same attention map.
    
    Args:
        attention_map: 2D attention heatmap
        token_confidence: Model confidence
        config: Analysis configuration
        static_eps: Fixed eps for baseline
        static_min_samples: Fixed min_samples for baseline
    
    Returns:
        Dictionary with 'adaptive' and 'static' ClusterResults
    """
    # Static clustering
    points = attention_to_points(attention_map)
    scaled_points = _standardize_points(points)
    
    db_static = DBSCAN(eps=static_eps, min_samples=static_min_samples)
    labels_static = db_static.fit_predict(scaled_points)
    
    n_clusters_static = len(set(labels_static)) - (1 if -1 in labels_static else 0)
    n_noise_static = int(np.sum(labels_static == -1))
    
    entropy_metrics_static = compute_attention_entropy(points[:, 2], config.entropy)
    
    static_result = ClusterResult(
        clusterer=ClustererType.DBSCAN,
        labels=labels_static,
        n_clusters=n_clusters_static,
        n_noise=n_noise_static,
        noise_ratio=n_noise_static / len(labels_static),
        average_strength=float(points[:, 2].

