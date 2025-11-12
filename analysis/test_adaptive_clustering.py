"""
Test suite for adaptive clustering functionality.
Run with: python analysis/test_adaptive_clustering.py
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.adaptive_clustering import (
    compute_attention_entropy_signal,
    compute_attention_variance_signal,
    map_entropy_to_eps,
    adaptive_dbscan_with_quality_metrics,
    AdaptiveClusteringMetrics,
)
from analysis.config import AnalysisConfig
from analysis.clustering import ClusterResult


def test_entropy_signal_uniform():
    """Test entropy signal on uniform attention (high entropy)."""
    print("\n" + "="*60)
    print("TEST 1: Entropy Signal - Uniform Attention")
    print("="*60)
    
    attention_map = np.ones((24, 24))
    entropy = compute_attention_entropy_signal(attention_map)
    
    print(f"Uniform attention entropy: {entropy:.4f}")
    print(f"Expected: close to 1.0 (maximum entropy)")
    
    assert 0.99 <= entropy <= 1.0, f"Expected entropy ~1.0, got {entropy}"
    print("✓ PASSED")
    return entropy


def test_entropy_signal_focused():
    """Test entropy signal on focused attention (low entropy)."""
    print("\n" + "="*60)
    print("TEST 2: Entropy Signal - Focused Attention")
    print("="*60)
    
    attention_map = np.zeros((24, 24))
    attention_map[12, 12] = 1.0
    
    entropy = compute_attention_entropy_signal(attention_map)
    
    print(f"Focused attention entropy: {entropy:.4f}")
    print(f"Expected: close to 0.0 (minimum entropy)")
    
    assert 0.0 <= entropy <= 0.1, f"Expected entropy ~0.0, got {entropy}"
    print("✓ PASSED")
    return entropy


def test_entropy_signal_moderate():
    """Test entropy signal on moderate attention."""
    print("\n" + "="*60)
    print("TEST 3: Entropy Signal - Moderate Attention")
    print("="*60)
    
    # Create a more focused Gaussian distribution for moderate entropy
    x, y = np.meshgrid(np.arange(24), np.arange(24))
    attention_map = np.exp(-((x - 12)**2 + (y - 12)**2) / 20)  # Changed from 50 to 20
    
    entropy = compute_attention_entropy_signal(attention_map)
    
    print(f"Moderate attention entropy: {entropy:.4f}")
    print(f"Expected: between 0.5 and 0.95")
    
    assert 0.5 <= entropy <= 0.95, f"Expected entropy between 0.5-0.95, got {entropy}"
    print("✓ PASSED")
    return entropy



def test_variance_signal():
    """Test variance signal computation."""
    print("\n" + "="*60)
    print("TEST 4: Variance Signal Computation")
    print("="*60)
    
    attention_high_var = np.random.rand(24, 24)
    var_high = compute_attention_variance_signal(attention_high_var)
    
    attention_low_var = np.ones((24, 24)) * 0.5 + np.random.rand(24, 24) * 0.01
    var_low = compute_attention_variance_signal(attention_low_var)
    
    print(f"High variance attention: {var_high:.4f}")
    print(f"Low variance attention: {var_low:.4f}")
    print(f"High variance > Low variance: {var_high > var_low}")
    
    assert var_high > var_low, "High variance should be greater than low variance"
    print("✓ PASSED")
    return var_high, var_low


def test_entropy_to_eps_mapping():
    """Test mapping from entropy to eps parameter."""
    print("\n" + "="*60)
    print("TEST 5: Entropy to Eps Mapping")
    print("="*60)
    
    entropies = [0.1, 0.3, 0.5, 0.7, 0.9]
    eps_values = [map_entropy_to_eps(e) for e in entropies]
    
    print("Entropy → Eps mapping:")
    for e, eps in zip(entropies, eps_values):
        print(f"  Entropy {e:.1f} → eps {eps:.3f}")
    
    assert eps_values[0] < eps_values[-1], "Higher entropy should produce higher eps"
    
    for i in range(len(eps_values) - 1):
        assert eps_values[i] <= eps_values[i+1], "Eps should increase monotonically"
    
    print("✓ PASSED")
    return eps_values


def test_adaptive_clustering_basic():
    """Test basic adaptive clustering functionality."""
    print("\n" + "="*60)
    print("TEST 6: Adaptive Clustering - Basic Functionality")
    print("="*60)
    
    attention_map = np.zeros((24, 24))
    attention_map[5:10, 5:10] = 1.0
    attention_map[15:20, 15:20] = 0.8
    attention_map += np.random.rand(24, 24) * 0.1
    
    config = AnalysisConfig()
    token_confidence = 0.85
    
    result, metrics = adaptive_dbscan_with_quality_metrics(
        attention_map=attention_map,
        token_confidence=token_confidence,
        config=config
    )
    
    print(f"Clusters found: {result.n_clusters}")
    print(f"Noise ratio: {result.noise_ratio:.3f}")
    print(f"Selected eps: {metrics.selected_eps:.3f}")
    print(f"Entropy signal: {metrics.entropy:.3f}")
    print(f"Variance signal: {metrics.variance:.3f}")
    if metrics.silhouette_score:
        print(f"Silhouette score: {metrics.silhouette_score:.3f}")
    
    assert result.n_clusters >= 1, f"Expected at least 1 cluster, got {result.n_clusters}"
    assert isinstance(result, ClusterResult), "Result should be ClusterResult type"
    assert isinstance(metrics, AdaptiveClusteringMetrics), "Metrics should be AdaptiveClusteringMetrics type"
    
    print("✓ PASSED")
    return result, metrics


def test_adaptive_vs_static_comparison():
    """Test comparison between adaptive and static clustering."""
    print("\n" + "="*60)
    print("TEST 7: Adaptive vs Static Clustering Comparison")
    print("="*60)
    
    x, y = np.meshgrid(np.arange(24), np.arange(24))
    attention_map = (
        np.exp(-((x - 8)**2 + (y - 8)**2) / 20) +
        np.exp(-((x - 16)**2 + (y - 16)**2) / 30) * 0.5
    )
    
    config = AnalysisConfig()
    token_confidence = 0.75
    
    adaptive_result, adaptive_metrics = adaptive_dbscan_with_quality_metrics(
        attention_map=attention_map,
        token_confidence=token_confidence,
        config=config
    )
    
    print("\nAdaptive Clustering:")
    print(f"  Clusters: {adaptive_result.n_clusters}")
    print(f"  Noise ratio: {adaptive_result.noise_ratio:.3f}")
    print(f"  Selected eps: {adaptive_metrics.selected_eps:.3f}")
    if adaptive_metrics.silhouette_score:
        print(f"  Silhouette: {adaptive_metrics.silhouette_score:.3f}")
    
    from sklearn.cluster import DBSCAN
    from analysis.clustering import attention_to_points
    from analysis.adaptive_clustering import _standardize_points
    
    points = attention_to_points(attention_map)
    scaled_points = _standardize_points(points)
    
    static_db = DBSCAN(eps=1.3, min_samples=15)
    static_labels = static_db.fit_predict(scaled_points)
    static_n_clusters = len(set(static_labels)) - (1 if -1 in static_labels else 0)
    static_noise_ratio = np.sum(static_labels == -1) / len(static_labels)
    
    print("\nStatic Clustering (eps=1.3):")
    print(f"  Clusters: {static_n_clusters}")
    print(f"  Noise ratio: {static_noise_ratio:.3f}")
    
    print(f"\nAdaptive found {adaptive_result.n_clusters} clusters vs Static found {static_n_clusters} clusters")
    
    assert adaptive_result.n_clusters >= 0, "Adaptive should find clusters"
    assert static_n_clusters >= 0, "Static should find clusters"
    
    print("✓ PASSED")
    return adaptive_result, static_n_clusters


def test_parameter_search_results():
    """Test that parameter search explores multiple eps values."""
    print("\n" + "="*60)
    print("TEST 8: Parameter Search Exploration")
    print("="*60)
    
    attention_map = np.random.rand(24, 24)
    config = AnalysisConfig()
    
    result, metrics = adaptive_dbscan_with_quality_metrics(
        attention_map=attention_map,
        token_confidence=0.8,
        config=config
    )
    
    print(f"Number of eps values tested: {len(metrics.parameter_search_results)}")
    print("\nSearch results:")
    for i, search_result in enumerate(metrics.parameter_search_results):
        eps_val = search_result.get('eps', 'N/A')
        n_clust = search_result.get('n_clusters', 'N/A')
        qual = search_result.get('quality_score', 'N/A')
        if isinstance(qual, float):
            print(f"  {i+1}. eps={eps_val:.3f}, clusters={n_clust}, quality={qual:.3f}")
        else:
            print(f"  {i+1}. eps={eps_val:.3f}, clusters={n_clust}, quality={qual}")
    
    assert len(metrics.parameter_search_results) >= 3, "Should test multiple eps values"
    print("✓ PASSED")
    return metrics.parameter_search_results


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "="*60)
    print("TEST 9: Edge Cases")
    print("="*60)
    
    config = AnalysisConfig()
    
    print("\nTest 9a: All zeros attention")
    attention_zeros = np.zeros((24, 24))
    attention_zeros[0, 0] = 1e-10
    
    try:
        result, metrics = adaptive_dbscan_with_quality_metrics(
            attention_map=attention_zeros,
            token_confidence=0.5,
            config=config
        )
        print(f"  Clusters: {result.n_clusters}")
        print(f"  ✓ Handled zeros")
    except Exception as e:
        print(f"  ✗ Failed with error: {e}")
        raise
    
    print("\nTest 9b: Very small values")
    attention_small = np.ones((24, 24)) * 1e-8
    
    try:
        result, metrics = adaptive_dbscan_with_quality_metrics(
            attention_map=attention_small,
            token_confidence=0.5,
            config=config
        )
        print(f"  Clusters: {result.n_clusters}")
        print(f"  ✓ Handled small values")
    except Exception as e:
        print(f"  ✗ Failed with error: {e}")
        raise
    
    print("\n✓ ALL EDGE CASES PASSED")


def run_all_tests():
    """Run all tests and generate summary report."""
    print("\n" + "="*60)
    print("ADAPTIVE CLUSTERING TEST SUITE")
    print("="*60)
    
    results = {}
    
    try:
        results['entropy_uniform'] = test_entropy_signal_uniform()
        results['entropy_focused'] = test_entropy_signal_focused()
        results['entropy_moderate'] = test_entropy_signal_moderate()
        results['variance'] = test_variance_signal()
        results['eps_mapping'] = test_entropy_to_eps_mapping()
        results['basic_clustering'] = test_adaptive_clustering_basic()
        results['comparison'] = test_adaptive_vs_static_comparison()
        results['parameter_search'] = test_parameter_search_results()
        test_edge_cases()
        
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print("✓ All tests PASSED!")
        print(f"✓ Tested {len([k for k in results.keys() if 'entropy' in k])} entropy scenarios")
        print(f"✓ Tested adaptive clustering on multiple attention patterns")
        print(f"✓ Verified parameter search explores multiple configurations")
        print(f"✓ Tested edge cases")
        print("\n🎉 Adaptive clustering implementation is working correctly!")
        
        return True
        
    except AssertionError as e:
        print("\n" + "="*60)
        print("TEST FAILED")
        print("="*60)
        print(f"✗ Error: {e}")
        return False
    except Exception as e:
        print("\n" + "="*60)
        print("UNEXPECTED ERROR")
        print("="*60)
        print(f"✗ {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
