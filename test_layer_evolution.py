from analysis.clustering import ClusteringPipeline
import numpy as np

# If your code requires a config, you can safely pass None for now
pipeline = ClusteringPipeline(config=None)

# Generate 32 fake layers, each with 8 attention heads, and 24x24 maps
fake_attns = [np.random.rand(8, 24, 24) for _ in range(32)]

# Run the new layer evolution analyzer
results = pipeline.analyze_layer_evolution(fake_attns)

# Print outputs to verify
print("\n--- TEST RESULTS ---")
print("Layer Entropies:", results["layer_entropies"])
print("Entropy Transitions:", results["entropy_transitions"])
print("Critical Layers:", results["critical_layers"])
