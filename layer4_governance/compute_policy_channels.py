def compute_policy_channels(embedding):
    """Compute 8 policy channels from embedding using D8 harmonic basis"""
    v = np.array(embedding)
    
    # D8 harmonic basis (8 channels: DC, Nyquist, 3 cosine-sine pairs)
    channels = np.zeros(8)
    
    # Channel 0: DC (average)
    channels[0] = np.mean(v)
    
    # Channel 1: Nyquist (alternating pattern)
    channels[1] = np.mean([(-1)**i * v[i] for i in range(8)])
    
    # Channels 2-7: Fourier-like components
    for k in range(1, 4):  # 3 harmonic pairs
        cos_sum = sum(v[i] * np.cos(2 * np.pi * k * i / 8) for i in range(8))
        sin_sum = sum(v[i] * np.sin(2 * np.pi * k * i / 8) for i in range(8))
        channels[2*k] = cos_sum / 4
        channels[2*k+1] = sin_sum / 4
    
    return channels.tolist()

# Create overlay states for all test scenarios
for scenario in test_scenarios:
    # Initial state
    initial_state = OverlayState(
        embedding=scenario['initial_embedding'],
        channels=compute_policy_channels(scenario['initial_embedding']),
        objective_value=scenario['initial_objective'],
        iteration=0,
        domain=scenario['domain'],
        test_name=scenario['test_name']
    )
    overlay_repo.add_overlay_state(initial_state)
    
    # Final state
    final_state = OverlayState(
        embedding=scenario['final_embedding'], 
        channels=compute_policy_channels(scenario['final_embedding']),
        objective_value=scenario['final_objective'],
        iteration=scenario['iterations'],
        domain=scenario['domain'],
        test_name=scenario['test_name']
    )
    overlay_repo.add_overlay_state(final_state)

print(f"Generated {len(overlay_repo.overlay_states)} overlay states")
print(f"Dimensional scopes: {list(overlay_repo.dimensional_scopes.keys())}")
print(f"Angular views: {len(overlay_repo.angular_views)}")

# Analyze E8 distances for a sample embedding
sample_embedding = test_scenarios[0]['final_embedding']
e8_distances = overlay_repo.compute_e8_distances(sample_embedding)

print(f"\nE8 distance analysis for sample embedding {sample_embedding}:")
print("Closest 10 E8 nodes:")
for i, dist_info in enumerate(e8_distances[:10]):
    print(f"Node {dist_info.node_id}: dist={dist_info.distance:.4f}, "
          f"angle={dist_info.angular_separation:.3f}rad, "
          f"coords=[{', '.join([f'{x:4.1f}' for x in dist_info.coordinates])}]")# Create Yang-Mills figure generation script
ym_figures = """
#!/usr/bin/env python3
\"\"\"
Generate figures for Yang-Mills Mass Gap E8 proof paper
Creates all diagrams needed for main manuscript
\"\"\"

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
