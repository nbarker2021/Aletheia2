# Now generate representative overlay states from the CQE session data
# These represent initial and final states across different test scenarios

# Simulate test run states based on session findings
test_scenarios = [
    {
        'domain': 'audio',
        'test_name': 'E8_Embedding_Accuracy',
        'initial_embedding': [0.2, -0.3, 0.1, 0.4, -0.2, 0.1, 0.3, -0.1],
        'final_embedding': [0.18, -0.29, 0.11, 0.39, -0.19, 0.12, 0.31, -0.09],
        'initial_objective': 0.847,
        'final_objective': 0.023,
        'iterations': 47
    },
    {
        'domain': 'scene_graph', 
        'test_name': 'Policy_Channel_Orthogonality',
        'initial_embedding': [0.5, 0.2, -0.1, 0.3, 0.1, -0.4, 0.2, 0.1],
        'final_embedding': [0.48, 0.21, -0.08, 0.32, 0.09, -0.38, 0.19, 0.11],
        'initial_objective': 1.234,
        'final_objective': 0.045,
        'iterations': 63
    },
    {
        'domain': 'permutation',
        'test_name': 'MORSR_Convergence', 
        'initial_embedding': [-0.3, 0.1, 0.4, -0.2, 0.5, 0.1, -0.1, 0.2],
        'final_embedding': [-0.31, 0.12, 0.41, -0.18, 0.52, 0.08, -0.12, 0.19],
        'initial_objective': 2.156,
        'final_objective': 0.089,
        'iterations': 82
    },
    {
        'domain': 'creative_ai',
        'test_name': 'TSP_Optimization_Quality',
        'initial_embedding': [0.1, -0.2, 0.3, 0.1, -0.1, 0.4, -0.3, 0.2],
        'final_embedding': [0.09, -0.18, 0.32, 0.12, -0.08, 0.42, -0.28, 0.21],
        'initial_objective': 3.421,
        'final_objective': 0.156,
        'iterations': 95
    },
    {
        'domain': 'scaling',
        'test_name': 'Scaling_Performance_64D',
        'initial_embedding': [0.4, 0.3, -0.2, -0.1, 0.2, -0.3, 0.1, 0.4],
        'final_embedding': [0.39, 0.31, -0.19, -0.08, 0.21, -0.29, 0.12, 0.38],
        'initial_objective': 1.876,
        'final_objective': 0.067,
        'iterations': 71
    },
    {
        'domain': 'distributed',
        'test_name': 'Distributed_MORSR_8_Nodes',
        'initial_embedding': [-0.1, 0.4, 0.2, -0.3, 0.1, 0.2, -0.4, 0.1],
        'final_embedding': [-0.09, 0.42, 0.19, -0.31, 0.12, 0.18, -0.39, 0.09],
        'initial_objective': 2.543,
        'final_objective': 0.134,
        'iterations': 58
    }
]

# Generate policy channels using harmonic decomposition
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
          f"coords=[{', '.join([f'{x:4.1f}' for x in dist_info.coordinates])}]")