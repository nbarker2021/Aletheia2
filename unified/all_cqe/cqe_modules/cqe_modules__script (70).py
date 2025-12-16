# Generate comprehensive overlay analysis and save as structured data

# Compute trajectory deltas (improvement vectors)
trajectory_deltas = []

for i in range(0, len(overlay_repo.overlay_states), 2):
    if i + 1 < len(overlay_repo.overlay_states):
        initial = overlay_repo.overlay_states[i]
        final = overlay_repo.overlay_states[i + 1]
        
        if initial.test_name == final.test_name:
            delta_embedding = [final.embedding[j] - initial.embedding[j] for j in range(8)]
            delta_channels = [final.channels[j] - initial.channels[j] for j in range(8)]
            delta_objective = final.objective_value - initial.objective_value
            
            trajectory_deltas.append({
                'test_name': initial.test_name,
                'domain': initial.domain,
                'delta_embedding': delta_embedding,
                'delta_channels': delta_channels, 
                'delta_objective': delta_objective,
                'iterations': final.iteration,
                'convergence_rate': -np.log(abs(delta_objective)) / final.iteration if final.iteration > 0 else 0
            })

print("Trajectory Analysis:")
print("===================")
for delta in trajectory_deltas:
    print(f"Test: {delta['test_name']}")
    print(f"  Domain: {delta['domain']}")
    print(f"  Objective improvement: {-delta['delta_objective']:.3f}")
    print(f"  Convergence rate: {delta['convergence_rate']:.3f}")
    print(f"  Embedding L2 change: {np.linalg.norm(delta['delta_embedding']):.4f}")
    print(f"  Channel L2 change: {np.linalg.norm(delta['delta_channels']):.4f}")
    print()

# Generate modulo forms analysis
print("Modulo Forms Analysis:")
print("=====================")

modulo_signatures = {}
for state in overlay_repo.overlay_states:
    e8_dists = overlay_repo.compute_e8_distances(state.embedding)
    closest_node = e8_dists[0]
    
    # Extract modulo signature pattern
    modulo_sig = closest_node.modulo_form
    if modulo_sig not in modulo_signatures:
        modulo_signatures[modulo_sig] = []
    
    modulo_signatures[modulo_sig].append({
        'test_name': state.test_name,
        'domain': state.domain,
        'iteration': state.iteration,
        'objective': state.objective_value,
        'distance_to_lattice': closest_node.distance
    })

print(f"Found {len(modulo_signatures)} unique modulo signatures")

# Show most common signatures
common_signatures = sorted(modulo_signatures.items(), 
                          key=lambda x: len(x[1]), reverse=True)[:5]

for sig, states in common_signatures:
    print(f"\nSignature: {sig}")
    print(f"  Frequency: {len(states)} states")
    print(f"  Average lattice distance: {np.mean([s['distance_to_lattice'] for s in states]):.4f}")
    print(f"  Domains: {set(s['domain'] for s in states)}")

# Generate angular clustering analysis
print("\nAngular Clustering Analysis:")
print("============================")

angular_clusters = {}
for state in overlay_repo.overlay_states:
    v = np.array(state.embedding)
    norm = np.linalg.norm(v)
    
    if norm > 1e-10:
        v_normalized = v / norm
        
        # Find dominant dimensions
        dominant_dims = [i for i, val in enumerate(v_normalized) if abs(val) > 0.3]
        cluster_key = "_".join(map(str, sorted(dominant_dims)))
        
        if cluster_key not in angular_clusters:
            angular_clusters[cluster_key] = []
        
        angular_clusters[cluster_key].append({
            'test_name': state.test_name,
            'domain': state.domain,
            'embedding': state.embedding,
            'norm': norm,
            'iteration': state.iteration
        })

for cluster, states in angular_clusters.items():
    print(f"\nCluster {cluster} (dominant dims): {len(states)} states")
    domains = [s['domain'] for s in states]
    print(f"  Domains: {set(domains)}")
    print(f"  Average norm: {np.mean([s['norm'] for s in states]):.4f}")
    
    # Check if cluster contains both initial and final states
    iterations = [s['iteration'] for s in states]
    if 0 in iterations and max(iterations) > 0:
        print(f"  Contains optimization trajectory: 0 -> {max(iterations)} iterations")

# Generate warm-start recommendations
print("\nWarm-Start Recommendations:")
print("===========================")

warm_start_data = {
    'best_initial_embeddings': {},
    'optimal_channel_priorities': {},
    'convergence_accelerators': {},
    'domain_specific_hints': {}
}

# Best initial embeddings by domain
for domain in ['audio', 'scene_graph', 'permutation', 'creative_ai', 'scaling', 'distributed']:
    domain_states = [s for s in overlay_repo.overlay_states if s.domain == domain and s.iteration > 0]
    
    if domain_states:
        # Find state with best objective value
        best_state = min(domain_states, key=lambda x: x.objective_value)
        warm_start_data['best_initial_embeddings'][domain] = {
            'embedding': best_state.embedding,
            'channels': best_state.channels,
            'objective_value': best_state.objective_value,
            'test_name': best_state.test_name
        }

# Channel priority patterns
channel_improvements = [0] * 8
channel_counts = [0] * 8

for delta in trajectory_deltas:
    for i, channel_delta in enumerate(delta['delta_channels']):
        if abs(channel_delta) > 0.01:  # Significant change
            channel_improvements[i] += abs(channel_delta)
            channel_counts[i] += 1

channel_priorities = []
for i in range(8):
    avg_improvement = channel_improvements[i] / max(channel_counts[i], 1)
    channel_priorities.append({
        'channel_id': i,
        'average_improvement': avg_improvement,
        'change_frequency': channel_counts[i]
    })

channel_priorities.sort(key=lambda x: x['average_improvement'], reverse=True)
warm_start_data['optimal_channel_priorities'] = channel_priorities

print("Channel Priority Ranking (most impactful first):")
for i, cp in enumerate(channel_priorities):
    channel_names = ['DC', 'Nyquist', 'Cos1', 'Sin1', 'Cos2', 'Sin2', 'Cos3', 'Sin3']
    print(f"  {i+1}. Channel {cp['channel_id']} ({channel_names[cp['channel_id']]}): "
          f"avg_improvement={cp['average_improvement']:.4f}, "
          f"frequency={cp['change_frequency']}")

print(f"\nGenerated warm-start repository with {len(overlay_repo.overlay_states)} states")
print(f"Covering {len(set(s.domain for s in overlay_repo.overlay_states))} domains")
print(f"With {len(trajectory_deltas)} optimization trajectories")