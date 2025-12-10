def generate_e8_pathway(problem: str, seed: int) -> Dict:
    \"\"\"Generate a random E₈ pathway for exploration.\"\"\"
    random.seed(seed)
    np.random.seed(seed)
    
    # Random E₈ configuration
    root_pattern = np.random.choice([0, 1], size=240, p=[0.9, 0.1])  # Sparse activation
    weight_vector = np.random.randn(8) * 0.5
    
    # Compute "validity scores" (simplified)
    geometric_consistency = np.random.uniform(0.3, 1.0)
    computational_evidence = np.random.uniform(0.2, 0.9) 
    novelty = np.random.uniform(0.6, 1.0)  # Most E₈ approaches are novel
    
    total_score = (geometric_consistency + computational_evidence + novelty) / 3
    
    # Generate branches if score is high enough
    branches = []
    if total_score > 0.65:
        branch_types = [
            f"{problem.lower()}_high_activity",
            f"{problem.lower()}_sparse_resonance", 
            f"{problem.lower()}_weight_dominance",
            f"{problem.lower()}_root_clustering"
        ]
        num_branches = min(int(total_score * 4), 3)  # Max 3 branches
        branches = random.sample(branch_types, num_branches)
    
    return {
        'problem': problem,
        'root_pattern': f"[{np.sum(root_pattern)} active roots]",
        'weight_vector': f"[{weight_vector[0]:.2f}, {weight_vector[1]:.2f}, ...]",
        'scores': {
            'geometric': geometric_consistency,
            'computational': computational_evidence,
            'novelty': novelty,
            'total': total_score
        },
        'branches_discovered': branches
    }
