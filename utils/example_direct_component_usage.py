def example_direct_component_usage():
    """Example: Using CQE components directly."""
    
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Direct Component Usage")
    print("=" * 60)
    
    # Initialize components individually
    domain_adapter = DomainAdapter()
    
    # Create a custom problem vector
    print("Creating custom problem embedding...")
    custom_vector = domain_adapter.embed_p_problem(size=75, complexity_hint=2)
    print(f"Custom vector: {custom_vector}")
    print(f"Vector norm: {np.linalg.norm(custom_vector):.4f}")
    
    # Load E₈ lattice (assuming embedding file exists)
    try:
        e8_lattice = E8Lattice("embeddings/e8_248_embedding.json")
        
        # Find nearest root
        nearest_idx, nearest_root, distance = e8_lattice.nearest_root(custom_vector)
        print(f"\nNearest E₈ root: #{nearest_idx}")
        print(f"Distance to root: {distance:.4f}")
        
        # Determine chamber
        chamber_sig, inner_prods = e8_lattice.determine_chamber(custom_vector)
        print(f"Weyl chamber: {chamber_sig}")
        print(f"Chamber inner products: {inner_prods[:4]}...")  # Show first 4
        
        # Assess embedding quality
        quality = e8_lattice.root_embedding_quality(custom_vector)
        print(f"\nEmbedding Quality:")
        print(f"  Nearest root distance: {quality['nearest_root_distance']:.4f}")
        print(f"  Chamber depth: {quality['chamber_depth']:.4f}")
        print(f"  Symmetry score: {quality['symmetry_score']:.4f}")
        print(f"  In fundamental chamber: {quality['fundamental_chamber']}")
        
    except FileNotFoundError:
        print("E₈ embedding file not found - skipping lattice operations")
    
    return custom_vector
