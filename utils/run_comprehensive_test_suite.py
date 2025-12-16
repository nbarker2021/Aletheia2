def run_comprehensive_test_suite():
    results = {}
    
    for variety in test_varieties:
        print(f"Testing {variety.name}...")
        
        # Step 1: Compute cohomology and Hodge structure
        setup_variety_data(variety)
        
        # Step 2: Construct E8 embedding
        embedding = construct_hodge_e8_embedding(variety)
        
        # Step 3: Verify embedding properties
        consistency = verify_e8_consistency(embedding, variety)
        
        # Step 4: Test cycle construction
        cycle_results = []
        for hodge_class in variety.known_hodge_classes:
            weight_vector = embedding[hodge_class]
            constructed_cycle = realize_weight_vector_as_cycle(weight_vector, variety)
