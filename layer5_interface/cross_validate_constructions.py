def cross_validate_constructions(hodge_class, variety, num_trials=5):
    # Multiple independent constructions of same algebraic cycle
    constructions = []
    
    for trial in range(num_trials):
        # Use slightly different numerical parameters
        perturbed_embedding = perturb_embedding(construct_hodge_e8_embedding(variety))
        weight_vector = perturbed_embedding[hodge_class]
        cycle = realize_weight_vector_as_cycle(weight_vector, variety)
        constructions.append(cycle)
    
    # Verify all constructions give same cohomology class
    cohomology_classes = [compute_cohomology_class(cycle, variety) 
                         for cycle in constructions]
    
