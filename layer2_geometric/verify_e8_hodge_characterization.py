def verify_e8_hodge_characterization(embedding_map):
    verification_results = []
    for alpha, weight_vector in embedding_map.items():
        # Check if Hodge class corresponds to correct E8 weight space
        is_hodge = is_hodge_class(alpha)
        weight_space_type = classify_e8_weight_space(weight_vector)
        
        matches_prediction = (is_hodge == weight_space_type['is_hodge_type'])
