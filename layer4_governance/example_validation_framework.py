def example_validation_framework():
    """Example: Using the validation framework."""
    
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Validation Framework")
    print("=" * 60)
    
    # Create a test solution
    test_vector = np.array([0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7, 0.1])
    test_problem = {"complexity_class": "P", "size": 50}
    
    # Mock analysis results
    test_analysis = {
        "embedding_quality": {
            "optimal": {
                "nearest_root_distance": 0.8,
                "chamber_depth": 0.3,
                "symmetry_score": 0.4,
                "fundamental_chamber": True
            }
        },
        "objective_breakdown": {
            "phi_total": 0.75,
            "lattice_quality": 0.8,
            "parity_consistency": 0.7,
            "chamber_stability": 0.8,
            "geometric_separation": 0.6,
            "domain_coherence": 0.7
        },
        "chamber_analysis": {
            "optimal_chamber": "11111111"
        },
        "geometric_metrics": {
            "convergence_quality": "good",
            "vector_improvement": 1.2
        }
    }
    
    # Initialize validation framework
    validator = ValidationFramework()
    
    # Run validation
    print("Running comprehensive validation...")
    validation_report = validator.validate_solution(
        test_problem, test_vector, test_analysis
    )
    
    # Display validation results
    print(f"\nValidation Results:")
    print(f"Overall Score: {validation_report['overall_score']:.3f}")
    print(f"Validation Category: {validation_report['validation_category']}")
    print(f"Validation Time: {validation_report['validation_time']:.3f}s")
    
    print(f"\nDimension Scores:")
    for dimension, scores in validation_report['dimension_scores'].items():
        print(f"  {dimension}: {scores['score']:.3f}")
    
    print(f"\nSummary:")
    print(validation_report['summary'])
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(validation_report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    return validation_report
