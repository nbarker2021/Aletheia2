def example_tqf_governance():
    """Example: Using TQF governance with quaternary encoding."""
    
    print("=" * 60)
    print("ENHANCED EXAMPLE 1: TQF Governance System")
    print("=" * 60)
    
    # Configure TQF system
    tqf_config = TQFConfig(
        quaternary_encoding=True,
        orbit4_symmetries=True,
        crt_locking=True,
        resonant_gates=True,
        e_scalar_metrics=True,
        acceptance_thresholds={"E4": 0.0, "E6": 0.0, "E8": 0.25}
    )
    
    # Initialize enhanced system with TQF governance
    system = EnhancedCQESystem(governance_type=GovernanceType.TQF, tqf_config=tqf_config)
    
    # Define a computational problem
    problem = {
        "type": "graph_connectivity",
        "complexity_class": "P", 
        "size": 75,
        "description": "Determine graph connectivity with TQF governance"
    }
    
    # Solve using TQF governance
    solution = system.solve_problem_enhanced(problem, domain_type="computational")
    
    # Display results
    print(f"Problem: {problem['description']}")
    print(f"Governance Type: {solution['governance_type']}")
    print(f"Objective Score: {solution['objective_score']:.6f}")
    
    # Show window validation results
    print(f"\nWindow Validation:")
    for window_type, passed in solution['window_validation'].items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {window_type}: {status}")
    
    # Show scene analysis
    if solution['scene_analysis']['viewer']['hot_zones']:
        print(f"\nHot Zones Detected: {len(solution['scene_analysis']['viewer']['hot_zones'])}")
        for i, (row, col) in enumerate(solution['scene_analysis']['viewer']['hot_zones']):
            print(f"  Hot Zone {i+1}: Position ({row}, {col})")
    else:
        print(f"\nNo hot zones detected - clean solution")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(solution['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    return solution
