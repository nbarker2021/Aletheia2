def example_optimization_problem():
    """Example: Multi-objective optimization problem."""
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Optimization Problem")
    print("=" * 60)
    
    # Initialize CQE system
    system = CQESystem()
    
    # Define optimization problem
    problem = {
        "type": "resource_allocation",
        "variables": 15,
        "constraints": 8,
        "objective_type": "quadratic",
        "description": "Optimize resource allocation with quadratic costs"
    }
    
    # Solve using CQE
    solution = system.solve_problem(problem, domain_type="optimization")
    
    # Display results
    print(f"Problem: {problem['description']}")
    print(f"Variables: {problem['variables']}")
    print(f"Constraints: {problem['constraints']}")
    print(f"Objective Type: {problem['objective_type']}")
    print(f"Objective Score: {solution['objective_score']:.6f}")
    print(f"Computation Time: {solution['computation_time']:.3f}s")
    
    # Show objective function breakdown
    breakdown = solution['analysis']['objective_breakdown']
    print("\nObjective Function Breakdown:")
    print(f"  Lattice Quality: {breakdown['lattice_quality']:.3f}")
    print(f"  Parity Consistency: {breakdown['parity_consistency']:.3f}")
    print(f"  Chamber Stability: {breakdown['chamber_stability']:.3f}")
    print(f"  Geometric Separation: {breakdown['geometric_separation']:.3f}")
    print(f"  Domain Coherence: {breakdown['domain_coherence']:.3f}")
    
    return solution
