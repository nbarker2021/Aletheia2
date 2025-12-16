"""
Basic Usage Examples for CQE System

Demonstrates fundamental operations and problem-solving workflows.
"""

import numpy as np
from cqe import CQESystem
from cqe.core import E8Lattice, MORSRExplorer, CQEObjectiveFunction
from cqe.domains import DomainAdapter
from cqe.validation import ValidationFramework

def example_computational_problem():
    """Example: Solving a P vs NP classification problem."""
    
    print("=" * 60)
    print("EXAMPLE 1: Computational Problem (P vs NP)")
    print("=" * 60)
    
    # Initialize CQE system
    system = CQESystem()
    
    # Define a computational problem
    problem = {
        "type": "graph_connectivity",
        "complexity_class": "P",
        "size": 100,
        "description": "Determine if graph is connected",
        "complexity_hint": 1
    }
    
    # Solve using CQE
    solution = system.solve_problem(problem, domain_type="computational")
    
    # Display results
    print(f"Problem: {problem['description']}")
    print(f"Complexity Class: {problem['complexity_class']}")
    print(f"Problem Size: {problem['size']}")
    print(f"Objective Score: {solution['objective_score']:.6f}")
    print(f"Computation Time: {solution['computation_time']:.3f}s")
    print(f"Convergence Quality: {solution['analysis']['geometric_metrics']['convergence_quality']}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(solution['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    return solution

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

def example_creative_problem():
    """Example: Creative scene generation problem."""
    
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Creative Problem")
    print("=" * 60)
    
    # Initialize CQE system
    system = CQESystem()
    
    # Define creative problem
    problem = {
        "type": "narrative_generation",
        "scene_complexity": 60,
        "narrative_depth": 35,
        "character_count": 6,
        "description": "Generate complex narrative scene with multiple characters"
    }
    
    # Solve using CQE
    solution = system.solve_problem(problem, domain_type="creative")
    
    # Display results
    print(f"Problem: {problem['description']}")
    print(f"Scene Complexity: {problem['scene_complexity']}")
    print(f"Narrative Depth: {problem['narrative_depth']}")
    print(f"Character Count: {problem['character_count']}")
    print(f"Objective Score: {solution['objective_score']:.6f}")
    print(f"Computation Time: {solution['computation_time']:.3f}s")
    
    # Show chamber analysis
    chamber_analysis = solution['analysis']['chamber_analysis']
    print(f"\nChamber Analysis:")
    print(f"  Initial Chamber: {chamber_analysis['initial_chamber']}")
    print(f"  Optimal Chamber: {chamber_analysis['optimal_chamber']}")
    print(f"  Chamber Transition: {chamber_analysis['chamber_transition']}")
    
    return solution

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

def example_benchmark_performance():
    """Example: Benchmarking CQE performance."""
    
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Performance Benchmarking")
    print("=" * 60)
    
    # Initialize CQE system
    system = CQESystem()
    
    # Run benchmark across different problem sizes
    print("Running performance benchmark...")
    benchmark_results = system.benchmark_performance([10, 25, 50, 100])
    
    # Display benchmark results
    print(f"\nBenchmark Results:")
    print(f"Problem Sizes: {benchmark_results['problem_sizes']}")
    print(f"Computation Times: {[f'{t:.3f}s' for t in benchmark_results['computation_times']]}")
    print(f"Objective Scores: {[f'{s:.3f}' for s in benchmark_results['objective_scores']]}")
    
    # Calculate performance metrics
    sizes = benchmark_results['problem_sizes']
    times = benchmark_results['computation_times']
    scores = benchmark_results['objective_scores']
    
    print(f"\nPerformance Analysis:")
    print(f"  Average computation time: {np.mean(times):.3f}s")
    print(f"  Average objective score: {np.mean(scores):.3f}")
    print(f"  Time scaling factor: {times[-1]/times[0]:.2f}x for {sizes[-1]/sizes[0]}x size increase")
    print(f"  Score consistency: {np.std(scores):.3f} (lower is better)")
    
    return benchmark_results

def main():
    """Run all examples."""
    
    print("CQE System - Basic Usage Examples")
    print("=" * 60)
    
    try:
        # Run examples
        example_computational_problem()
        example_optimization_problem()
        example_creative_problem()
        example_direct_component_usage()
        example_validation_framework()
        example_benchmark_performance()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("This may be due to missing E₈ embedding files or other dependencies.")
        print("Please ensure all required data files are present.")

if __name__ == "__main__":
    main()
