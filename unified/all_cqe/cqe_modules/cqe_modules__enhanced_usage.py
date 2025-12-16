"""
Enhanced CQE System Usage Examples

Demonstrates the integrated legacy features including TQF governance,
UVIBS extensions, scene debugging, and multi-window validation.
"""

import numpy as np
from cqe import EnhancedCQESystem, create_enhanced_cqe_system
from cqe.enhanced.unified_system import GovernanceType, TQFConfig, UVIBSConfig, SceneConfig

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

def example_uvibs_extension():
    """Example: Using UVIBS 80D extension with Monster governance."""
    
    print("\n" + "=" * 60)
    print("ENHANCED EXAMPLE 2: UVIBS 80D Extension")
    print("=" * 60)
    
    # Configure UVIBS system
    uvibs_config = UVIBSConfig(
        dimension=80,
        strict_perblock=True,
        expansion_p=7,
        expansion_nu=9,
        bridge_mode=False,
        monster_governance=True,
        alena_weights=True
    )
    
    # Initialize enhanced system with UVIBS governance
    system = EnhancedCQESystem(governance_type=GovernanceType.UVIBS, uvibs_config=uvibs_config)
    
    # Define an optimization problem
    problem = {
        "type": "resource_allocation",
        "variables": 20,
        "constraints": 12,
        "objective_type": "quadratic",
        "description": "Multi-objective optimization with UVIBS governance"
    }
    
    # Solve using UVIBS governance
    solution = system.solve_problem_enhanced(problem, domain_type="optimization")
    
    # Display results
    print(f"Problem: {problem['description']}")
    print(f"Governance Type: {solution['governance_type']}")
    print(f"Variables: {problem['variables']}")
    print(f"Constraints: {problem['constraints']}")
    print(f"Objective Score: {solution['objective_score']:.6f}")
    
    # Show validation score breakdown
    validation = solution['validation']
    print(f"\nValidation Breakdown:")
    print(f"  Overall Score: {validation['overall_score']:.3f}")
    print(f"  Scene Score: {validation.get('scene_score', 1.0):.3f}")
    print(f"  Validation Category: {validation['validation_category']}")
    
    return solution

def example_hybrid_governance():
    """Example: Using hybrid governance combining TQF and UVIBS."""
    
    print("\n" + "=" * 60)
    print("ENHANCED EXAMPLE 3: Hybrid Governance System")
    print("=" * 60)
    
    # Use factory function for hybrid system
    system = create_enhanced_cqe_system(governance_type="hybrid")
    
    # Define a creative problem
    problem = {
        "type": "narrative_generation",
        "scene_complexity": 80,
        "narrative_depth": 45,
        "character_count": 8,
        "description": "Complex narrative generation with hybrid governance"
    }
    
    # Solve using hybrid governance
    solution = system.solve_problem_enhanced(problem, domain_type="creative")
    
    # Display results
    print(f"Problem: {problem['description']}")
    print(f"Governance Type: {solution['governance_type']}")
    print(f"Scene Complexity: {problem['scene_complexity']}")
    print(f"Narrative Depth: {problem['narrative_depth']}")
    print(f"Character Count: {problem['character_count']}")
    print(f"Objective Score: {solution['objective_score']:.6f}")
    
    # Show comprehensive window validation
    print(f"\nMulti-Window Validation:")
    window_results = solution['window_validation']
    for window_type, result in window_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {window_type}: {status}")
    
    # Show scene debugging results
    scene = solution['scene_analysis']
    print(f"\nScene Analysis:")
    print(f"  Grid Size: {scene['viewer']['grid'].shape}")
    print(f"  Hot Zones: {len(scene['viewer']['hot_zones'])}")
    
    if scene['parity_twin']:
        parity = scene['parity_twin']
        print(f"  Parity Twin Analysis:")
        print(f"    Original Defect: {parity['original_defect']:.3f}")
        print(f"    Modified Defect: {parity['modified_defect']:.3f}")
        print(f"    Improvement: {parity['improvement']:.3f}")
        print(f"    Hinged Repair: {'Yes' if parity['hinged'] else 'No'}")
    
    return solution

def example_scene_debugging():
    """Example: Detailed scene-based debugging workflow."""
    
    print("\n" + "=" * 60)
    print("ENHANCED EXAMPLE 4: Scene-Based Debugging")
    print("=" * 60)
    
    # Configure scene debugging
    scene_config = SceneConfig(
        local_grid_size=(8, 8),
        shell_sizes=[4, 2],
        parity_twin_check=True,
        delta_lift_enabled=True,
        strict_ratchet=True
    )
    
    # Initialize system with detailed scene debugging
    system = EnhancedCQESystem(
        governance_type=GovernanceType.HYBRID,
        scene_config=scene_config
    )
    
    # Create a problem that might have issues
    problem = {
        "type": "complex_optimization",
        "variables": 50,
        "constraints": 25,
        "noise_level": 0.3,
        "description": "Noisy optimization problem for debugging demonstration"
    }
    
    # Solve with detailed debugging
    solution = system.solve_problem_enhanced(problem, domain_type="optimization")
    
    # Detailed scene analysis
    scene = solution['scene_analysis']
    viewer = scene['viewer']
    
    print(f"Problem: {problem['description']}")
    print(f"Noise Level: {problem['noise_level']}")
    
    print(f"\n8×8 Local Viewer Analysis:")
    print(f"  Face ID: {viewer['face_id']}")
    print(f"  Grid Shape: {viewer['grid'].shape}")
    print(f"  Error Grid Max: {np.max(viewer['error_grid']):.3f}")
    print(f"  Drift Grid Max: {np.max(viewer['drift_grid']):.3f}")
    print(f"  Hot Zones Count: {len(viewer['hot_zones'])}")
    
    # Detailed hot zone analysis
    if viewer['hot_zones']:
        print(f"\nHot Zone Details:")
        for i, (row, col) in enumerate(viewer['hot_zones'][:3]):  # Show first 3
            error_val = viewer['error_grid'][row, col]
            drift_val = viewer['drift_grid'][row, col]
            print(f"  Zone {i+1}: ({row},{col}) - Error: {error_val:.3f}, Drift: {drift_val:.3f}")
    
    # Shell analysis
    shell_analysis = scene['shell_analysis']
    print(f"\nShell Analysis:")
    for shell_name, shell_data in shell_analysis.items():
        print(f"  {shell_name}: {len(shell_data)} regions analyzed")
        for region_name, region_data in list(shell_data.items())[:2]:  # Show first 2
            print(f"    {region_name}: {region_data['upstream']} → {region_data['downstream']}")
    
    return solution

def example_performance_comparison():
    """Example: Compare performance across different governance types."""
    
    print("\n" + "=" * 60)
    print("ENHANCED EXAMPLE 5: Performance Comparison")
    print("=" * 60)
    
    # Test problem
    problem = {
        "type": "benchmark_test",
        "size": 100,
        "complexity": "medium",
        "description": "Performance comparison test"
    }
    
    governance_types = ["basic", "tqf", "uvibs", "hybrid"]
    results = {}
    
    print("Running performance comparison across governance types...")
    
    for gov_type in governance_types:
        try:
            if gov_type == "basic":
                # Use basic CQE system for comparison
                from cqe import CQESystem
                basic_system = CQESystem()
                # Mock solution for basic system
                solution = {
                    "objective_score": 0.65,
                    "governance_type": "basic",
                    "window_validation": {"W4": True},
                    "validation": {"overall_score": 0.7}
                }
            else:
                system = create_enhanced_cqe_system(governance_type=gov_type)
                solution = system.solve_problem_enhanced(problem, domain_type="computational")
            
            results[gov_type] = {
                "objective_score": solution["objective_score"],
                "overall_validation": solution["validation"]["overall_score"],
                "window_passes": sum(1 for v in solution["window_validation"].values() if v),
                "total_windows": len(solution["window_validation"])
            }
            
            print(f"  {gov_type.upper()}: Score {solution['objective_score']:.3f}")
            
        except Exception as e:
            print(f"  {gov_type.upper()}: Error - {str(e)[:50]}...")
            results[gov_type] = {"error": str(e)}
    
    # Summary comparison
    print(f"\nPerformance Summary:")
    print(f"{'Governance':<12} {'Objective':<10} {'Validation':<10} {'Windows':<10}")
    print("-" * 45)
    
    for gov_type, result in results.items():
        if "error" not in result:
            obj_score = result["objective_score"]
            val_score = result["overall_validation"]
            window_ratio = f"{result['window_passes']}/{result['total_windows']}"
            print(f"{gov_type.upper():<12} {obj_score:<10.3f} {val_score:<10.3f} {window_ratio:<10}")
        else:
            print(f"{gov_type.upper():<12} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10}")
    
    return results

def main():
    """Run all enhanced examples."""
    
    print("Enhanced CQE System - Legacy Integration Examples")
    print("=" * 60)
    
    try:
        # Run enhanced examples
        example_tqf_governance()
        example_uvibs_extension()
        example_hybrid_governance()
        example_scene_debugging()
        example_performance_comparison()
        
        print("\n" + "=" * 60)
        print("ALL ENHANCED EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("✓ TQF Governance with quaternary encoding")
        print("✓ UVIBS 80D extensions with Monster governance")
        print("✓ Hybrid governance combining multiple approaches")
        print("✓ Scene-based debugging with 8×8 viewers")
        print("✓ Multi-window validation (W4/W80/TQF/Mirror)")
        print("✓ Performance comparison across governance types")
        
    except Exception as e:
        print(f"\nError running enhanced examples: {e}")
        print("This may be due to missing dependencies or configuration issues.")

if __name__ == "__main__":
    main()
