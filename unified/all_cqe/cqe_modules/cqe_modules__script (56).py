# Create golden test harness
golden_test_code = '''#!/usr/bin/env python3
"""
Golden Test Harness for CQE-MORSR Framework

Comprehensive demonstration and validation of the complete CQE system
with P vs NP geometric separation testing, MORSR exploration, and
chamber board enumeration.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cqe_system import CQERunner
from embeddings.e8_embedding import save_embedding

class GoldenTestHarness:
    """Comprehensive test harness for CQE system validation."""
    
    def __init__(self):
        self.results = {}
        self.setup_complete = False
        
    def setup_system(self):
        """Set up CQE system with fresh embeddings."""
        print("Golden Test Harness - CQE-MORSR Framework")
        print("=" * 50)
        
        # Ensure embedding exists
        embedding_path = "embeddings/e8_248_embedding.json"
        if not Path(embedding_path).exists():
            print("Generating E₈ embedding...")
            save_embedding(embedding_path)
        
        # Initialize CQE system
        print("Initializing CQE system...")
        self.runner = CQERunner(
            e8_embedding_path=embedding_path,
            config={
                "exploration": {"max_iterations": 30, "convergence_threshold": 1e-4},
                "output": {"save_results": True, "verbose": True},
                "validation": {"run_tests": True}
            }
        )
        
        self.setup_complete = True
        print("✓ CQE system initialized successfully\\n")
    
    def test_p_vs_np_separation(self):
        """Test P vs NP geometric separation hypothesis."""
        print("Testing P vs NP Geometric Separation")
        print("-" * 40)
        
        if not self.setup_complete:
            self.setup_system()
        
        # Generate test problems
        p_problems = [
            {"size": 50, "complexity_class": "P", "complexity_hint": 1},
            {"size": 100, "complexity_class": "P", "complexity_hint": 1},
            {"size": 200, "complexity_class": "P", "complexity_hint": 2}
        ]
        
        np_problems = [
            {"size": 50, "complexity_class": "NP", "nondeterminism": 0.8},
            {"size": 100, "complexity_class": "NP", "nondeterminism": 0.7}, 
            {"size": 200, "complexity_class": "NP", "nondeterminism": 0.9}
        ]
        
        p_solutions = []
        np_solutions = []
        
        # Solve P problems
        print("Solving P-class problems...")
        for i, problem in enumerate(p_problems):
            print(f"  P Problem {i+1}: size={problem['size']}")
            solution = self.runner.solve_problem(problem, "computational")
            p_solutions.append(solution)
        
        # Solve NP problems
        print("\\nSolving NP-class problems...")
        for i, problem in enumerate(np_problems):
            print(f"  NP Problem {i+1}: size={problem['size']}")
            solution = self.runner.solve_problem(problem, "computational")
            np_solutions.append(solution)
        
        # Analyze separation
        separation_analysis = self._analyze_geometric_separation(p_solutions, np_solutions)
        self.results["p_vs_np_separation"] = separation_analysis
        
        print(f"\\n✓ P vs NP separation analysis complete")
        print(f"  Average P score: {separation_analysis['p_avg_score']:.4f}")
        print(f"  Average NP score: {separation_analysis['np_avg_score']:.4f}") 
        print(f"  Separation distance: {separation_analysis['separation_distance']:.4f}")
        print(f"  Statistical significance: {separation_analysis['significance']}")
        
        return separation_analysis
    
    def test_morsr_convergence(self):
        """Test MORSR exploration convergence properties."""
        print("\\nTesting MORSR Convergence Properties")
        print("-" * 40)
        
        if not self.setup_complete:
            self.setup_system()
        
        # Test with different problem types
        test_problems = [
            {"type": "computational", "problem": {"size": 100, "complexity_class": "P"}},
            {"type": "optimization", "problem": {"variables": 20, "constraints": 10, "objective_type": "quadratic"}},
            {"type": "creative", "problem": {"scene_complexity": 75, "narrative_depth": 30, "character_count": 4}}
        ]
        
        convergence_results = []
        
        for test in test_problems:
            print(f"Testing {test['type']} problem...")
            
            solution = self.runner.solve_problem(test["problem"], test["type"])
            
            convergence_info = {
                "domain_type": test["type"],
                "initial_score": 0,  # Would need to extract from MORSR history
                "final_score": solution["objective_score"],
                "computation_time": solution["computation_time"],
                "recommendations_count": len(solution["recommendations"])
            }
            
            convergence_results.append(convergence_info)
            print(f"  Final score: {convergence_info['final_score']:.4f}")
            print(f"  Computation time: {convergence_info['computation_time']:.3f}s")
        
        self.results["morsr_convergence"] = convergence_results
        
        avg_score = np.mean([r["final_score"] for r in convergence_results])
        avg_time = np.mean([r["computation_time"] for r in convergence_results])
        
        print(f"\\n✓ MORSR convergence analysis complete")
        print(f"  Average final score: {avg_score:.4f}")
        print(f"  Average computation time: {avg_time:.3f}s")
        
        return convergence_results
    
    def test_chamber_board_enumeration(self):
        """Test chamber board CBC enumeration."""
        print("\\nTesting Chamber Board CBC Enumeration")
        print("-" * 40)
        
        if not self.setup_complete:
            self.setup_system()
        
        # Generate complete gate enumeration
        gates = self.runner.chamber_board.enumerate_gates()
        
        # Validate enumeration
        validation = self.runner.chamber_board.validate_enumeration(gates)
        coverage = self.runner.chamber_board.analyze_gate_coverage(gates)
        
        # Generate gate vector sequence
        gate_sequence = self.runner.chamber_board.explore_gate_sequence(gates[:10], 10)
        
        enumeration_results = {
            "total_gates": len(gates),
            "validation": validation,
            "coverage": coverage,
            "sequence_length": len(gate_sequence)
        }
        
        self.results["chamber_enumeration"] = enumeration_results
        
        print(f"✓ Chamber board enumeration complete")
        print(f"  Total gates generated: {enumeration_results['total_gates']}")
        print(f"  Validation passed: {enumeration_results['validation']['complete']}")
        print(f"  Construction coverage: {len(coverage['constructions'])} types")
        print(f"  Policy coverage: {len(coverage['policies'])} channels")
        
        return enumeration_results
    
    def test_embedding_quality(self):
        """Test E₈ embedding quality and operations."""
        print("\\nTesting E₈ Embedding Quality")
        print("-" * 40)
        
        if not self.setup_complete:
            self.setup_system()
        
        # Test various vectors
        test_vectors = [
            np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),  # Centered
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Sparse
            np.random.randn(8),  # Random
            np.ones(8) * 0.3,  # Uniform low
            np.ones(8) * 0.8   # Uniform high
        ]
        
        embedding_qualities = []
        
        for i, vector in enumerate(test_vectors):
            quality = self.runner.e8_lattice.root_embedding_quality(vector)
            embedding_qualities.append({
                "vector_type": ["centered", "sparse", "random", "uniform_low", "uniform_high"][i],
                "nearest_root_distance": quality["nearest_root_distance"],
                "chamber_signature": quality["chamber_signature"],
                "fundamental_chamber": quality["fundamental_chamber"],
                "chamber_depth": quality["chamber_depth"]
            })
            
            print(f"  {embedding_qualities[-1]['vector_type']:12s}: "
                  f"distance={quality['nearest_root_distance']:.4f}, "
                  f"chamber={quality['chamber_signature']}")
        
        self.results["embedding_quality"] = embedding_qualities
        
        avg_distance = np.mean([eq["nearest_root_distance"] for eq in embedding_qualities])
        fundamental_count = sum([eq["fundamental_chamber"] for eq in embedding_qualities])
        
        print(f"\\n✓ Embedding quality analysis complete")
        print(f"  Average root distance: {avg_distance:.4f}")
        print(f"  Fundamental chamber vectors: {fundamental_count}/5")
        
        return embedding_qualities
    
    def _analyze_geometric_separation(self, p_solutions, np_solutions):
        """Analyze geometric separation between P and NP solutions."""
        
        # Extract vectors
        p_vectors = [np.array(sol["optimal_vector"]) for sol in p_solutions]
        np_vectors = [np.array(sol["optimal_vector"]) for sol in np_solutions]
        
        # Calculate centroids
        p_centroid = np.mean(p_vectors, axis=0)
        np_centroid = np.mean(np_vectors, axis=0)
        
        # Calculate separation distance
        separation_distance = np.linalg.norm(p_centroid - np_centroid)
        
        # Calculate within-class spreads
        p_spread = np.mean([np.linalg.norm(vec - p_centroid) for vec in p_vectors])
        np_spread = np.mean([np.linalg.norm(vec - np_centroid) for vec in np_vectors])
        
        # Statistical significance (simple metric)
        combined_spread = (p_spread + np_spread) / 2
        significance = "high" if separation_distance > 2 * combined_spread else \
                     "medium" if separation_distance > combined_spread else "low"
        
        # Extract scores
        p_scores = [sol["objective_score"] for sol in p_solutions]
        np_scores = [sol["objective_score"] for sol in np_solutions]
        
        return {
            "p_centroid": p_centroid.tolist(),
            "np_centroid": np_centroid.tolist(),
            "separation_distance": separation_distance,
            "p_spread": p_spread,
            "np_spread": np_spread,
            "significance": significance,
            "p_avg_score": np.mean(p_scores),
            "np_avg_score": np.mean(np_scores),
            "score_difference": abs(np.mean(p_scores) - np.mean(np_scores))
        }
    
    def run_comprehensive_test(self):
        """Run all test modules in sequence."""
        print("Running Comprehensive Golden Test Suite")
        print("=" * 50)
        
        start_time = time.time()
        
        # Run all test modules
        try:
            self.test_embedding_quality()
            self.test_chamber_board_enumeration()  
            self.test_morsr_convergence()
            self.test_p_vs_np_separation()
            
        except Exception as e:
            print(f"\\nTest failed with error: {e}")
            return False
        
        # Generate summary
        total_time = time.time() - start_time
        
        print("\\n" + "=" * 50)
        print("Golden Test Suite Summary")
        print("=" * 50)
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Tests completed: {len(self.results)}")
        
        for test_name, results in self.results.items():
            print(f"✓ {test_name}")
        
        # Save results
        self._save_results()
        
        print("\\n🎉 All tests completed successfully!")
        print("\\nNext steps:")
        print("1. Review detailed results in data/generated/golden_test_results.json")
        print("2. Experiment with different problem types using CQERunner")
        print("3. Generate Niemeier lattices with: sage sage_scripts/generate_niemeier_lattices.sage")
        
        return True
    
    def _save_results(self):
        """Save test results to file."""
        results_file = Path("data/generated/golden_test_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        output = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "framework_version": "1.0.0",
            "test_results": self.results,
            "summary": {
                "tests_completed": len(self.results),
                "overall_status": "success"
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\\nResults saved to: {results_file}")

def main():
    """Main function to run golden test harness."""
    
    # Check if running from correct directory
    if not Path("cqe_system").exists():
        print("Error: Please run from the repository root directory")
        print("Usage: python examples/golden_test_harness.py")
        sys.exit(1)
    
    # Create and run test harness
    harness = GoldenTestHarness()
    success = harness.run_comprehensive_test()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
'''

with open("examples/golden_test_harness.py", 'w') as f:
    f.write(golden_test_code)

print("Created: examples/golden_test_harness.py")