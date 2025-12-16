"""
CQE Runner - Main Orchestrator

Coordinates all CQE system components for end-to-end problem solving:
domain adaptation, E₈ embedding, MORSR exploration, and result analysis.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time

from .domain_adapter import DomainAdapter
from .e8_lattice import E8Lattice
from .parity_channels import ParityChannels
from .objective_function import CQEObjectiveFunction
from .morsr_explorer import MORSRExplorer
from .chamber_board import ChamberBoard

class CQERunner:
    """Main orchestrator for CQE system operations."""

    def __init__(self, 
                 e8_embedding_path: str = "embeddings/e8_248_embedding.json",
                 config: Optional[Dict] = None):

        print("Initializing CQE system...")

        # Load configuration
        self.config = config or self._default_config()

        # Initialize components
        self.domain_adapter = DomainAdapter()
        self.e8_lattice = E8Lattice(e8_embedding_path)
        self.parity_channels = ParityChannels()

        self.objective_function = CQEObjectiveFunction(
            self.e8_lattice, self.parity_channels
        )

        self.morsr_explorer = MORSRExplorer(
            self.objective_function, self.parity_channels
        )

        self.chamber_board = ChamberBoard()

        print("CQE system initialization complete")

    def _default_config(self) -> Dict:
        """Default configuration for CQE system."""
        return {
            "exploration": {
                "max_iterations": 50,
                "convergence_threshold": 1e-4,
                "pulse_count": 10
            },
            "output": {
                "save_results": True,
                "results_dir": "data/generated",
                "verbose": True
            },
            "validation": {
                "run_tests": True,
                "comparison_baseline": True
            }
        }

    def solve_problem(self, 
                     problem_description: Dict,
                     domain_type: str = "computational") -> Dict[str, Any]:
        """
        Solve a problem using the complete CQE pipeline.

        Args:
            problem_description: Dictionary describing the problem
            domain_type: Type of domain (computational, optimization, creative)

        Returns:
            Complete solution with analysis and recommendations
        """

        start_time = time.time()

        print(f"\nSolving {domain_type} problem...")
        if self.config["output"]["verbose"]:
            print(f"Problem description: {problem_description}")

        # Phase 1: Domain Adaptation
        initial_vector = self._adapt_problem_to_e8(problem_description, domain_type)

        # Phase 2: Extract Reference Channels
        reference_channels = self.parity_channels.extract_channels(initial_vector)

        # Phase 3: MORSR Exploration
        domain_context = {
            "domain_type": domain_type,
            "problem_size": problem_description.get("size", 100),
            "complexity_class": problem_description.get("complexity_class", "unknown")
        }

        optimal_vector, optimal_channels, best_score = self.morsr_explorer.explore(
            initial_vector,
            reference_channels,
            max_iterations=self.config["exploration"]["max_iterations"],
            domain_context=domain_context,
            convergence_threshold=self.config["exploration"]["convergence_threshold"]
        )

        # Phase 4: Analysis and Interpretation
        analysis = self._analyze_solution(
            initial_vector, optimal_vector, optimal_channels, 
            best_score, domain_context
        )

        # Phase 5: Generate Recommendations
        recommendations = self._generate_recommendations(
            analysis, problem_description, domain_type
        )

        # Compile complete solution
        solution = {
            "problem": problem_description,
            "domain_type": domain_type,
            "initial_vector": initial_vector.tolist(),
            "optimal_vector": optimal_vector.tolist(),
            "initial_channels": reference_channels,
            "optimal_channels": optimal_channels,
            "objective_score": best_score,
            "analysis": analysis,
            "recommendations": recommendations,
            "computation_time": time.time() - start_time,
            "metadata": {
                "cqe_version": "1.0.0",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }

        # Save results if configured
        if self.config["output"]["save_results"]:
            self._save_solution(solution)

        return solution

    def _adapt_problem_to_e8(self, problem_description: Dict, domain_type: str) -> np.ndarray:
        """Adapt problem to E₈ configuration space."""

        if domain_type == "computational":
            if "complexity_class" in problem_description:
                if problem_description["complexity_class"] == "P":
                    return self.domain_adapter.embed_p_problem(
                        problem_description.get("size", 100),
                        problem_description.get("complexity_hint", 1)
                    )
                elif problem_description["complexity_class"] == "NP":
                    return self.domain_adapter.embed_np_problem(
                        problem_description.get("size", 100),
                        problem_description.get("nondeterminism", 0.8)
                    )

        elif domain_type == "optimization":
            return self.domain_adapter.embed_optimization_problem(
                problem_description.get("variables", 10),
                problem_description.get("constraints", 5),
                problem_description.get("objective_type", "linear")
            )

        elif domain_type == "creative":
            return self.domain_adapter.embed_scene_problem(
                problem_description.get("scene_complexity", 50),
                problem_description.get("narrative_depth", 25),
                problem_description.get("character_count", 5)
            )

        else:
            # Fallback: hash-based embedding
            problem_str = json.dumps(problem_description, sort_keys=True)
            return self.domain_adapter.hash_to_features(problem_str)

    def _analyze_solution(self, 
                         initial_vector: np.ndarray,
                         optimal_vector: np.ndarray,
                         optimal_channels: Dict[str, float],
                         best_score: float,
                         domain_context: Dict) -> Dict[str, Any]:
        """Analyze the solution quality and characteristics."""

        # E₈ embedding analysis
        initial_quality = self.e8_lattice.root_embedding_quality(initial_vector)
        optimal_quality = self.e8_lattice.root_embedding_quality(optimal_vector)

        # Objective function breakdown
        score_breakdown = self.objective_function.evaluate(
            optimal_vector, optimal_channels, domain_context
        )

        # Chamber analysis
        initial_chamber, _ = self.e8_lattice.determine_chamber(initial_vector)
        optimal_chamber, _ = self.e8_lattice.determine_chamber(optimal_vector)

        # Improvement metrics
        improvement = np.linalg.norm(optimal_vector - initial_vector)
        chamber_distance = self.e8_lattice.chamber_distance(initial_vector, optimal_vector)

        return {
            "embedding_quality": {
                "initial": initial_quality,
                "optimal": optimal_quality,
                "improvement": optimal_quality["nearest_root_distance"] - initial_quality["nearest_root_distance"]
            },
            "objective_breakdown": score_breakdown,
            "chamber_analysis": {
                "initial_chamber": initial_chamber,
                "optimal_chamber": optimal_chamber,
                "chamber_transition": initial_chamber != optimal_chamber
            },
            "geometric_metrics": {
                "vector_improvement": float(improvement),
                "chamber_distance": float(chamber_distance),
                "convergence_quality": "excellent" if best_score > 0.8 else "good" if best_score > 0.6 else "fair"
            }
        }

    def _generate_recommendations(self, 
                                analysis: Dict,
                                problem_description: Dict,
                                domain_type: str) -> List[str]:
        """Generate actionable recommendations based on analysis."""

        recommendations = []

        # Embedding quality recommendations
        embedding_quality = analysis["embedding_quality"]["optimal"]
        if embedding_quality["nearest_root_distance"] > 1.0:
            recommendations.append(
                "Consider refining problem representation - vector is far from E₈ roots"
            )

        # Objective score recommendations  
        score_breakdown = analysis["objective_breakdown"]
        if score_breakdown["parity_consistency"] < 0.5:
            recommendations.append(
                "Improve parity channel consistency through additional repair iterations"
            )

        if score_breakdown["chamber_stability"] < 0.6:
            recommendations.append(
                "Enhance chamber stability - consider alternative projection methods"
            )

        # Domain-specific recommendations
        if domain_type == "computational":
            complexity_class = problem_description.get("complexity_class", "unknown")
            if complexity_class in ["P", "NP"]:
                separation_score = score_breakdown["geometric_separation"]
                if separation_score < 0.7:
                    recommendations.append(
                        f"Geometric separation suggests potential misclassification of {complexity_class} problem"
                    )

        # Performance recommendations
        convergence = analysis["geometric_metrics"]["convergence_quality"]
        if convergence == "fair":
            recommendations.append(
                "Increase MORSR iterations or adjust exploration parameters for better convergence"
            )

        # Chamber transition recommendations
        if analysis["chamber_analysis"]["chamber_transition"]:
            recommendations.append(
                "Chamber transition occurred - validate solution stability across chambers"
            )

        if not recommendations:
            recommendations.append("Solution quality is excellent - no specific improvements needed")

        return recommendations

    def _save_solution(self, solution: Dict):
        """Save solution to configured output directory."""

        results_dir = Path(self.config["output"]["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        domain_type = solution["domain_type"]
        filename = f"cqe_solution_{domain_type}_{timestamp}.json"

        filepath = results_dir / filename

        with open(filepath, 'w') as f:
            json.dump(solution, f, indent=2)

        print(f"Solution saved to: {filepath}")

    def run_test_suite(self) -> Dict[str, bool]:
        """Run comprehensive test suite on CQE system."""

        print("\nRunning CQE test suite...")

        tests = {
            "e8_embedding_load": False,
            "domain_adaptation": False,
            "parity_extraction": False,
            "objective_evaluation": False,
            "morsr_exploration": False,
            "chamber_enumeration": False
        }

        try:
            # Test E₈ embedding
            test_vector = np.random.randn(8)
            nearest_idx, nearest_root, distance = self.e8_lattice.nearest_root(test_vector)
            tests["e8_embedding_load"] = distance >= 0

            # Test domain adaptation
            test_problem = {"size": 50, "complexity_class": "P"}
            adapted = self.domain_adapter.embed_p_problem(50, 1)
            tests["domain_adaptation"] = len(adapted) == 8

            # Test parity extraction
            channels = self.parity_channels.extract_channels(adapted)
            tests["parity_extraction"] = len(channels) == 8

            # Test objective evaluation
            scores = self.objective_function.evaluate(adapted, channels)
            tests["objective_evaluation"] = "phi_total" in scores

            # Test MORSR exploration
            result_vec, result_ch, result_score = self.morsr_explorer.explore(
                adapted, channels, max_iterations=5
            )
            tests["morsr_exploration"] = len(result_vec) == 8

            # Test chamber enumeration
            gates = self.chamber_board.enumerate_gates(max_count=10)
            tests["chamber_enumeration"] = len(gates) == 10

        except Exception as e:
            print(f"Test suite error: {e}")

        # Report results
        passed = sum(tests.values())
        total = len(tests)
        print(f"Test suite complete: {passed}/{total} tests passed")

        for test_name, result in tests.items():
            status = "PASS" if result else "FAIL"
            print(f"  {test_name}: {status}")

        return tests

    def benchmark_performance(self, problem_sizes: List[int] = [10, 50, 100, 200]) -> Dict:
        """Benchmark CQE performance across different problem sizes."""

        print("\nBenchmarking CQE performance...")

        benchmark_results = {
            "problem_sizes": problem_sizes,
            "computation_times": [],
            "objective_scores": [],
            "convergence_iterations": []
        }

        for size in problem_sizes:
            print(f"  Benchmarking problem size: {size}")

            # Create test problem
            test_problem = {
                "size": size,
                "complexity_class": "P",
                "complexity_hint": 1
            }

            # Solve and measure performance
            start_time = time.time()
            solution = self.solve_problem(test_problem, "computational")
            computation_time = time.time() - start_time

            # Record metrics
            benchmark_results["computation_times"].append(computation_time)
            benchmark_results["objective_scores"].append(solution["objective_score"])

            # Note: convergence_iterations would need to be extracted from MORSR history
            # For now, using a placeholder
            benchmark_results["convergence_iterations"].append(25)  # Placeholder

        return benchmark_results
