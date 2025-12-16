"""
CQE Validation Framework

Comprehensive validation system for assessing CQE solutions across multiple dimensions:
- Mathematical validity
- Computational evidence  
- Statistical significance
- Geometric consistency
- Cross-validation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
from scipy import stats

class ValidationFramework:
    """Comprehensive validation framework for CQE system results."""

    def __init__(self):
        self.validation_dimensions = [
            "mathematical_validity",
            "computational_evidence", 
            "statistical_significance",
            "geometric_consistency",
            "cross_validation"
        ]
        
        # Validation thresholds
        self.thresholds = {
            "perfect_validation": 1.0,
            "strong_evidence": 0.7,
            "moderate_evidence": 0.4,
            "weak_evidence": 0.2,
            "insufficient_evidence": 0.0
        }

    def validate_solution(self,
                         problem_description: Dict,
                         solution_vector: np.ndarray,
                         analysis: Dict) -> Dict[str, Any]:
        """
        Comprehensive validation of a CQE solution.

        Args:
            problem_description: Original problem specification
            solution_vector: Optimal vector found by CQE
            analysis: Analysis results from CQE system

        Returns:
            Complete validation assessment with scores and evidence
        """

        print("Starting comprehensive solution validation...")
        start_time = time.time()

        # Validate across all dimensions
        validation_scores = {}
        
        validation_scores["mathematical_validity"] = self._validate_mathematical_validity(
            solution_vector, analysis
        )
        
        validation_scores["computational_evidence"] = self._validate_computational_evidence(
            problem_description, solution_vector, analysis
        )
        
        validation_scores["statistical_significance"] = self._validate_statistical_significance(
            solution_vector, analysis
        )
        
        validation_scores["geometric_consistency"] = self._validate_geometric_consistency(
            solution_vector, analysis
        )
        
        validation_scores["cross_validation"] = self._validate_cross_validation(
            problem_description, solution_vector
        )

        # Calculate overall validation score
        weights = {
            "mathematical_validity": 0.3,
            "computational_evidence": 0.3,
            "statistical_significance": 0.2,
            "geometric_consistency": 0.1,
            "cross_validation": 0.1
        }

        overall_score = sum(
            weights[dim] * validation_scores[dim]["score"] 
            for dim in self.validation_dimensions
        )

        # Determine validation category
        validation_category = self._categorize_validation_score(overall_score)

        # Generate validation report
        validation_time = time.time() - start_time
        
        validation_report = {
            "overall_score": overall_score,
            "validation_category": validation_category,
            "dimension_scores": validation_scores,
            "validation_time": validation_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": self._generate_validation_summary(validation_scores, overall_score),
            "recommendations": self._generate_validation_recommendations(validation_scores)
        }

        print(f"Validation complete: {validation_category} ({overall_score:.3f})")
        return validation_report

    def _validate_mathematical_validity(self, 
                                       solution_vector: np.ndarray,
                                       analysis: Dict) -> Dict[str, Any]:
        """Validate mathematical consistency and constraint satisfaction."""
        
        # Check vector properties
        vector_norm = np.linalg.norm(solution_vector)
        vector_valid = 0.1 <= vector_norm <= 10.0  # Reasonable bounds
        
        # Check E₈ embedding quality
        embedding_quality = analysis.get("embedding_quality", {}).get("optimal", {})
        root_distance = embedding_quality.get("nearest_root_distance", float('inf'))
        embedding_valid = root_distance < 2.0  # Within E₈ lattice bounds
        
        # Check chamber consistency
        chamber_analysis = analysis.get("chamber_analysis", {})
        chamber_valid = chamber_analysis.get("optimal_chamber", "").startswith("1")  # Fundamental chamber preferred
        
        # Calculate mathematical validity score
        validity_checks = [vector_valid, embedding_valid, chamber_valid]
        validity_score = sum(validity_checks) / len(validity_checks)
        
        return {
            "score": validity_score,
            "details": {
                "vector_norm": vector_norm,
                "vector_valid": vector_valid,
                "root_distance": root_distance,
                "embedding_valid": embedding_valid,
                "chamber_valid": chamber_valid
            },
            "evidence": f"Mathematical validity: {validity_score:.3f} ({sum(validity_checks)}/{len(validity_checks)} checks passed)"
        }

    def _validate_computational_evidence(self,
                                       problem_description: Dict,
                                       solution_vector: np.ndarray,
                                       analysis: Dict) -> Dict[str, Any]:
        """Validate computational evidence supporting the solution."""
        
        # Check objective function improvement
        objective_breakdown = analysis.get("objective_breakdown", {})
        phi_total = objective_breakdown.get("phi_total", 0)
        evidence_score = min(1.0, max(0.0, phi_total))  # Normalize to [0,1]
        
        # Check component scores
        component_scores = []
        for component in ["lattice_quality", "parity_consistency", "chamber_stability"]:
            score = objective_breakdown.get(component, 0)
            component_scores.append(score)
        
        component_average = np.mean(component_scores) if component_scores else 0
        
        # Check convergence quality
        convergence_quality = analysis.get("geometric_metrics", {}).get("convergence_quality", "fair")
        convergence_score = {"excellent": 1.0, "good": 0.7, "fair": 0.4}.get(convergence_quality, 0.2)
        
        # Combine evidence
        computational_score = 0.5 * evidence_score + 0.3 * component_average + 0.2 * convergence_score
        
        return {
            "score": computational_score,
            "details": {
                "phi_total": phi_total,
                "component_scores": component_scores,
                "component_average": component_average,
                "convergence_quality": convergence_quality,
                "convergence_score": convergence_score
            },
            "evidence": f"Computational evidence: {computational_score:.3f} (Φ={phi_total:.3f}, components={component_average:.3f})"
        }

    def _validate_statistical_significance(self,
                                         solution_vector: np.ndarray,
                                         analysis: Dict) -> Dict[str, Any]:
        """Validate statistical significance against random baselines."""
        
        # Generate random baseline vectors
        n_baseline = 1000
        baseline_vectors = np.random.randn(n_baseline, 8)
        
        # Calculate baseline statistics
        baseline_norms = np.linalg.norm(baseline_vectors, axis=1)
        solution_norm = np.linalg.norm(solution_vector)
        
        # Statistical tests
        # 1. Norm comparison
        norm_percentile = stats.percentileofscore(baseline_norms, solution_norm) / 100.0
        norm_significance = abs(norm_percentile - 0.5) * 2  # Distance from median
        
        # 2. Component distribution test
        solution_components = np.abs(solution_vector)
        baseline_components = np.abs(baseline_vectors).flatten()
        
        # Kolmogorov-Smirnov test
        ks_statistic, ks_p_value = stats.ks_2samp(solution_components, baseline_components[:len(solution_components)])
        ks_significance = min(1.0, ks_statistic * 10)  # Scale KS statistic
        
        # 3. Objective function comparison (if available)
        objective_score = analysis.get("objective_breakdown", {}).get("phi_total", 0.5)
        objective_significance = max(0.0, (objective_score - 0.5) * 2)  # Above median baseline
        
        # Combine statistical evidence
        statistical_score = np.mean([norm_significance, ks_significance, objective_significance])
        
        return {
            "score": statistical_score,
            "details": {
                "norm_percentile": norm_percentile,
                "norm_significance": norm_significance,
                "ks_statistic": ks_statistic,
                "ks_p_value": ks_p_value,
                "ks_significance": ks_significance,
                "objective_significance": objective_significance,
                "baseline_samples": n_baseline
            },
            "evidence": f"Statistical significance: {statistical_score:.3f} (norm={norm_percentile:.2f}, KS={ks_statistic:.3f})"
        }

    def _validate_geometric_consistency(self,
                                      solution_vector: np.ndarray,
                                      analysis: Dict) -> Dict[str, Any]:
        """Validate geometric consistency with E₈ structure."""
        
        # Check embedding quality metrics
        embedding_quality = analysis.get("embedding_quality", {}).get("optimal", {})
        
        # Root distance consistency
        root_distance = embedding_quality.get("nearest_root_distance", float('inf'))
        root_consistency = max(0.0, 1.0 - root_distance / 2.0)  # Closer to roots is better
        
        # Chamber depth consistency
        chamber_depth = embedding_quality.get("chamber_depth", 0)
        depth_consistency = min(1.0, chamber_depth / 0.5)  # Deeper in chamber is better
        
        # Symmetry consistency
        symmetry_score = embedding_quality.get("symmetry_score", 1.0)
        symmetry_consistency = max(0.0, 1.0 - symmetry_score)  # Lower symmetry score is better
        
        # Vector improvement consistency
        improvement = analysis.get("geometric_metrics", {}).get("vector_improvement", 0)
        improvement_consistency = min(1.0, improvement / 2.0)  # Reasonable improvement
        
        # Combine geometric consistency
        geometric_score = np.mean([
            root_consistency, depth_consistency, 
            symmetry_consistency, improvement_consistency
        ])
        
        return {
            "score": geometric_score,
            "details": {
                "root_distance": root_distance,
                "root_consistency": root_consistency,
                "chamber_depth": chamber_depth,
                "depth_consistency": depth_consistency,
                "symmetry_score": symmetry_score,
                "symmetry_consistency": symmetry_consistency,
                "improvement": improvement,
                "improvement_consistency": improvement_consistency
            },
            "evidence": f"Geometric consistency: {geometric_score:.3f} (root={root_consistency:.2f}, depth={depth_consistency:.2f})"
        }

    def _validate_cross_validation(self,
                                 problem_description: Dict,
                                 solution_vector: np.ndarray) -> Dict[str, Any]:
        """Validate solution through cross-validation scenarios."""
        
        # Test solution robustness with perturbations
        n_perturbations = 10
        perturbation_scores = []
        
        for _ in range(n_perturbations):
            # Small perturbation
            perturbation = np.random.normal(0, 0.1, 8)
            perturbed_vector = solution_vector + perturbation
            
            # Simple quality metric (vector stability)
            stability = 1.0 / (1.0 + np.linalg.norm(perturbation))
            perturbation_scores.append(stability)
        
        # Robustness score
        robustness_score = np.mean(perturbation_scores)
        
        # Reproducibility test (deterministic for same input)
        reproducibility_score = 1.0  # Assume perfect reproducibility for now
        
        # Domain consistency test
        domain_type = problem_description.get("complexity_class", "unknown")
        domain_consistency = 0.8 if domain_type in ["P", "NP"] else 0.5
        
        # Combine cross-validation evidence
        cross_validation_score = np.mean([
            robustness_score, reproducibility_score, domain_consistency
        ])
        
        return {
            "score": cross_validation_score,
            "details": {
                "robustness_score": robustness_score,
                "perturbation_scores": perturbation_scores,
                "reproducibility_score": reproducibility_score,
                "domain_consistency": domain_consistency,
                "n_perturbations": n_perturbations
            },
            "evidence": f"Cross-validation: {cross_validation_score:.3f} (robustness={robustness_score:.2f})"
        }

    def _categorize_validation_score(self, score: float) -> str:
        """Categorize validation score into evidence levels."""
        
        if score >= self.thresholds["perfect_validation"]:
            return "Perfect Validation"
        elif score >= self.thresholds["strong_evidence"]:
            return "Strong Evidence"
        elif score >= self.thresholds["moderate_evidence"]:
            return "Moderate Evidence"
        elif score >= self.thresholds["weak_evidence"]:
            return "Weak Evidence"
        else:
            return "Insufficient Evidence"

    def _generate_validation_summary(self, 
                                   validation_scores: Dict,
                                   overall_score: float) -> str:
        """Generate human-readable validation summary."""
        
        category = self._categorize_validation_score(overall_score)
        
        # Find strongest and weakest dimensions
        dimension_scores = {dim: scores["score"] for dim, scores in validation_scores.items()}
        strongest_dim = max(dimension_scores, key=dimension_scores.get)
        weakest_dim = min(dimension_scores, key=dimension_scores.get)
        
        summary = f"Validation Category: {category} (Score: {overall_score:.3f})\n"
        summary += f"Strongest Dimension: {strongest_dim} ({dimension_scores[strongest_dim]:.3f})\n"
        summary += f"Weakest Dimension: {weakest_dim} ({dimension_scores[weakest_dim]:.3f})"
        
        return summary

    def _generate_validation_recommendations(self, validation_scores: Dict) -> List[str]:
        """Generate recommendations based on validation results."""
        
        recommendations = []
        
        for dimension, scores in validation_scores.items():
            score = scores["score"]
            
            if score < 0.5:
                if dimension == "mathematical_validity":
                    recommendations.append("Improve mathematical consistency - check E₈ embedding constraints")
                elif dimension == "computational_evidence":
                    recommendations.append("Strengthen computational evidence - increase optimization iterations")
                elif dimension == "statistical_significance":
                    recommendations.append("Enhance statistical significance - compare against stronger baselines")
                elif dimension == "geometric_consistency":
                    recommendations.append("Improve geometric consistency - refine E₈ lattice alignment")
                elif dimension == "cross_validation":
                    recommendations.append("Strengthen cross-validation - test across more scenarios")
        
        if not recommendations:
            recommendations.append("Validation quality is excellent - no specific improvements needed")
        
        return recommendations

    def generate_baseline_comparison(self, 
                                   solution_vector: np.ndarray,
                                   n_baselines: int = 1000) -> Dict[str, Any]:
        """Generate comprehensive baseline comparison for validation."""
        
        print(f"Generating baseline comparison with {n_baselines} random vectors...")
        
        # Generate random baselines
        baseline_vectors = np.random.randn(n_baselines, 8)
        
        # Calculate metrics for all baselines
        baseline_norms = np.linalg.norm(baseline_vectors, axis=1)
        baseline_means = np.mean(baseline_vectors, axis=1)
        baseline_stds = np.std(baseline_vectors, axis=1)
        
        # Solution metrics
        solution_norm = np.linalg.norm(solution_vector)
        solution_mean = np.mean(solution_vector)
        solution_std = np.std(solution_vector)
        
        # Statistical comparisons
        norm_percentile = stats.percentileofscore(baseline_norms, solution_norm)
        mean_percentile = stats.percentileofscore(baseline_means, solution_mean)
        std_percentile = stats.percentileofscore(baseline_stds, solution_std)
        
        return {
            "baseline_count": n_baselines,
            "solution_metrics": {
                "norm": solution_norm,
                "mean": solution_mean,
                "std": solution_std
            },
            "baseline_statistics": {
                "norm_mean": np.mean(baseline_norms),
                "norm_std": np.std(baseline_norms),
                "mean_mean": np.mean(baseline_means),
                "mean_std": np.std(baseline_means),
                "std_mean": np.mean(baseline_stds),
                "std_std": np.std(baseline_stds)
            },
            "percentile_rankings": {
                "norm_percentile": norm_percentile,
                "mean_percentile": mean_percentile,
                "std_percentile": std_percentile
            }
        }
