"""
CQE Objective Function (Φ)

Multi-component objective function combining lattice embedding quality,
parity consistency, chamber stability, and domain-specific metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .e8_lattice import E8Lattice
from .parity_channels import ParityChannels

class CQEObjectiveFunction:
    """Multi-component objective function for CQE optimization."""

    def __init__(self, e8_lattice: E8Lattice, parity_channels: ParityChannels):
        self.e8_lattice = e8_lattice
        self.parity_channels = parity_channels

        # Component weights (can be tuned)
        self.weights = {
            "lattice_quality": 0.3,
            "parity_consistency": 0.25,
            "chamber_stability": 0.2,
            "geometric_separation": 0.15,
            "domain_coherence": 0.1
        }

    def evaluate(self, 
                vector: np.ndarray, 
                reference_channels: Dict[str, float],
                domain_context: Optional[Dict] = None) -> Dict[str, float]:
        """Evaluate the complete Φ objective function."""

        if len(vector) != 8:
            raise ValueError("Vector must be 8-dimensional")

        # Component evaluations
        lattice_score = self._evaluate_lattice_quality(vector)
        parity_score = self._evaluate_parity_consistency(vector, reference_channels)
        chamber_score = self._evaluate_chamber_stability(vector)
        separation_score = self._evaluate_geometric_separation(vector, domain_context)
        coherence_score = self._evaluate_domain_coherence(vector, domain_context)

        # Weighted combination
        phi_total = (
            self.weights["lattice_quality"] * lattice_score +
            self.weights["parity_consistency"] * parity_score +
            self.weights["chamber_stability"] * chamber_score +
            self.weights["geometric_separation"] * separation_score +
            self.weights["domain_coherence"] * coherence_score
        )

        return {
            "phi_total": phi_total,
            "lattice_quality": lattice_score,
            "parity_consistency": parity_score,
            "chamber_stability": chamber_score,
            "geometric_separation": separation_score,
            "domain_coherence": coherence_score
        }

    def _evaluate_lattice_quality(self, vector: np.ndarray) -> float:
        """Evaluate how well vector embeds in E₈ lattice structure."""
        quality_metrics = self.e8_lattice.root_embedding_quality(vector)

        # Distance to nearest root (smaller is better)
        root_distance = quality_metrics["nearest_root_distance"]
        root_score = max(0, 1.0 - root_distance / 2.0)

        # Chamber depth (distance from chamber walls)
        chamber_depth = quality_metrics["chamber_depth"]
        depth_score = min(1.0, chamber_depth / 0.5)

        # Symmetry of placement
        symmetry_score = max(0, 1.0 - quality_metrics["symmetry_score"])

        return 0.5 * root_score + 0.3 * depth_score + 0.2 * symmetry_score

    def _evaluate_parity_consistency(self, vector: np.ndarray, reference_channels: Dict[str, float]) -> float:
        """Evaluate parity channel consistency."""
        penalty = self.parity_channels.calculate_parity_penalty(vector, reference_channels)

        # Convert penalty to score (lower penalty = higher score)
        consistency_score = max(0, 1.0 - penalty / 2.0)

        return consistency_score

    def _evaluate_chamber_stability(self, vector: np.ndarray) -> float:
        """Evaluate stability within Weyl chamber."""
        chamber_sig, inner_prods = self.e8_lattice.determine_chamber(vector)

        # Stability based on distance from chamber boundaries
        min_distance_to_boundary = np.min(np.abs(inner_prods))
        stability_score = min(1.0, min_distance_to_boundary / 0.3)

        # Bonus for fundamental chamber
        fundamental_bonus = 0.1 if chamber_sig == "11111111" else 0.0

        return stability_score + fundamental_bonus

    def _evaluate_geometric_separation(self, vector: np.ndarray, domain_context: Optional[Dict]) -> float:
        """Evaluate geometric separation properties for complexity classes."""
        if not domain_context or "complexity_class" not in domain_context:
            return 0.5  # Neutral score if no context

        complexity_class = domain_context["complexity_class"]

        # Expected regions for different complexity classes
        if complexity_class == "P":
            # P problems should cluster near low-energy regions
            target_region = np.array([0.3, 0.1, 0.8, 0.4, 0.5, 0.3, 0.4, 0.2])
        elif complexity_class == "NP":
            # NP problems should occupy higher-energy, more dispersed regions
            target_region = np.array([0.6, 0.9, 0.5, 0.8, 0.7, 0.6, 0.8, 0.5])
        else:
            # Unknown complexity class
            return 0.5

        # Calculate distance to target region
        distance = np.linalg.norm(vector - target_region)
        separation_score = max(0, 1.0 - distance / 2.0)

        return separation_score

    def _evaluate_domain_coherence(self, vector: np.ndarray, domain_context: Optional[Dict]) -> float:
        """Evaluate coherence with domain-specific expectations."""
        if not domain_context:
            return 0.5

        domain_type = domain_context.get("domain_type", "unknown")

        if domain_type == "optimization":
            # Optimization problems should have structured patterns
            structure_score = 1.0 - np.std(vector)  # Prefer less chaotic vectors
            return max(0, min(1, structure_score))

        elif domain_type == "creative":
            # Creative problems should have more variability
            creativity_score = min(1.0, np.std(vector) * 2.0)  # Prefer more varied vectors
            return creativity_score

        elif domain_type == "computational":
            # Computational problems should balance structure and complexity
            balance = abs(np.mean(vector) - 0.5)  # Distance from center
            balance_score = max(0, 1.0 - balance * 2.0)
            return balance_score

        return 0.5  # Default neutral score

    def gradient(self, 
                vector: np.ndarray,
                reference_channels: Dict[str, float],
                domain_context: Optional[Dict] = None,
                epsilon: float = 1e-5) -> np.ndarray:
        """Calculate approximate gradient of objective function."""

        gradient = np.zeros(8)
        base_score = self.evaluate(vector, reference_channels, domain_context)["phi_total"]

        for i in range(8):
            # Forward difference
            perturbed = vector.copy()
            perturbed[i] += epsilon

            perturbed_score = self.evaluate(perturbed, reference_channels, domain_context)["phi_total"]
            gradient[i] = (perturbed_score - base_score) / epsilon

        return gradient

    def suggest_improvement_direction(self, 
                                    vector: np.ndarray,
                                    reference_channels: Dict[str, float],
                                    domain_context: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, str]]:
        """Suggest improvement direction and provide reasoning."""

        grad = self.gradient(vector, reference_channels, domain_context)
        scores = self.evaluate(vector, reference_channels, domain_context)

        # Normalize gradient
        if np.linalg.norm(grad) > 0:
            direction = grad / np.linalg.norm(grad)
        else:
            direction = np.zeros(8)

        # Provide reasoning based on component scores
        reasoning = {}
        for component, score in scores.items():
            if component != "phi_total":
                if score < 0.3:
                    reasoning[component] = "needs_significant_improvement"
                elif score < 0.6:
                    reasoning[component] = "needs_minor_improvement"
                else:
                    reasoning[component] = "acceptable"

        return direction, reasoning

    def set_weights(self, new_weights: Dict[str, float]):
        """Update component weights (must sum to 1.0)."""
        total = sum(new_weights.values())
        if abs(total - 1.0) > 1e-6:
            # Normalize weights
            new_weights = {k: v/total for k, v in new_weights.items()}

        self.weights.update(new_weights)
