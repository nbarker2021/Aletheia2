"""
Domain Adapter for CQE System

Converts problem instances from various domains (P/NP, optimization, scenes)
into 8-dimensional feature vectors suitable for E₈ lattice embedding.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import hashlib

class DomainAdapter:
    """Adapts various problem domains into CQE-compatible feature vectors."""

    def __init__(self):
        self.feature_dim = 8  # E₈ embedding dimension

    def embed_p_problem(self, instance_size: int, complexity_hint: int = 1) -> np.ndarray:
        """Embed a P-class problem instance into 8D space."""
        # P problems typically have polynomial-time characteristics
        features = np.zeros(8)

        # Dimension 0: Problem size (log scale)
        features[0] = np.log10(max(1, instance_size)) / 10.0

        # Dimension 1: Complexity class indicator (0 for P)
        features[1] = 0.1 * complexity_hint

        # Dimension 2: Deterministic factor (high for P)
        features[2] = 0.8 + 0.1 * np.sin(instance_size * 0.1)

        # Dimension 3: Resource scaling (polynomial)
        features[3] = min(0.9, np.power(instance_size, 0.3) / 100.0)

        # Dimensions 4-7: Problem-specific features
        features[4] = 0.5 + 0.2 * np.cos(instance_size * 0.05)
        features[5] = 0.3 + 0.1 * np.sin(instance_size * 0.03)
        features[6] = 0.4 + 0.15 * np.cos(instance_size * 0.07)
        features[7] = 0.2 + 0.1 * np.sin(instance_size * 0.02)

        return features

    def embed_np_problem(self, instance_size: int, nondeterminism: float = 0.8) -> np.ndarray:
        """Embed an NP-class problem instance into 8D space."""
        # NP problems have exponential-time worst-case characteristics
        features = np.zeros(8)

        # Dimension 0: Problem size (log scale)
        features[0] = np.log10(max(1, instance_size)) / 10.0

        # Dimension 1: Complexity class indicator (1 for NP)
        features[1] = 0.9 + 0.1 * nondeterminism

        # Dimension 2: Nondeterministic factor (high for NP)
        features[2] = nondeterminism

        # Dimension 3: Resource scaling (exponential tendency)
        features[3] = min(1.0, np.power(instance_size, 0.5) / 50.0)

        # Dimensions 4-7: NP-specific features (more erratic)
        features[4] = 0.7 + 0.3 * np.sin(instance_size * 0.1 * nondeterminism)
        features[5] = 0.6 + 0.2 * np.cos(instance_size * 0.08 * nondeterminism)
        features[6] = 0.8 + 0.2 * np.sin(instance_size * 0.12 * nondeterminism)
        features[7] = 0.5 + 0.3 * np.cos(instance_size * 0.15 * nondeterminism)

        return features

    def embed_optimization_problem(self, 
                                  variables: int, 
                                  constraints: int,
                                  objective_type: str = "linear") -> np.ndarray:
        """Embed an optimization problem into 8D space."""
        features = np.zeros(8)

        # Dimension 0-1: Problem structure
        features[0] = np.log10(max(1, variables)) / 10.0
        features[1] = np.log10(max(1, constraints)) / 10.0

        # Dimension 2: Objective type encoding
        obj_encoding = {"linear": 0.2, "quadratic": 0.5, "nonlinear": 0.8}
        features[2] = obj_encoding.get(objective_type, 0.5)

        # Dimension 3: Constraint density
        density = constraints / max(1, variables)
        features[3] = min(1.0, density / 10.0)

        # Dimensions 4-7: Additional optimization features
        features[4] = 0.5 + 0.2 * np.sin(variables * 0.1)
        features[5] = 0.4 + 0.3 * np.cos(constraints * 0.05)
        features[6] = 0.6 + 0.1 * np.sin((variables + constraints) * 0.03)
        features[7] = 0.3 + 0.2 * np.cos(density)

        return features

    def embed_scene_problem(self, 
                           scene_complexity: int,
                           narrative_depth: int,
                           character_count: int) -> np.ndarray:
        """Embed a creative scene generation problem into 8D space."""
        features = np.zeros(8)

        # Dimension 0-2: Scene structure
        features[0] = min(1.0, scene_complexity / 100.0)
        features[1] = min(1.0, narrative_depth / 50.0)
        features[2] = min(1.0, character_count / 20.0)

        # Dimension 3: Creative tension
        tension = (scene_complexity * narrative_depth) / (character_count + 1)
        features[3] = min(1.0, tension / 1000.0)

        # Dimensions 4-7: Creative features
        features[4] = 0.4 + 0.3 * np.sin(scene_complexity * 0.1)
        features[5] = 0.5 + 0.2 * np.cos(narrative_depth * 0.2)
        features[6] = 0.3 + 0.4 * np.sin(character_count * 0.3)
        features[7] = 0.6 + 0.1 * np.cos(tension * 0.01)

        return features

    def hash_to_features(self, data: str) -> np.ndarray:
        """Convert arbitrary string data to 8D features via hashing."""
        # Use SHA-256 hash for deterministic feature generation
        hash_bytes = hashlib.sha256(data.encode()).digest()

        # Convert first 8 bytes to features in [0, 1]
        features = np.array([b / 255.0 for b in hash_bytes[:8]])

        return features

    def validate_features(self, features: np.ndarray) -> bool:
        """Validate that features are in valid range for E₈ embedding."""
        if len(features) != 8:
            return False

        # Features should be roughly in [0, 1] range
        if np.any(features < -2.0) or np.any(features > 2.0):
            return False

        return True
