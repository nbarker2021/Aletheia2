"""
E₈ Lattice Operations

Handles E₈ lattice embedding operations including nearest root lookup,
Weyl chamber determination, and canonical projection.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class E8Lattice:
    """E₈ lattice operations for CQE system."""

    def __init__(self, embedding_path: str = "embeddings/e8_248_embedding.json"):
        """Initialize with cached E₈ embedding data."""
        self.embedding_path = embedding_path
        self.roots = None
        self.cartan_matrix = None
        self.simple_roots = None
        self._load_embedding()
        self._setup_chambers()

    def _load_embedding(self):
        """Load the cached E₈ embedding."""
        if not Path(self.embedding_path).exists():
            raise FileNotFoundError(f"E₈ embedding not found at {self.embedding_path}")

        with open(self.embedding_path, 'r') as f:
            data = json.load(f)

        self.roots = np.array(data["roots_8d"])  # 240×8
        self.cartan_matrix = np.array(data["cartan_8x8"])  # 8×8

        print(f"Loaded E₈ embedding: {len(self.roots)} roots, {self.cartan_matrix.shape} Cartan matrix")

    def _setup_chambers(self):
        """Setup simple roots for Weyl chamber calculations."""
        # Simple roots are the first 8 roots (by convention)
        # For E₈, these form the basis of the root system
        self.simple_roots = self.roots[:8]  # 8×8

        # Verify we have a valid simple root system
        if self.simple_roots.shape != (8, 8):
            raise ValueError("Invalid simple root system shape")

    def nearest_root(self, vector: np.ndarray) -> Tuple[int, np.ndarray, float]:
        """Find the nearest E₈ root to the given vector."""
        if len(vector) != 8:
            raise ValueError("Vector must be 8-dimensional")

        # Calculate distances to all roots
        distances = np.linalg.norm(self.roots - vector, axis=1)

        # Find minimum distance
        nearest_idx = np.argmin(distances)
        nearest_root = self.roots[nearest_idx]
        min_distance = distances[nearest_idx]

        return nearest_idx, nearest_root, min_distance

    def determine_chamber(self, vector: np.ndarray) -> Tuple[str, np.ndarray]:
        """Determine which Weyl chamber contains the vector."""
        if len(vector) != 8:
            raise ValueError("Vector must be 8-dimensional")

        # Calculate inner products with simple roots
        inner_products = np.dot(self.simple_roots, vector)

        # Determine chamber by sign pattern
        signs = np.sign(inner_products)

        # Fundamental chamber: all inner products ≥ 0
        is_fundamental = np.all(signs >= 0)

        # Create chamber signature
        chamber_sig = ''.join(['1' if s >= 0 else '0' for s in signs])

        return chamber_sig, inner_products

    def project_to_chamber(self, vector: np.ndarray, target_chamber: str = "11111111") -> np.ndarray:
        """Project vector to specified Weyl chamber (default: fundamental)."""
        if len(vector) != 8:
            raise ValueError("Vector must be 8-dimensional")

        current_chamber, inner_prods = self.determine_chamber(vector)

        if current_chamber == target_chamber:
            return vector.copy()

        # Simple projection: reflect across hyperplanes to reach target chamber
        projected = vector.copy()

        for i, (current_bit, target_bit) in enumerate(zip(current_chamber, target_chamber)):
            if current_bit != target_bit:
                # Reflect across the i-th simple root hyperplane
                simple_root = self.simple_roots[i]
                # Reflection formula: v' = v - 2<v,α>/<α,α> α
                inner_prod = np.dot(projected, simple_root)
                root_norm_sq = np.dot(simple_root, simple_root)

                if root_norm_sq > 1e-10:  # Avoid division by zero
                    projected = projected - 2 * inner_prod / root_norm_sq * simple_root

        return projected

    def chamber_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate chamber-aware distance between vectors."""
        # Project both vectors to fundamental chamber
        proj1 = self.project_to_chamber(vec1)
        proj2 = self.project_to_chamber(vec2)

        # Calculate Euclidean distance
        return np.linalg.norm(proj1 - proj2)

    def root_embedding_quality(self, vector: np.ndarray) -> Dict[str, float]:
        """Assess the quality of a vector's embedding in E₈ space."""
        nearest_idx, nearest_root, min_dist = self.nearest_root(vector)
        chamber_sig, inner_prods = self.determine_chamber(vector)

        # Calculate various quality metrics
        metrics = {
            "nearest_root_distance": float(min_dist),
            "nearest_root_index": int(nearest_idx),
            "chamber_signature": chamber_sig,
            "fundamental_chamber": chamber_sig == "11111111",
            "vector_norm": float(np.linalg.norm(vector)),
            "chamber_depth": float(np.min(np.abs(inner_prods))),  # Distance to chamber walls
            "symmetry_score": float(np.std(inner_prods))  # How symmetric the placement is
        }

        return metrics

    def generate_chamber_samples(self, chamber_sig: str, count: int = 10) -> np.ndarray:
        """Generate random samples from specified Weyl chamber."""
        samples = []

        for _ in range(count * 3):  # Generate extra to account for rejections
            # Generate random vector
            vec = np.random.randn(8)

            # Project to desired chamber
            projected = self.project_to_chamber(vec, chamber_sig)

            # Verify it's in the right chamber
            actual_chamber, _ = self.determine_chamber(projected)

            if actual_chamber == chamber_sig:
                samples.append(projected)
                if len(samples) >= count:
                    break

        return np.array(samples[:count])
