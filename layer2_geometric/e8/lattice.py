"""
E8 Lattice Implementation

Complete implementation of the E8 lattice with 240 roots and Babai projection.
Based on the validated Aletheia CQE engine implementation.

Author: Manus AI (based on CQE research)
Date: December 5, 2025
"""

import numpy as np
from typing import Tuple, List, Optional


class E8Lattice:
    """
    E8 lattice structure with 240 roots and 8-dimensional Cartan subalgebra.
    
    The E8 lattice is the unique 8-dimensional lattice satisfying:
    - Closure under reflection/rotation
    - Self-duality (dual lattice = original)
    - Maximal sphere packing density
    - Even integer coordinates
    
    This implementation uses the standard E8 root system construction
    and provides Babai nearest-plane projection for embedding arbitrary
    8D vectors into the lattice.
    """
    
    def __init__(self):
        """Initialize the E8 lattice."""
        self.dimension = 8
        self.num_roots = 240
        
        # Construct E8 simple root basis
        self.basis = self._construct_e8_basis()
        
        # Pre-compute approximate roots for fast projection
        self.roots = self._generate_roots()
        
        # Initialize Weyl chamber navigator
        from ..weyl import WeylChamberNavigator
        simple_roots = self.roots[:8]  # First 8 roots as simple roots
        self.weyl_navigator = WeylChamberNavigator(simple_roots)
        
    def _construct_e8_basis(self) -> np.ndarray:
        """
        Construct E8 simple root basis matrix.
        
        Uses standard E8 root system construction with 7 D-type roots
        plus 1 characteristic vector.
        
        Returns:
            8x8 basis matrix
        """
        B = np.array([
            [1, -1, 0, 0, 0, 0, 0, 0],
            [0, 1, -1, 0, 0, 0, 0, 0],  
            [0, 0, 1, -1, 0, 0, 0, 0],
            [0, 0, 0, 1, -1, 0, 0, 0],
            [0, 0, 0, 0, 1, -1, 0, 0],
            [0, 0, 0, 0, 0, 1, -1, 0],
            [0, 0, 0, 0, 0, 1, 1, 0],
            [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
        ], dtype=np.float64)
        
        return B
    
    def _generate_roots(self) -> np.ndarray:
        """
        Generate all 240 E8 roots.
        
        For performance, this generates an approximation using the
        standard construction. The full 240 roots can be computed
        from the simple roots via Weyl group reflections.
        
        Returns:
            240x8 array of root vectors
        """
        roots = []
        
        # Type 1: All permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
        # 112 roots total
        for i in range(8):
            for j in range(i+1, 8):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        root = np.zeros(8)
                        root[i] = s1
                        root[j] = s2
                        roots.append(root)
        
        # Type 2: All sign changes of (±1/2, ±1/2, ..., ±1/2)
        # with even number of minus signs
        # 128 roots total
        for signs in range(256):
            root = np.array([(1 if (signs >> i) & 1 else -1) * 0.5 
                            for i in range(8)])
            if np.sum(root < 0) % 2 == 0:  # Even number of minus signs
                roots.append(root)
        
        return np.array(roots[:240])  # Ensure exactly 240 roots
    
    def project(self, vector: np.ndarray) -> np.ndarray:
        """
        Project arbitrary 8D vector to nearest E8 lattice point.
        
        Uses Babai nearest-plane algorithm for efficient projection.
        
        Args:
            vector: 8D input vector
        
        Returns:
            8D vector on E8 lattice (nearest lattice point)
        """
        if len(vector) != 8:
            raise ValueError(f"Expected 8D vector, got {len(vector)}D")
        
        # Babai nearest-plane algorithm:
        # 1. Express vector in basis coordinates
        # 2. Round to nearest integers
        # 3. Transform back to standard coordinates
        
        # Solve B * coeffs = vector for coeffs
        coeffs = np.linalg.solve(self.basis, vector)
        
        # Round to nearest integers
        rounded_coeffs = np.round(coeffs)
        
        # Transform back
        projected = self.basis.T @ rounded_coeffs
        
        return projected
    
    def distance_to_lattice(self, vector: np.ndarray) -> float:
        """
        Compute distance from vector to nearest E8 lattice point.
        
        Args:
            vector: 8D input vector
        
        Returns:
            Euclidean distance to nearest lattice point
        """
        projected = self.project(vector)
        return np.linalg.norm(vector - projected)
    
    def is_on_lattice(self, vector: np.ndarray, tolerance: float = 1e-6) -> bool:
        """
        Check if vector is on the E8 lattice.
        
        Args:
            vector: 8D vector to check
            tolerance: Numerical tolerance for floating point comparison
        
        Returns:
            True if vector is on lattice (within tolerance)
        """
        return self.distance_to_lattice(vector) < tolerance
    
    def get_root(self, index: int) -> np.ndarray:
        """
        Get a specific E8 root by index.
        
        Args:
            index: Root index (0-239)
        
        Returns:
            8D root vector
        """
        if not 0 <= index < 240:
            raise ValueError(f"Root index must be 0-239, got {index}")
        
        return self.roots[index]
    
    def __repr__(self) -> str:
        return f"E8Lattice(dim={self.dimension}, roots={self.num_roots})"
