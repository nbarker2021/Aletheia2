"""
E8 Lattice operations and root system
"""

import numpy as np
from scipy.linalg import qr
from typing import Tuple, Optional


class E8Lattice:
    """
    E8 lattice structure with 240 roots and 8-dimensional Cartan subalgebra.

    Provides:
    - E8 simple root basis construction
    - QR factorization for Babai algorithm
    - Root system operations
    - Weyl chamber projections
    """

    def __init__(self):
        self.dimension = 8
        self.num_roots = 240
        self.num_cartan = 8
        self.total_slots = 248

        # Construct E8 simple root basis
        self.B = self._construct_e8_basis()
        self.B_inv = np.linalg.inv(self.B)

        # QR factorization for Babai
        self.Q, self.R = qr(self.B)

        # Basis condition number
        self.condition_number = np.linalg.cond(self.B)

    def _construct_e8_basis(self) -> np.ndarray:
        """
        Construct E8 simple root basis matrix.
        Uses standard E8 root system construction.
        """
        B = np.array([
            [1, -1, 0, 0, 0, 0, 0, 0],
            [0, 1, -1, 0, 0, 0, 0, 0],  
            [0, 0, 1, -1, 0, 0, 0, 0],
            [0, 0, 0, 1, -1, 0, 0, 0],
            [0, 0, 0, 0, 1, -1, 0, 0],
            [0, 0, 0, 0, 0, 1, -1, 0],
            [0, 0, 0, 0, 0, 0, 1, -1],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # E8 characteristic
        ], dtype=float)
        return B

    def project_to_lattice(self, vector: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Project vector to E8 lattice using Babai nearest-plane algorithm.

        Args:
            vector: 8-dimensional input vector

        Returns:
            (lattice_point, error): Nearest lattice point and projection error
        """
        # Babai nearest-plane
        y = self.B @ vector
        z = np.linalg.solve(self.R, self.Q.T @ y)
        z_rounded = np.round(z)
        y_snapped = self.Q @ (self.R @ z_rounded)

        # Compute error
        offset = y - y_snapped
        error = np.linalg.norm(offset)

        return y_snapped, error

    def get_simple_root(self, index: int) -> np.ndarray:
        """Get simple root by index (0-7)"""
        if not 0 <= index < 8:
            raise ValueError(f"Root index must be in [0, 7], got {index}")
        return self.B[index]

    def weyl_reflect(self, vector: np.ndarray, root_index: int) -> np.ndarray:
        """
        Apply Weyl reflection across simple root hyperplane.

        Args:
            vector: Point to reflect
            root_index: Index of simple root (0-7)

        Returns:
            Reflected vector
        """
        alpha = self.get_simple_root(root_index)
        alpha_norm_sq = np.dot(alpha, alpha)

        # Reflection formula: v - 2(v·α/α·α)α
        reflection = vector - 2 * np.dot(vector, alpha) / alpha_norm_sq * alpha
        return reflection

    def is_in_weyl_chamber(self, vector: np.ndarray, tolerance: float = 1e-6) -> bool:
        """
        Check if vector is in dominant Weyl chamber.

        Args:
            vector: Point to check
            tolerance: Numerical tolerance

        Returns:
            True if in dominant chamber
        """
        # Check if all simple root pairings are non-negative
        for i in range(8):
            alpha = self.get_simple_root(i)
            if np.dot(vector, alpha) < -tolerance:
                return False
        return True

    def info(self) -> dict:
        """Return lattice information"""
        return {
            'dimension': self.dimension,
            'num_roots': self.num_roots,
            'num_cartan': self.num_cartan,
            'total_slots': self.total_slots,
            'condition_number': self.condition_number
        }
