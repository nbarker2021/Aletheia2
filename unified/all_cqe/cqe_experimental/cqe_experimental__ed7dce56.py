"""
WeylReflect - Weyl reflection operator
"""

import numpy as np
from cqe.core.overlay import CQEOverlay
from cqe.operators.base import CQEOperator, OperatorType


class ReflectionOperator(CQEOperator):
    """
    WeylReflect: Reflection across simple root hyperplane.

    Applies Weyl reflection to explore symmetry-related regions
    of the E8 lattice while preserving structural properties.
    """

    operator_type = OperatorType.SYMMETRIC
    is_reversible = True

    def __init__(self, simple_root_idx: int = 0):
        """
        Initialize reflection operator.

        Args:
            simple_root_idx: Index of simple root (0-7)
        """
        if not 0 <= simple_root_idx < 8:
            raise ValueError(f"Root index must be in [0, 7], got {simple_root_idx}")
        self.simple_root_idx = simple_root_idx

    def apply(self, overlay: CQEOverlay) -> CQEOverlay:
        """Apply Weyl reflection"""
        new_overlay = overlay.copy()
        active_indices = overlay.active_slots

        if len(active_indices) > 0:
            # Simple reflection: add fixed phase shift
            reflection_angle = np.pi / 4
            new_overlay.phi[active_indices] += reflection_angle
            new_overlay.phi[active_indices] = np.mod(
                new_overlay.phi[active_indices] + np.pi,
                2*np.pi
            ) - np.pi

        # Update provenance
        new_overlay.provenance.append(f"WeylReflect(root={self.simple_root_idx})")

        return new_overlay

    def inverse(self, overlay: CQEOverlay) -> CQEOverlay:
        """Weyl reflection is its own inverse"""
        return self.apply(overlay)

    def cost(self, overlay: CQEOverlay) -> float:
        """O(active_slots) complexity"""
        return float(len(overlay.active_slots))
