"""
Rθ - Quantized Coxeter-plane rotation operator
"""

import numpy as np
from cqe.core.overlay import CQEOverlay
from cqe.operators.base import CQEOperator, OperatorType


class RotationOperator(CQEOperator):
    """
    Rθ: Quantized rotation in Coxeter plane.

    Rotates phases by quantized angle θ = k·(π/12) for k ∈ ℤ.
    Preserves geometric structure while exploring phase space.
    """

    operator_type = OperatorType.SYMMETRIC
    is_reversible = True

    def __init__(self, theta: float = np.pi/8):
        """
        Initialize rotation operator.

        Args:
            theta: Rotation angle (will be quantized to π/12 multiples)
        """
        # Quantize to π/12 increments
        self.theta = np.round(theta / (np.pi/12)) * (np.pi/12)

    def apply(self, overlay: CQEOverlay) -> CQEOverlay:
        """Apply rotation to active slots"""
        new_overlay = overlay.copy()
        active_indices = overlay.active_slots

        if len(active_indices) > 0:
            # Rotate phases
            new_overlay.phi[active_indices] += self.theta
            # Wrap to [-π, π]
            new_overlay.phi[active_indices] = np.mod(
                new_overlay.phi[active_indices] + np.pi, 
                2*np.pi
            ) - np.pi

        # Update provenance
        new_overlay.provenance.append(f"R_theta({self.theta:.4f})")

        return new_overlay

    def inverse(self, overlay: CQEOverlay) -> CQEOverlay:
        """Apply inverse rotation"""
        inverse_op = RotationOperator(-self.theta)
        return inverse_op.apply(overlay)

    def cost(self, overlay: CQEOverlay) -> float:
        """O(active_slots) complexity"""
        return float(len(overlay.active_slots))
