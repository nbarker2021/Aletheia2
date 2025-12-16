"""
Midpoint - Palindromic expansion operator
"""

import numpy as np
from cqe.core.overlay import CQEOverlay
from cqe.operators.base import CQEOperator, OperatorType


class MidpointOperator(CQEOperator):
    """
    Midpoint: Palindromic parity reduction.

    Creates symmetry by averaging phases in Cartan lanes,
    reducing angular variance and improving geometric smoothness.
    """

    operator_type = OperatorType.ASYMMETRIC
    is_reversible = False

    def __init__(self, cartan_start: int = 240):
        self.cartan_start = cartan_start

    def apply(self, overlay: CQEOverlay) -> CQEOverlay:
        """Apply palindromic midpoint operation"""
        new_overlay = overlay.copy()

        # Get active Cartan lanes
        active_indices = overlay.active_slots
        cartan_indices = active_indices[active_indices >= self.cartan_start]

        if len(cartan_indices) >= 2:
            # Create palindrome by averaging symmetric pairs
            mid_idx = len(cartan_indices) // 2

            for i in range(mid_idx):
                j = len(cartan_indices) - 1 - i
                if i != j:
                    avg_phase = (
                        new_overlay.phi[cartan_indices[i]] +
                        new_overlay.phi[cartan_indices[j]]
                    ) / 2.0
                    new_overlay.phi[cartan_indices[i]] = avg_phase
                    new_overlay.phi[cartan_indices[j]] = avg_phase

        # Update provenance
        new_overlay.provenance.append("Midpoint(palindrome)")

        return new_overlay

    def cost(self, overlay: CQEOverlay) -> float:
        """O(cartan_active) complexity"""
        return float(overlay.cartan_active)
