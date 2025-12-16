"""
SingleInsert - Controlled slot insertion operator
"""

import numpy as np
from cqe.core.overlay import CQEOverlay
from cqe.operators.base import CQEOperator, OperatorType
from typing import Optional


class SingleInsertOperator(CQEOperator):
    """
    SingleInsert: Add single new active slot.

    Controlled expansion operator that activates one new slot
    with specified weight and phase.
    """

    operator_type = OperatorType.EXPANSION
    is_reversible = False

    def __init__(self, target_idx: Optional[int] = None, weight: float = 1.0):
        """
        Initialize insertion operator.

        Args:
            target_idx: Index to insert (None = auto-select)
            weight: Weight for new slot
        """
        self.target_idx = target_idx
        self.weight = weight

    def apply(self, overlay: CQEOverlay) -> CQEOverlay:
        """Apply single insertion"""
        new_overlay = overlay.copy()

        # Determine insertion index
        if self.target_idx is None:
            # Auto-select: first inactive Cartan lane
            cartan_start = 240
            for i in range(8):
                idx = cartan_start + i
                if not overlay.present[idx]:
                    insert_idx = idx
                    break
            else:
                # All Cartan active, use first inactive root
                inactive_roots = np.where(~overlay.present[:240])[0]
                if len(inactive_roots) > 0:
                    insert_idx = inactive_roots[0]
                else:
                    return new_overlay  # No space to insert
        else:
            insert_idx = self.target_idx

        # Insert if not already active
        if not overlay.present[insert_idx]:
            new_overlay.present[insert_idx] = True
            new_overlay.w[insert_idx] = self.weight
            new_overlay.phi[insert_idx] = 0.0

        # Update provenance
        new_overlay.provenance.append(f"SingleInsert(idx={insert_idx})")

        return new_overlay

    def cost(self, overlay: CQEOverlay) -> float:
        """O(1) complexity"""
        return 1.0
