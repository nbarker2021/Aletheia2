"""
Parity operators: ParityMirror and ECC-Parity
"""

import numpy as np
from cqe.core.overlay import CQEOverlay
from cqe.operators.base import CQEOperator, OperatorType


class ParityMirrorOperator(CQEOperator):
    """
    ParityMirror: Mirror Cartan lanes across center.

    Creates symmetry by reflecting low lanes to high lanes,
    establishing parity relationships.
    """

    operator_type = OperatorType.ASYMMETRIC
    is_reversible = False

    def __init__(self, cartan_start: int = 240):
        self.cartan_start = cartan_start

    def apply(self, overlay: CQEOverlay) -> CQEOverlay:
        """Apply parity mirroring"""
        new_overlay = overlay.copy()

        # Mirror low Cartan lanes (0-3) to high lanes (4-7)
        for lane_offset in range(4):
            src_idx = self.cartan_start + lane_offset
            dst_idx = self.cartan_start + (7 - lane_offset)

            if overlay.present[src_idx]:
                new_overlay.present[dst_idx] = True
                new_overlay.w[dst_idx] = overlay.w[src_idx]
                new_overlay.phi[dst_idx] = -overlay.phi[src_idx]  # Negative for mirror

        # Update provenance
        new_overlay.provenance.append("ParityMirror")

        return new_overlay

    def cost(self, overlay: CQEOverlay) -> float:
        """O(1) - fixed 4 lanes"""
        return 4.0


class ECCParityOperator(CQEOperator):
    """
    ECC-Parity: Error-correcting code parity check.

    Ensures even parity across Cartan lanes by flipping
    if necessary to maintain ECC invariant.
    """

    operator_type = OperatorType.PARITY
    is_reversible = True

    def __init__(self, cartan_start: int = 240):
        self.cartan_start = cartan_start

    def apply(self, overlay: CQEOverlay) -> CQEOverlay:
        """Apply ECC parity correction"""
        new_overlay = overlay.copy()

        # Count active Cartan bits
        cartan_bits = overlay.present[self.cartan_start:self.cartan_start+8].astype(int)
        parity = np.sum(cartan_bits) % 2

        # If odd parity, flip first active bit
        if parity == 1:
            active_cartan = np.where(cartan_bits)[0]
            if len(active_cartan) > 0:
                flip_idx = self.cartan_start + active_cartan[0]
                new_overlay.present[flip_idx] = False

        # Update provenance
        new_overlay.provenance.append("ECC_Parity")

        return new_overlay

    def inverse(self, overlay: CQEOverlay) -> CQEOverlay:
        """ECC parity is its own inverse"""
        return self.apply(overlay)

    def cost(self, overlay: CQEOverlay) -> float:
        """O(1) - fixed 8 lanes"""
        return 8.0
