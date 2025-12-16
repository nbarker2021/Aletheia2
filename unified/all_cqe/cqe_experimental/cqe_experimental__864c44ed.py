"""Overlay canonicalization using Weyl reflections"""

import numpy as np
import hashlib
from cqe.core.overlay import CQEOverlay
from cqe.core.lattice import E8Lattice


class Canonicalizer:
    """Canonicalizes overlays for consistent representation"""

    def __init__(self, lattice: E8Lattice):
        self.lattice = lattice

    def canonicalize(self, overlay: CQEOverlay) -> CQEOverlay:
        """
        Canonicalize overlay using gauge fixing and Weyl reflections.

        Args:
            overlay: Overlay to canonicalize

        Returns:
            Canonicalized overlay with hash_id
        """
        # Create copy
        canonical = overlay.copy()

        # Gauge fixing: align phase of maximum weight
        active_indices = canonical.active_slots
        if len(active_indices) > 0 and len(canonical.w) > 0:
            max_weight_idx = active_indices[np.argmax(canonical.w[active_indices])]
            if len(canonical.phi) > max_weight_idx:
                phase_shift = canonical.phi[max_weight_idx]
                canonical.phi[active_indices] -= phase_shift

        # Round for canonical representation
        canonical.phi = np.round(canonical.phi, 9)
        canonical.w = np.round(canonical.w, 8)

        # Compute content hash
        canonical.hash_id = canonical.compute_hash()

        return canonical
