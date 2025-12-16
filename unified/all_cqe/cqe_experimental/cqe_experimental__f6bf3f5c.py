"""Babai nearest-plane embedding for E8 lattice"""

import numpy as np
from cqe.core.lattice import E8Lattice
from cqe.core.overlay import CQEOverlay
from typing import Tuple


class BabaiEmbedder:
    """Embeds feature vectors into E8 lattice using Babai algorithm"""

    def __init__(self, lattice: E8Lattice):
        self.lattice = lattice
        self.cartan_start_idx = 240

    def embed(self, features: np.ndarray, domain: str) -> CQEOverlay:
        """
        Embed 8-dimensional features into E8 lattice.

        Args:
            features: 8-dimensional feature vector
            domain: Domain type (text, code, etc.)

        Returns:
            CQEOverlay with embedded representation
        """
        # Project to lattice
        y_snapped, error = self.lattice.project_to_lattice(features)

        # Create overlay
        present = np.zeros(248, dtype=bool)
        w = np.zeros(248)
        phi = np.zeros(248)

        # Activate root based on features
        root_idx = int(abs(hash(features.tobytes())) % 240)
        present[root_idx] = True
        w[root_idx] = np.linalg.norm(y_snapped)
        phi[root_idx] = 0.0

        # Activate Cartan lanes based on feature magnitudes
        for i, feat_val in enumerate(features):
            if abs(feat_val) > 1e-6:
                cartan_idx = self.cartan_start_idx + i
                present[cartan_idx] = True
                w[cartan_idx] = abs(feat_val)
                phi[cartan_idx] = np.arctan2(0, feat_val)

        # Create overlay
        overlay = CQEOverlay(
            present=present,
            w=w,
            phi=phi,
            pose={
                'domain_type': domain,
                'embedding_error': error,
                'root_index': root_idx,
                'features': features.tolist()
            }
        )

        return overlay
