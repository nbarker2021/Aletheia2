"""Φ objective function computation"""

import numpy as np
from cqe.core.overlay import CQEOverlay
from typing import Dict


class PhiComputer:
    """Computes Φ objective components for CQE overlays"""

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            'alpha': 1.0,    # geometry
            'beta': 5.0,     # parity  
            'gamma': 0.5,    # sparsity
            'delta': 0.1     # kissing
        }
        self.cartan_start = 240

    def compute_components(self, overlay: CQEOverlay) -> Dict[str, float]:
        """Compute all Φ components"""
        active_indices = overlay.active_slots

        if len(active_indices) == 0:
            return {
                'geom': 0.0,
                'parity': 0.0, 
                'sparsity': 0.0,
                'kissing': 0.0
            }

        # Φ_geom: geometric smoothness
        phi_geom = self._compute_geometric(overlay, active_indices)

        # Φ_parity: ECC syndrome
        phi_parity = self._compute_parity(overlay)

        # Φ_sparsity: L1 norm
        phi_sparsity = np.sum(np.abs(overlay.w[active_indices]))

        # Φ_kissing: deviation from E8 kissing number (240)
        phi_kissing = abs(len(active_indices) / 240.0 - 1.0)

        return {
            'geom': phi_geom,
            'parity': phi_parity,
            'sparsity': phi_sparsity,
            'kissing': phi_kissing
        }

    def compute_total(self, phi_components: Dict[str, float]) -> float:
        """Compute weighted total Φ"""
        return (
            self.weights['alpha'] * phi_components['geom'] +
            self.weights['beta'] * phi_components['parity'] +
            self.weights['gamma'] * phi_components['sparsity'] +
            self.weights['delta'] * phi_components['kissing']
        )

    def _compute_geometric(self, overlay: CQEOverlay, active_indices: np.ndarray) -> float:
        """Compute geometric smoothness component"""
        if len(active_indices) < 3:
            return 0.0

        phases = overlay.phi[active_indices]
        weights = overlay.w[active_indices]

        # Angular acceleration (second derivative approximation)
        angular_accel = np.var(np.diff(phases, 2)) if len(phases) > 2 else 0.0

        # Radial jitter
        radial_jitter = np.var(weights) if len(weights) > 1 else 0.0

        return angular_accel + radial_jitter

    def _compute_parity(self, overlay: CQEOverlay) -> float:
        """Compute parity/ECC component"""
        cartan_bits = overlay.present[self.cartan_start:self.cartan_start+8].astype(int)
        return float(np.sum(cartan_bits % 2))
