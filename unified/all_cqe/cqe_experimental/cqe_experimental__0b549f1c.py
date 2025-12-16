"""
Core MORSR protocol implementation
"""

import numpy as np
from typing import List, Optional, Tuple
from cqe.core.overlay import CQEOverlay
from cqe.core.phi import PhiComputer
from cqe.core.canonicalization import Canonicalizer
from cqe.operators.base import CQEOperator
from cqe.operators.rotation import RotationOperator
from cqe.operators.reflection import ReflectionOperator
from cqe.operators.midpoint import MidpointOperator
from cqe.operators.parity import ECCParityOperator
from cqe.morsr.acceptance import AcceptanceChecker
from cqe.morsr.handshake import HandshakeRecord, HandshakeLogger


class MORSRProtocol:
    """
    MORSR: Middle-Out Ripple Shape Reader

    Implements monotone optimization protocol with:
    - Pulse sweep through operator sequence
    - Monotone acceptance (ΔΦ ≤ 0)
    - Provenance tracking via handshakes
    - Convergence detection
    """

    def __init__(self, phi_computer: PhiComputer, canonicalizer: Canonicalizer):
        self.phi_computer = phi_computer
        self.canonicalizer = canonicalizer
        self.acceptance_checker = AcceptanceChecker()
        self.handshake_logger = HandshakeLogger()

        # Default operator sequence
        self.operators: List[CQEOperator] = [
            RotationOperator(),
            ReflectionOperator(),
            MidpointOperator(),
            ECCParityOperator(),
        ]

    def pulse_sweep(
        self,
        seed_overlay: CQEOverlay,
        max_iterations: int = 10,
        convergence_threshold: float = 1e-6
    ) -> CQEOverlay:
        """
        Execute MORSR pulse sweep.

        Args:
            seed_overlay: Initial overlay
            max_iterations: Maximum pulse iterations
            convergence_threshold: Convergence criterion for ΔΦ

        Returns:
            Optimized overlay after pulse sweep
        """
        current = self.canonicalizer.canonicalize(seed_overlay)
        iteration = 0

        # Compute initial Φ
        phi_components = self.phi_computer.compute_components(current)
        phi_current = self.phi_computer.compute_total(phi_components)

        while iteration < max_iterations:
            iteration += 1
            any_accepted = False

            # Try each operator
            for operator in self.operators:
                # Apply operator
                candidate = operator.apply(current)
                candidate = self.canonicalizer.canonicalize(candidate)

                # Compute Φ after transformation
                phi_comp_candidate = self.phi_computer.compute_components(candidate)
                phi_candidate = self.phi_computer.compute_total(phi_comp_candidate)

                # Check acceptance
                accepted, reason = self.acceptance_checker.check(
                    phi_current, phi_candidate
                )

                # Log handshake
                handshake = HandshakeRecord(
                    operator_name=operator.__class__.__name__,
                    phi_before=phi_current,
                    phi_after=phi_candidate,
                    delta_phi=phi_candidate - phi_current,
                    accepted=accepted,
                    reason=reason,
                    overlay_hash=candidate.hash_id
                )
                self.handshake_logger.log(handshake)

                # Accept if monotone improvement
                if accepted:
                    current = candidate
                    phi_current = phi_candidate
                    any_accepted = True

            # Check convergence
            if not any_accepted:
                break

        return current

    def get_handshake_log(self) -> List[HandshakeRecord]:
        """Retrieve complete handshake log"""
        return self.handshake_logger.get_log()

    def clear_log(self):
        """Clear handshake log"""
        self.handshake_logger.clear()
