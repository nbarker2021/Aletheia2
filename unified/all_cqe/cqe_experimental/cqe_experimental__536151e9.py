"""
MORSR acceptance logic - monotone ΔΦ ≤ 0 rule
"""

from typing import Tuple


class AcceptanceChecker:
    """
    Checks monotone acceptance criterion for MORSR.

    Accepts transformations where:
    - ΔΦ < 0 (strict decrease)
    - ΔΦ ≈ 0 (plateau, within tolerance)

    Rejects:
    - ΔΦ > 0 (increase)
    """

    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize acceptance checker.

        Args:
            tolerance: Numerical tolerance for plateau detection
        """
        self.tolerance = tolerance

    def check(self, phi_before: float, phi_after: float) -> Tuple[bool, str]:
        """
        Check if transformation is acceptable.

        Args:
            phi_before: Φ before transformation
            phi_after: Φ after transformation

        Returns:
            (accepted, reason) tuple
        """
        delta_phi = phi_after - phi_before

        if delta_phi < -self.tolerance:
            return True, "strict_decrease"
        elif abs(delta_phi) <= self.tolerance:
            return True, "plateau"
        else:
            return False, "increase_rejected"

    def is_converged(self, delta_phi: float) -> bool:
        """Check if optimization has converged"""
        return abs(delta_phi) < self.tolerance
