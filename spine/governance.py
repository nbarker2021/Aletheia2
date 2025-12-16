"""
Governance Engine - Constraint Enforcement

Enforces all system constraints:
- ΔΦ ≤ 0 (monotonic improvement)
- Parity validity
- Digital root governance
- Policy enforcement
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Tuple
from enum import Enum
import numpy as np

from spine.kernel import CQEAtom
from spine.speedlight import get_speedlight


class ConstraintResult(Enum):
    """Result of a constraint check."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


@dataclass
class CheckResult:
    """Result of a governance check."""
    allowed: bool
    reason: str
    constraint: str
    severity: ConstraintResult


class GovernancePolicy:
    """Base class for governance policies."""
    
    name: str = "base"
    
    def check(self, atom: CQEAtom, context: Dict[str, Any] = None) -> CheckResult:
        """Check if the atom passes this policy."""
        raise NotImplementedError


class DeltaPhiPolicy(GovernancePolicy):
    """Enforce ΔΦ ≤ 0 (monotonic improvement)."""
    
    name = "delta_phi"
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self._last_phi: Optional[float] = None
    
    def check(self, atom: CQEAtom, context: Dict[str, Any] = None) -> CheckResult:
        current_phi = atom.phi()
        
        if self._last_phi is None:
            self._last_phi = current_phi
            return CheckResult(
                allowed=True,
                reason="First observation, no delta",
                constraint=self.name,
                severity=ConstraintResult.PASS
            )
        
        delta = current_phi - self._last_phi
        self._last_phi = current_phi
        
        if delta <= self.tolerance:
            return CheckResult(
                allowed=True,
                reason=f"ΔΦ = {delta:.6f} ≤ 0",
                constraint=self.name,
                severity=ConstraintResult.PASS
            )
        else:
            return CheckResult(
                allowed=False,
                reason=f"ΔΦ = {delta:.6f} > 0 (monotonic improvement violated)",
                constraint=self.name,
                severity=ConstraintResult.FAIL
            )


class ParityPolicy(GovernancePolicy):
    """Enforce parity validity."""
    
    name = "parity"
    
    def check(self, atom: CQEAtom, context: Dict[str, Any] = None) -> CheckResult:
        if atom.parity_ok:
            return CheckResult(
                allowed=True,
                reason="Parity valid",
                constraint=self.name,
                severity=ConstraintResult.PASS
            )
        else:
            return CheckResult(
                allowed=False,
                reason="Parity invalid (odd sum)",
                constraint=self.name,
                severity=ConstraintResult.FAIL
            )


class DigitalRootPolicy(GovernancePolicy):
    """Enforce digital root governance (DR 0/9 preferred)."""
    
    name = "digital_root"
    preferred_roots = {0, 9}
    
    def check(self, atom: CQEAtom, context: Dict[str, Any] = None) -> CheckResult:
        dr = atom.digital_root()
        
        if dr in self.preferred_roots:
            return CheckResult(
                allowed=True,
                reason=f"DR = {dr} (preferred)",
                constraint=self.name,
                severity=ConstraintResult.PASS
            )
        else:
            # Not a hard failure, just a warning
            return CheckResult(
                allowed=True,
                reason=f"DR = {dr} (not preferred, but allowed)",
                constraint=self.name,
                severity=ConstraintResult.WARN
            )


class GovernanceEngine:
    """
    Governance Engine - Enforces all system constraints.
    
    All atoms must pass governance checks before being accepted.
    """
    
    def __init__(self):
        self.policies: Dict[str, GovernancePolicy] = {}
        self.speedlight = get_speedlight()
        
        # Register default policies
        self.register_policy(DeltaPhiPolicy())
        self.register_policy(ParityPolicy())
        self.register_policy(DigitalRootPolicy())
    
    def register_policy(self, policy: GovernancePolicy):
        """Register a governance policy."""
        self.policies[policy.name] = policy
    
    def check(self, atom: CQEAtom, operation: str = "") -> Tuple[bool, str]:
        """
        Check if an atom passes all governance constraints.
        
        Returns (allowed, reason).
        """
        context = {"operation": operation}
        failures = []
        warnings = []
        
        for name, policy in self.policies.items():
            result = policy.check(atom, context)
            if not result.allowed:
                failures.append(result.reason)
            elif result.severity == ConstraintResult.WARN:
                warnings.append(result.reason)
        
        if failures:
            return False, "; ".join(failures)
        elif warnings:
            return True, f"Passed with warnings: {'; '.join(warnings)}"
        else:
            return True, "All constraints passed"
    
    def enforce_delta_phi(self, old_phi: float, new_phi: float) -> bool:
        """Check if ΔΦ ≤ 0."""
        return new_phi <= old_phi
    
    def validate_parity(self, atom: CQEAtom) -> bool:
        """Check if parity is valid."""
        return atom.parity_ok
    
    def apply_policy(self, atom: CQEAtom, policy_name: str) -> CQEAtom:
        """
        Apply a specific policy to an atom.
        
        This may modify the atom to make it compliant.
        """
        if policy_name == "parity" and not atom.parity_ok:
            # Repair parity by flipping one bit
            atom.parity[0] = 1 - atom.parity[0]
        
        return atom
    
    def get_status(self) -> Dict[str, Any]:
        """Get governance engine status."""
        return {
            "policies": list(self.policies.keys()),
            "active": True
        }
