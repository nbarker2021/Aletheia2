"""
Conservation - Information Conservation Laws

Based on "WP-021: Information Conservation Laws in Physical Systems"

Information conservation is a fundamental principle: information is neither
created nor destroyed, only transformed. This module enforces conservation
laws to ensure lossless encoding/decoding.

Key principles:
- Information = distinguishability of physical states
- Logically reversible operations = zero energy dissipation
- Geometric invariants preserve information
- dI_total/dt = 0 for closed systems
"""

import numpy as np
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import math


def shannon_entropy(probabilities: List[float]) -> float:
    """
    Compute Shannon entropy: H = -Σ p_i * log2(p_i)
    
    Measures the average uncertainty or information content.
    """
    H = 0.0
    for p in probabilities:
        if p > 0:
            H -= p * math.log2(p)
    return H


def information_content(n_states: int) -> float:
    """
    Compute information content in bits: log2(N)
    
    N distinguishable states = log2(N) bits of information.
    """
    if n_states <= 0:
        return 0.0
    return math.log2(n_states)


@dataclass
class InformationState:
    """Tracks information state for conservation verification."""
    accessible: float  # Accessible information (bits)
    inaccessible: float  # Inaccessible information (dissipated, entangled)
    total: float = field(init=False)  # Total conserved information
    
    def __post_init__(self):
        self.total = self.accessible + self.inaccessible
    
    def transform(self, delta_accessible: float) -> 'InformationState':
        """
        Transform information state.
        
        Conservation law: dI_accessible = -dI_inaccessible
        """
        new_accessible = self.accessible + delta_accessible
        new_inaccessible = self.inaccessible - delta_accessible
        return InformationState(
            accessible=max(0, new_accessible),
            inaccessible=max(0, new_inaccessible)
        )
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "accessible": self.accessible,
            "inaccessible": self.inaccessible,
            "total": self.total
        }


@dataclass
class ConservationCheck:
    """Result of a conservation check."""
    conserved: bool
    initial_total: float
    final_total: float
    delta: float
    tolerance: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "conserved": self.conserved,
            "initial_total": self.initial_total,
            "final_total": self.final_total,
            "delta": self.delta,
            "tolerance": self.tolerance
        }


class ConservationLaw:
    """
    Enforces information conservation laws.
    
    Core principle: For a closed system, dI_total/dt = 0
    """
    
    def __init__(self, tolerance: float = 1e-9):
        self.tolerance = tolerance
        self.history: List[InformationState] = []
    
    def check_conservation(
        self,
        initial: InformationState,
        final: InformationState
    ) -> ConservationCheck:
        """
        Check if information is conserved between two states.
        
        Returns True if |I_total_final - I_total_initial| < tolerance
        """
        delta = abs(final.total - initial.total)
        conserved = delta < self.tolerance
        
        return ConservationCheck(
            conserved=conserved,
            initial_total=initial.total,
            final_total=final.total,
            delta=delta,
            tolerance=self.tolerance
        )
    
    def verify_reversibility(
        self,
        forward_op: callable,
        reverse_op: callable,
        input_data: Any
    ) -> Tuple[bool, Dict]:
        """
        Verify that an operation is logically reversible.
        
        Reversible operations preserve information perfectly.
        """
        # Apply forward operation
        forward_result = forward_op(input_data)
        
        # Apply reverse operation
        reverse_result = reverse_op(forward_result)
        
        # Check if we recovered the original
        if isinstance(input_data, np.ndarray):
            is_reversible = np.allclose(input_data, reverse_result, rtol=self.tolerance)
        elif isinstance(input_data, (list, tuple)):
            is_reversible = all(
                abs(a - b) < self.tolerance 
                for a, b in zip(input_data, reverse_result)
            )
        else:
            is_reversible = input_data == reverse_result
        
        return is_reversible, {
            "input": str(input_data)[:100],
            "forward_result": str(forward_result)[:100],
            "reverse_result": str(reverse_result)[:100],
            "reversible": is_reversible
        }
    
    def compute_encoding_efficiency(
        self,
        initial_accessible: float,
        final_accessible: float
    ) -> float:
        """
        Compute efficiency of lossless encoding/decoding.
        
        Efficiency = I_accessible_final / I_accessible_initial
        For truly lossless: efficiency = 1.0
        """
        if initial_accessible <= 0:
            return 0.0
        return final_accessible / initial_accessible
    
    def record_state(self, state: InformationState):
        """Record state for history tracking."""
        self.history.append(state)
    
    def verify_history_conservation(self) -> List[ConservationCheck]:
        """Verify conservation across all recorded states."""
        if len(self.history) < 2:
            return []
        
        checks = []
        for i in range(1, len(self.history)):
            check = self.check_conservation(self.history[i-1], self.history[i])
            checks.append(check)
        
        return checks


class GeometricInvariant:
    """
    Geometric invariants that preserve information.
    
    Based on WP-008: Geometric Governance - invariants represent
    conserved quantities of information.
    """
    
    def __init__(self, name: str, compute_fn: callable):
        self.name = name
        self.compute_fn = compute_fn
    
    def compute(self, data: Any) -> float:
        """Compute the invariant value."""
        return self.compute_fn(data)
    
    def is_preserved(
        self,
        before: Any,
        after: Any,
        tolerance: float = 1e-9
    ) -> Tuple[bool, Dict]:
        """Check if invariant is preserved under transformation."""
        val_before = self.compute(before)
        val_after = self.compute(after)
        delta = abs(val_after - val_before)
        preserved = delta < tolerance
        
        return preserved, {
            "invariant": self.name,
            "before": val_before,
            "after": val_after,
            "delta": delta,
            "preserved": preserved
        }


class QuadraticInvariant(GeometricInvariant):
    """
    Quadratic invariant from the Law of Quadratic Invariance.
    
    Q(v) = Σ v_i² (L2 norm squared)
    """
    
    def __init__(self):
        super().__init__(
            name="quadratic_invariant",
            compute_fn=self._compute_quadratic
        )
    
    def _compute_quadratic(self, v: Any) -> float:
        """Compute quadratic invariant (L2 norm squared)."""
        if isinstance(v, np.ndarray):
            return float(np.sum(v ** 2))
        elif isinstance(v, (list, tuple)):
            return sum(x ** 2 for x in v)
        else:
            return float(v) ** 2


class ParityInvariant(GeometricInvariant):
    """
    Parity invariant for CQE states.
    
    Parity = Σ cartan_i mod 2
    """
    
    def __init__(self):
        super().__init__(
            name="parity_invariant",
            compute_fn=self._compute_parity
        )
    
    def _compute_parity(self, state: Any) -> int:
        """Compute parity invariant."""
        if hasattr(state, 'cartan'):
            return sum(state.cartan) % 2
        elif isinstance(state, dict) and 'cartan' in state:
            return sum(state['cartan']) % 2
        elif isinstance(state, (list, tuple)):
            return sum(state) % 2
        else:
            return 0


class DigitalRootInvariant(GeometricInvariant):
    """
    Digital root invariant.
    
    DR(n) = repeated digit sum until single digit
    """
    
    def __init__(self):
        super().__init__(
            name="digital_root_invariant",
            compute_fn=self._compute_dr
        )
    
    def _compute_dr(self, n: Any) -> int:
        """Compute digital root."""
        if isinstance(n, (list, tuple, np.ndarray)):
            n = int(sum(abs(x) for x in n))
        else:
            n = abs(int(n))
        
        while n > 9:
            n = sum(int(d) for d in str(n))
        return n


class ConservationEngine:
    """
    Engine for enforcing conservation laws across the system.
    
    Ensures all transformations preserve information.
    """
    
    def __init__(self):
        self.law = ConservationLaw()
        self.invariants: Dict[str, GeometricInvariant] = {}
        self._register_default_invariants()
    
    def _register_default_invariants(self):
        """Register default geometric invariants."""
        self.invariants["quadratic"] = QuadraticInvariant()
        self.invariants["parity"] = ParityInvariant()
        self.invariants["digital_root"] = DigitalRootInvariant()
    
    def register_invariant(self, invariant: GeometricInvariant):
        """Register a custom invariant."""
        self.invariants[invariant.name] = invariant
    
    def verify_transformation(
        self,
        before: Any,
        after: Any,
        required_invariants: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Verify that a transformation preserves required invariants.
        
        Returns detailed report of which invariants were preserved.
        """
        if required_invariants is None:
            required_invariants = list(self.invariants.keys())
        
        results = {}
        all_preserved = True
        
        for name in required_invariants:
            if name in self.invariants:
                preserved, details = self.invariants[name].is_preserved(before, after)
                results[name] = details
                if not preserved:
                    all_preserved = False
        
        return {
            "all_preserved": all_preserved,
            "invariants": results
        }
    
    def create_reversible_transform(
        self,
        forward_fn: callable,
        reverse_fn: callable,
        name: str = "transform"
    ) -> callable:
        """
        Create a verified reversible transformation.
        
        Wraps the forward function with conservation verification.
        """
        def verified_transform(data: Any) -> Tuple[Any, Dict]:
            # Apply forward
            result = forward_fn(data)
            
            # Verify reversibility
            is_reversible, details = self.law.verify_reversibility(
                forward_fn, reverse_fn, data
            )
            
            return result, {
                "transform": name,
                "reversible": is_reversible,
                "details": details
            }
        
        return verified_transform
    
    def compute_total_information(self, data: Any) -> float:
        """
        Compute total information content of data.
        
        Uses hash-based estimation for complex data.
        """
        if isinstance(data, (int, float)):
            # Simple numeric: estimate based on precision
            return math.log2(max(1, abs(data) * 1000))
        elif isinstance(data, (list, tuple, np.ndarray)):
            # Array: sum of element information
            return sum(self.compute_total_information(x) for x in data)
        elif isinstance(data, str):
            # String: character entropy
            if not data:
                return 0.0
            char_counts = {}
            for c in data:
                char_counts[c] = char_counts.get(c, 0) + 1
            probs = [count / len(data) for count in char_counts.values()]
            return shannon_entropy(probs) * len(data)
        elif isinstance(data, dict):
            # Dict: sum of key and value information
            return sum(
                self.compute_total_information(k) + self.compute_total_information(v)
                for k, v in data.items()
            )
        else:
            # Default: hash-based estimate
            h = hashlib.sha256(str(data).encode()).hexdigest()
            return len(h) * 4  # 4 bits per hex char


# Global instance
_global_engine: Optional[ConservationEngine] = None


def get_conservation_engine() -> ConservationEngine:
    """Get the global conservation engine."""
    global _global_engine
    if _global_engine is None:
        _global_engine = ConservationEngine()
    return _global_engine
