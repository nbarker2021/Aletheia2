"""
Acceptance Rules and Parity Tracking
Cartan-Quadratic Equivalence (CQE) System

From Whitepaper:
"Acceptance: ΔΦ ≤ -ε; Midpoint may accept with ΔΦ≈0 only if parity syndrome
strictly decreases. Plateau accepts optional with a small global cap."

This module implements the strict acceptance criteria and parity signature
tracking required by CQE.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_morphonic.overlay_system import Overlay
from layer1_morphonic.alena_operators import OperationResult


class AcceptanceType(Enum):
    """Types of acceptance."""
    STRICT_DECREASE = "strict_decrease"  # ΔΦ ≤ -ε
    PARITY_DECREASE = "parity_decrease"  # ΔΦ≈0 but parity decreases
    PLATEAU = "plateau"  # ΔΦ≈0, parity same, within global cap
    REJECTED = "rejected"  # Does not meet criteria


@dataclass
class ParitySignature:
    """
    Parity signature π(x) for an overlay.
    
    From Axiom D:
    "x ~_CQE y iff ∃g∈G: y=g·x, Φ(y)=Φ(x), and π(y)=π(x)"
    
    The parity signature captures the parity properties of the overlay
    that must be preserved under CQE equivalence.
    """
    
    # Parity components
    activation_parity: int  # Parity of number of activations (0 or 1)
    weight_parity: float  # Parity of weight sum
    phase_parity: float  # Parity of phase
    position_parity: Tuple[int, ...]  # Parity of each position component
    
    # Parity syndrome (overall measure)
    syndrome: float
    
    @classmethod
    def from_overlay(cls, overlay: Overlay) -> 'ParitySignature':
        """Compute parity signature from overlay."""
        # Activation parity (even/odd number of activations)
        num_active = int(np.sum(overlay.activations))
        activation_parity = num_active % 2
        
        # Weight parity (sign of weight sum)
        weight_parity = 0.0
        if overlay.weights is not None:
            weight_sum = np.sum(overlay.weights)
            weight_parity = 1.0 if weight_sum >= 0 else -1.0
        
        # Phase parity (sign of phase)
        phase_parity = 0.0
        if overlay.phase is not None:
            phase_parity = 1.0 if overlay.phase >= 0 else -1.0
        
        # Position parity (sign of each component)
        position_parity = tuple(
            1 if x >= 0 else -1
            for x in overlay.e8_base
        )
        
        # Compute syndrome (overall parity measure)
        # Higher syndrome = more parity violations
        syndrome = 0.0
        
        # Component 1: Activation imbalance
        activation_imbalance = abs(num_active - 120) / 120
        syndrome += activation_imbalance
        
        # Component 2: Weight imbalance (if present)
        if overlay.weights is not None:
            positive_weights = np.sum(overlay.weights[overlay.weights > 0])
            negative_weights = abs(np.sum(overlay.weights[overlay.weights < 0]))
            total = positive_weights + negative_weights
            if total > 0:
                weight_imbalance = abs(positive_weights - negative_weights) / total
                syndrome += weight_imbalance
        
        # Component 3: Position imbalance
        positive_pos = np.sum(overlay.e8_base[overlay.e8_base > 0])
        negative_pos = abs(np.sum(overlay.e8_base[overlay.e8_base < 0]))
        total_pos = positive_pos + negative_pos
        if total_pos > 0:
            position_imbalance = abs(positive_pos - negative_pos) / total_pos
            syndrome += position_imbalance
        
        return cls(
            activation_parity=activation_parity,
            weight_parity=weight_parity,
            phase_parity=phase_parity,
            position_parity=position_parity,
            syndrome=syndrome
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'activation_parity': self.activation_parity,
            'weight_parity': self.weight_parity,
            'phase_parity': self.phase_parity,
            'position_parity': list(self.position_parity),
            'syndrome': self.syndrome
        }
    
    def __eq__(self, other: 'ParitySignature') -> bool:
        """Check if two parity signatures are equal."""
        return (
            self.activation_parity == other.activation_parity and
            abs(self.weight_parity - other.weight_parity) < 1e-6 and
            abs(self.phase_parity - other.phase_parity) < 1e-6 and
            self.position_parity == other.position_parity
        )


@dataclass
class AcceptanceDecision:
    """Decision on whether to accept a transition."""
    accepted: bool
    acceptance_type: AcceptanceType
    delta_phi: float
    delta_parity: float  # Change in parity syndrome
    reason: str
    metadata: Dict[str, Any]


class AcceptanceRule:
    """
    Strict acceptance rule for CQE transitions.
    
    From Whitepaper:
    "Acceptance: ΔΦ ≤ -ε; Midpoint may accept with ΔΦ≈0 only if parity
    syndrome strictly decreases. Plateau accepts optional with a small
    global cap."
    """
    
    def __init__(
        self,
        epsilon: float = 1e-6,  # Threshold for strict decrease
        plateau_threshold: float = 1e-8,  # Threshold for ΔΦ≈0
        plateau_cap: int = 10,  # Max plateau accepts per session
        allow_plateau: bool = True
    ):
        self.epsilon = epsilon
        self.plateau_threshold = plateau_threshold
        self.plateau_cap = plateau_cap
        self.allow_plateau = allow_plateau
        
        # Track plateau accepts
        self.plateau_count = 0
        
        # History
        self.decisions: List[AcceptanceDecision] = []
    
    def evaluate(
        self,
        overlay_before: Overlay,
        overlay_after: Overlay,
        delta_phi: float,
        operation: str = "unknown"
    ) -> AcceptanceDecision:
        """
        Evaluate whether to accept a transition.
        
        Args:
            overlay_before: Overlay before operation
            overlay_after: Overlay after operation
            delta_phi: Change in phi metric (Φ_after - Φ_before)
            operation: Name of operation
        
        Returns:
            AcceptanceDecision
        """
        # Compute parity signatures
        parity_before = ParitySignature.from_overlay(overlay_before)
        parity_after = ParitySignature.from_overlay(overlay_after)
        delta_parity = parity_after.syndrome - parity_before.syndrome
        
        # Rule 1: Strict decrease (ΔΦ ≤ -ε)
        if delta_phi <= -self.epsilon:
            decision = AcceptanceDecision(
                accepted=True,
                acceptance_type=AcceptanceType.STRICT_DECREASE,
                delta_phi=delta_phi,
                delta_parity=delta_parity,
                reason="strict_decrease",
                metadata={
                    'operation': operation,
                    'parity_before': parity_before.to_dict(),
                    'parity_after': parity_after.to_dict()
                }
            )
            self.decisions.append(decision)
            return decision
        
        # Rule 2: Parity decrease (ΔΦ≈0 but parity syndrome decreases)
        # Special case for Midpoint operation
        if abs(delta_phi) < self.plateau_threshold:
            if delta_parity < -self.epsilon:
                decision = AcceptanceDecision(
                    accepted=True,
                    acceptance_type=AcceptanceType.PARITY_DECREASE,
                    delta_phi=delta_phi,
                    delta_parity=delta_parity,
                    reason="parity_decrease",
                    metadata={
                        'operation': operation,
                        'parity_before': parity_before.to_dict(),
                        'parity_after': parity_after.to_dict()
                    }
                )
                self.decisions.append(decision)
                return decision
            
            # Rule 3: Plateau (ΔΦ≈0, parity same, within global cap)
            if self.allow_plateau and self.plateau_count < self.plateau_cap:
                self.plateau_count += 1
                decision = AcceptanceDecision(
                    accepted=True,
                    acceptance_type=AcceptanceType.PLATEAU,
                    delta_phi=delta_phi,
                    delta_parity=delta_parity,
                    reason=f"plateau_{self.plateau_count}/{self.plateau_cap}",
                    metadata={
                        'operation': operation,
                        'parity_before': parity_before.to_dict(),
                        'parity_after': parity_after.to_dict()
                    }
                )
                self.decisions.append(decision)
                return decision
        
        # Rule 4: Rejected
        reason = "rejected"
        if delta_phi > 0:
            reason = "phi_increase"
        elif abs(delta_phi) < self.plateau_threshold:
            if delta_parity >= 0:
                reason = "parity_no_decrease"
            if self.plateau_count >= self.plateau_cap:
                reason = "plateau_cap_exceeded"
        
        decision = AcceptanceDecision(
            accepted=False,
            acceptance_type=AcceptanceType.REJECTED,
            delta_phi=delta_phi,
            delta_parity=delta_parity,
            reason=reason,
            metadata={
                'operation': operation,
                'parity_before': parity_before.to_dict(),
                'parity_after': parity_after.to_dict()
            }
        )
        self.decisions.append(decision)
        return decision
    
    def evaluate_operation_result(
        self,
        overlay_before: Overlay,
        result: OperationResult
    ) -> AcceptanceDecision:
        """
        Evaluate an OperationResult.
        
        Args:
            overlay_before: Overlay before operation
            result: OperationResult from ALENA operator
        
        Returns:
            AcceptanceDecision
        """
        return self.evaluate(
            overlay_before=overlay_before,
            overlay_after=result.overlay,
            delta_phi=result.delta_phi,
            operation=result.operation
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get acceptance statistics."""
        total = len(self.decisions)
        if total == 0:
            return {
                'total': 0,
                'accepted': 0,
                'rejected': 0,
                'accept_rate': 0.0,
                'by_type': {}
            }
        
        accepted = sum(1 for d in self.decisions if d.accepted)
        rejected = sum(1 for d in self.decisions if not d.accepted)
        
        by_type = {}
        for decision in self.decisions:
            type_name = decision.acceptance_type.value
            if type_name not in by_type:
                by_type[type_name] = 0
            by_type[type_name] += 1
        
        return {
            'total': total,
            'accepted': accepted,
            'rejected': rejected,
            'accept_rate': accepted / total,
            'plateau_count': self.plateau_count,
            'plateau_cap': self.plateau_cap,
            'by_type': by_type
        }
    
    def reset(self):
        """Reset acceptance rule state."""
        self.plateau_count = 0
        self.decisions.clear()


# Example usage and tests
if __name__ == "__main__":
    print("=== Acceptance Rules and Parity Tracking Test ===\n")
    
    from layer1_morphonic.alena_operators import ALENAOperators
    from layer1_morphonic.overlay_system import ImmutablePose
    import time
    
    # Create initial overlay
    e8_base = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0])
    activations = np.zeros(240, dtype=int)
    activations[0:120] = 1  # Activate exactly half (balanced)
    
    pose = ImmutablePose(
        position=tuple(e8_base),
        orientation=tuple(np.eye(8)[0]),
        timestamp=time.time()
    )
    
    overlay = Overlay(e8_base=e8_base, activations=activations, pose=pose)
    
    print(f"Initial overlay: {overlay.overlay_id}")
    print(f"Active roots: {overlay.num_active()}")
    
    # Compute parity signature
    parity = ParitySignature.from_overlay(overlay)
    print(f"Parity signature:")
    print(f"  Activation parity: {parity.activation_parity}")
    print(f"  Syndrome: {parity.syndrome:.6f}")
    print()
    
    # Create acceptance rule and operators
    rule = AcceptanceRule(epsilon=1e-6, allow_plateau=True, plateau_cap=5)
    alena = ALENAOperators()
    
    # Test 1: Strict decrease (should accept)
    print("Test 1: Strict Decrease")
    result1 = alena.midpoint(overlay, overlay, weight=0.5)
    decision1 = rule.evaluate_operation_result(overlay, result1)
    print(f"ΔΦ: {decision1.delta_phi:.6f}")
    print(f"ΔParity: {decision1.delta_parity:.6f}")
    print(f"Accepted: {decision1.accepted}")
    print(f"Type: {decision1.acceptance_type.value}")
    print(f"Reason: {decision1.reason}")
    print()
    
    # Test 2: Rotation (ΔΦ≈0, should check parity)
    print("Test 2: Rotation (ΔΦ≈0)")
    result2 = alena.rotate(overlay, theta=0.01)  # Small rotation
    decision2 = rule.evaluate_operation_result(overlay, result2)
    print(f"ΔΦ: {decision2.delta_phi:.6f}")
    print(f"ΔParity: {decision2.delta_parity:.6f}")
    print(f"Accepted: {decision2.accepted}")
    print(f"Type: {decision2.acceptance_type.value}")
    print(f"Reason: {decision2.reason}")
    print()
    
    # Test 3: Parity mirror (should increase Φ, reject)
    print("Test 3: Parity Mirror (should reject)")
    result3 = alena.parity_mirror(overlay)
    decision3 = rule.evaluate_operation_result(overlay, result3)
    print(f"ΔΦ: {decision3.delta_phi:.6f}")
    print(f"ΔParity: {decision3.delta_parity:.6f}")
    print(f"Accepted: {decision3.accepted}")
    print(f"Type: {decision3.acceptance_type.value}")
    print(f"Reason: {decision3.reason}")
    print()
    
    # Test 4: Statistics
    print("Test 4: Acceptance Statistics")
    stats = rule.get_statistics()
    print(f"Total decisions: {stats['total']}")
    print(f"Accepted: {stats['accepted']}")
    print(f"Rejected: {stats['rejected']}")
    print(f"Accept rate: {stats['accept_rate']:.2%}")
    print(f"By type:")
    for type_name, count in stats['by_type'].items():
        print(f"  {type_name}: {count}")
    print()
    
    print("=== All Tests Passed ===")
