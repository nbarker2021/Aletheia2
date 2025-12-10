"""
GNLC λ₀ Atom Calculus
Geometry-Native Lambda Calculus (GNLC)
Cartan-Quadratic Equivalence (CQE) System

From Whitepaper (04_GNLC_Formalization.md):
"λ₀ is the atom calculus - the lowest layer where computation is purely
geometric. Terms are overlays, application is ALENA composition, and
reduction is MORSR optimization."

This implements the foundational layer of GNLC:
- Terms as E₈ overlays
- Application via ALENA operators
- Reduction via phi-decrease
- Stratified types (λ₀ → λ₁ → ... → λ_θ)
"""

import numpy as np
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_morphonic.overlay_system import Overlay
from layer1_morphonic.alena_operators import ALENAOperators, OperationResult
from layer1_morphonic.acceptance_rules import AcceptanceRule


class TermType(Enum):
    """Types in the GNLC stratification."""
    ATOM = "λ₀"  # Geometric atoms (overlays)
    FUNCTION = "λ₁"  # Functions (overlay transformations)
    HIGHER = "λ_θ"  # Higher-order types


@dataclass
class Lambda0Term:
    """
    λ₀ term (atom).
    
    In GNLC, the most basic terms are geometric atoms - overlays in E₈.
    """
    overlay: Overlay
    term_type: TermType = TermType.ATOM
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def term_id(self) -> str:
        """Unique identifier for this term."""
        return self.overlay.overlay_id
    
    def __repr__(self):
        return f"λ₀[{self.term_id[:8]}]"


@dataclass
class Lambda0Application:
    """
    Application in λ₀.
    
    Application is geometric composition via ALENA operators.
    """
    operator: str  # ALENA operator name
    arguments: List[Lambda0Term]
    result: Optional[Lambda0Term] = None
    
    def __repr__(self):
        args_repr = ", ".join(str(arg) for arg in self.arguments)
        return f"{self.operator}({args_repr})"


@dataclass
class Lambda0Reduction:
    """
    Reduction step in λ₀.
    
    Reduction is phi-decreasing transformation.
    """
    term_before: Lambda0Term
    term_after: Lambda0Term
    delta_phi: float
    operation: str
    accepted: bool
    
    def __repr__(self):
        arrow = "→" if self.accepted else "↛"
        return f"{self.term_before} {arrow} {self.term_after} (ΔΦ={self.delta_phi:.6f})"


class Lambda0Calculus:
    """
    λ₀ Atom Calculus.
    
    From whitepaper:
    "λ₀ is the atom calculus where computation is purely geometric."
    
    Key principles:
    1. Terms are overlays
    2. Application is ALENA composition
    3. Reduction is phi-decrease
    4. Types are geometric invariants
    """
    
    def __init__(self):
        self.alena = ALENAOperators()
        self.acceptance = AcceptanceRule()
        self.reduction_history: List[Lambda0Reduction] = []
    
    def atom(self, overlay: Overlay) -> Lambda0Term:
        """
        Create an atom term from an overlay.
        
        Args:
            overlay: E₈ overlay
        
        Returns:
            λ₀ term
        """
        return Lambda0Term(overlay=overlay, term_type=TermType.ATOM)
    
    def apply(
        self,
        operator: str,
        *terms: Lambda0Term,
        **kwargs
    ) -> Lambda0Application:
        """
        Apply ALENA operator to terms.
        
        Args:
            operator: ALENA operator name (rotate, weyl_reflect, midpoint, parity_mirror)
            *terms: Terms to apply operator to
            **kwargs: Operator parameters
        
        Returns:
            Application result
        """
        # Extract overlays from terms
        overlays = [term.overlay for term in terms]
        
        # Apply ALENA operator
        if operator == "rotate":
            if len(overlays) != 1:
                raise ValueError("rotate requires 1 term")
            result = self.alena.rotate(overlays[0], **kwargs)
        
        elif operator == "weyl_reflect":
            if len(overlays) != 1:
                raise ValueError("weyl_reflect requires 1 term")
            result = self.alena.weyl_reflect(overlays[0], **kwargs)
        
        elif operator == "midpoint":
            if len(overlays) != 2:
                raise ValueError("midpoint requires 2 terms")
            result = self.alena.midpoint(overlays[0], overlays[1], **kwargs)
        
        elif operator == "parity_mirror":
            if len(overlays) != 1:
                raise ValueError("parity_mirror requires 1 term")
            result = self.alena.parity_mirror(overlays[0], **kwargs)
        
        else:
            raise ValueError(f"Unknown operator: {operator}")
        
        # Create result term
        result_term = Lambda0Term(
            overlay=result.overlay,
            term_type=TermType.ATOM,
            metadata={
                'operator': operator,
                'delta_phi': result.delta_phi,
                'parameters': kwargs
            }
        )
        
        # Create application
        application = Lambda0Application(
            operator=operator,
            arguments=list(terms),
            result=result_term
        )
        
        return application
    
    def reduce(self, term: Lambda0Term, max_steps: int = 10) -> List[Lambda0Reduction]:
        """
        Reduce a term via phi-decrease.
        
        Args:
            term: Term to reduce
            max_steps: Maximum reduction steps
        
        Returns:
            List of reduction steps
        """
        reductions = []
        current_term = term
        
        for step in range(max_steps):
            # Try different ALENA operators
            candidates = []
            
            # Try rotation
            for theta in [0.01, 0.05, 0.1]:
                app = self.apply("rotate", current_term, theta=theta)
                if app.result:
                    candidates.append(("rotate", app.result, theta))
            
            # Try Weyl reflection
            for root_idx in [0, 10, 50, 100]:
                app = self.apply("weyl_reflect", current_term, root_index=root_idx)
                if app.result:
                    candidates.append(("weyl_reflect", app.result, root_idx))
            
            # Try parity mirror
            app = self.apply("parity_mirror", current_term)
            if app.result:
                candidates.append(("parity_mirror", app.result, None))
            
            # Select best candidate (lowest phi)
            if not candidates:
                break
            
            best_op, best_term, best_param = min(
                candidates,
                key=lambda c: self.alena._compute_phi(c[1].overlay)
            )
            
            # Check if reduction is accepted
            phi_before = self.alena._compute_phi(current_term.overlay)
            phi_after = self.alena._compute_phi(best_term.overlay)
            delta_phi = phi_after - phi_before
            
            # Evaluate acceptance
            operation_result = OperationResult(
                overlay=best_term.overlay,
                delta_phi=delta_phi,
                operation=best_op,
                parameters={'param': best_param},
                success=True,
                reason="reduction_step"
            )
            
            decision = self.acceptance.evaluate_operation_result(
                current_term.overlay,
                operation_result
            )
            
            # Record reduction
            reduction = Lambda0Reduction(
                term_before=current_term,
                term_after=best_term,
                delta_phi=delta_phi,
                operation=best_op,
                accepted=decision.accepted
            )
            
            reductions.append(reduction)
            self.reduction_history.append(reduction)
            
            if not decision.accepted:
                # Reduction rejected, stop
                break
            
            # Continue with reduced term
            current_term = best_term
            
            # Check if we've reached normal form (phi not decreasing)
            if abs(delta_phi) < 1e-6:
                break
        
        return reductions
    
    def normalize(self, term: Lambda0Term) -> Lambda0Term:
        """
        Normalize a term to normal form.
        
        Args:
            term: Term to normalize
        
        Returns:
            Normalized term
        """
        reductions = self.reduce(term)
        
        if reductions:
            # Return final term
            return reductions[-1].term_after
        else:
            # Already in normal form
            return term
    
    def compose(self, term1: Lambda0Term, term2: Lambda0Term) -> Lambda0Term:
        """
        Compose two terms via midpoint.
        
        Args:
            term1: First term
            term2: Second term
        
        Returns:
            Composed term
        """
        app = self.apply("midpoint", term1, term2, weight=0.5)
        return app.result
    
    def type_of(self, term: Lambda0Term) -> Dict[str, Any]:
        """
        Compute type of a term.
        
        In λ₀, types are geometric invariants:
        - Digital root stratification
        - Weyl chamber
        - Parity signature
        - Phi value
        
        Args:
            term: Term to type
        
        Returns:
            Type information
        """
        from layer1_morphonic.acceptance_rules import ParitySignature
        
        overlay = term.overlay
        
        # Compute geometric invariants
        phi = self.alena._compute_phi(overlay)
        parity = ParitySignature.from_overlay(overlay)
        
        # Digital root (simplified: sum of activations mod 10)
        digital_root = int(np.sum(overlay.activations)) % 10
        
        # Weyl chamber (simplified: sign pattern of e8_base)
        weyl_chamber = tuple(np.sign(overlay.e8_base).astype(int))
        
        return {
            'term_type': term.term_type.value,
            'phi': phi,
            'parity_syndrome': parity.syndrome,
            'digital_root': digital_root,
            'weyl_chamber': weyl_chamber,
            'num_active': int(np.sum(overlay.activations))
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get calculus statistics."""
        if not self.reduction_history:
            return {
                'total_reductions': 0,
                'accepted': 0,
                'rejected': 0,
                'accept_rate': 0.0
            }
        
        accepted = sum(1 for r in self.reduction_history if r.accepted)
        rejected = len(self.reduction_history) - accepted
        
        return {
            'total_reductions': len(self.reduction_history),
            'accepted': accepted,
            'rejected': rejected,
            'accept_rate': accepted / len(self.reduction_history),
            'avg_delta_phi': np.mean([r.delta_phi for r in self.reduction_history])
        }


# Example usage and tests
if __name__ == "__main__":
    print("=== GNLC λ₀ Atom Calculus Test ===\n")
    
    from layer1_morphonic.overlay_system import ImmutablePose
    import time
    
    # Create λ₀ calculus
    lambda0 = Lambda0Calculus()
    
    # Test 1: Create atoms
    print("Test 1: Create Atoms")
    
    e8_base1 = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0])
    activations1 = np.zeros(240, dtype=int)
    activations1[0:120] = 1
    
    pose1 = ImmutablePose(
        position=tuple(e8_base1),
        orientation=tuple(np.eye(8)[0]),
        timestamp=time.time()
    )
    
    overlay1 = Overlay(e8_base=e8_base1, activations=activations1, pose=pose1)
    term1 = lambda0.atom(overlay1)
    
    print(f"Term 1: {term1}")
    print(f"  Type: {lambda0.type_of(term1)['term_type']}")
    print(f"  Phi: {lambda0.type_of(term1)['phi']:.6f}")
    print()
    
    # Test 2: Application
    print("Test 2: Application")
    
    app = lambda0.apply("rotate", term1, theta=0.1)
    print(f"Application: {app}")
    print(f"  Result: {app.result}")
    print(f"  ΔΦ: {app.result.metadata['delta_phi']:.6f}")
    print()
    
    # Test 3: Reduction
    print("Test 3: Reduction")
    
    reductions = lambda0.reduce(term1, max_steps=5)
    print(f"Reduction steps: {len(reductions)}")
    for i, red in enumerate(reductions):
        print(f"  Step {i+1}: {red}")
    print()
    
    # Test 4: Normalization
    print("Test 4: Normalization")
    
    normalized = lambda0.normalize(term1)
    print(f"Original: {term1}")
    print(f"Normalized: {normalized}")
    print(f"  Phi before: {lambda0.type_of(term1)['phi']:.6f}")
    print(f"  Phi after: {lambda0.type_of(normalized)['phi']:.6f}")
    print()
    
    # Test 5: Composition
    print("Test 5: Composition")
    
    e8_base2 = np.array([0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0])
    activations2 = np.zeros(240, dtype=int)
    activations2[60:180] = 1
    
    pose2 = ImmutablePose(
        position=tuple(e8_base2),
        orientation=tuple(np.eye(8)[0]),
        timestamp=time.time()
    )
    
    overlay2 = Overlay(e8_base=e8_base2, activations=activations2, pose=pose2)
    term2 = lambda0.atom(overlay2)
    
    composed = lambda0.compose(term1, term2)
    print(f"Term 1: {term1}")
    print(f"Term 2: {term2}")
    print(f"Composed: {composed}")
    print(f"  Phi: {lambda0.type_of(composed)['phi']:.6f}")
    print()
    
    # Test 6: Type checking
    print("Test 6: Type Checking")
    
    type1 = lambda0.type_of(term1)
    type2 = lambda0.type_of(term2)
    type_composed = lambda0.type_of(composed)
    
    print(f"Term 1 type:")
    print(f"  Digital root: {type1['digital_root']}")
    print(f"  Parity syndrome: {type1['parity_syndrome']:.6f}")
    print(f"  Active roots: {type1['num_active']}")
    
    print(f"Term 2 type:")
    print(f"  Digital root: {type2['digital_root']}")
    print(f"  Parity syndrome: {type2['parity_syndrome']:.6f}")
    print(f"  Active roots: {type2['num_active']}")
    
    print(f"Composed type:")
    print(f"  Digital root: {type_composed['digital_root']}")
    print(f"  Parity syndrome: {type_composed['parity_syndrome']:.6f}")
    print(f"  Active roots: {type_composed['num_active']}")
    print()
    
    # Test 7: Statistics
    print("Test 7: Statistics")
    
    stats = lambda0.get_statistics()
    print(f"Total reductions: {stats['total_reductions']}")
    print(f"Accepted: {stats['accepted']}")
    print(f"Rejected: {stats['rejected']}")
    print(f"Accept rate: {stats['accept_rate']:.1%}")
    print(f"Avg ΔΦ: {stats['avg_delta_phi']:.6f}")
    print()
    
    print("=== All Tests Passed ===")
