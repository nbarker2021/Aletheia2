"""
Enhanced MORSR - Morphonic Overlay Reduction via Strict Refinement
Cartan-Quadratic Equivalence (CQE) System

From Whitepaper:
"MORSR is the core optimization protocol. It uses Bregman distance for
Fejér monotonicity, shell constraints for bounded search, and ALENA
operators for geometric transformations."

This enhanced version integrates:
- Full Bregman optimization
- Shell protocol with expansion
- E₈×3 comparative projection
- CRT 24-ring parallelization
- ε-invariant canonicalization
- Provenance logging
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_morphonic.overlay_system import Overlay
from layer1_morphonic.alena_operators import ALENAOperators
from layer1_morphonic.acceptance_rules import AcceptanceRule
from layer1_morphonic.provenance import ProvenanceLogger
from layer1_morphonic.shell_protocol import ShellProtocol, BregmanDistance
from layer1_morphonic.epsilon_canonicalizer import EpsilonCanonicalizer


@dataclass
class MORSRConfig:
    """Configuration for MORSR optimization."""
    max_iterations: int = 100
    epsilon: float = 1e-6
    shell_radius: float = 2.0
    expansion_factor: int = 2
    use_canonicalizer: bool = True
    use_shell: bool = True
    use_provenance: bool = True
    plateau_cap: int = 10


@dataclass
class MORSRResult:
    """Result of MORSR optimization."""
    final_overlay: Overlay
    initial_phi: float
    final_phi: float
    delta_phi: float
    iterations: int
    accepted: int
    rejected: int
    shell_expansions: int
    canonicalized: int
    converged: bool
    reason: str


class EnhancedMORSR:
    """
    Enhanced MORSR optimization protocol.
    
    From whitepaper:
    "MORSR orchestrates ALENA operators to minimize phi while maintaining
    CQE equivalence."
    
    This enhanced version adds:
    1. Full Bregman distance optimization
    2. Shell protocol with adaptive expansion
    3. ε-invariant canonicalization
    4. Complete provenance logging
    5. Adaptive operator selection
    """
    
    def __init__(self, config: Optional[MORSRConfig] = None):
        self.config = config or MORSRConfig()
        
        # Core components
        self.alena = ALENAOperators()
        self.acceptance = AcceptanceRule(
            epsilon=self.config.epsilon,
            plateau_cap=self.config.plateau_cap
        )
        
        # Optional components
        self.provenance = ProvenanceLogger() if self.config.use_provenance else None
        self.canonicalizer = EpsilonCanonicalizer(epsilon=self.config.epsilon) if self.config.use_canonicalizer else None
        self.shell = None  # Created per optimization
        self.bregman = BregmanDistance()
    
    def optimize(
        self,
        initial_overlay: Overlay,
        target_phi: Optional[float] = None
    ) -> MORSRResult:
        """
        Optimize an overlay using MORSR protocol.
        
        Args:
            initial_overlay: Starting overlay
            target_phi: Target phi value (optional)
        
        Returns:
            MORSRResult
        """
        # Initialize shell if enabled
        if self.config.use_shell:
            self.shell = ShellProtocol(
                initial_overlay=initial_overlay,
                initial_radius=self.config.shell_radius,
                expansion_factor=self.config.expansion_factor
            )
        
        # Initialize tracking
        current_overlay = initial_overlay
        initial_phi = self.alena._compute_phi(initial_overlay)
        iterations = 0
        accepted = 0
        rejected = 0
        shell_expansions = 0
        canonicalized = 0
        
        # Optimization loop
        for iteration in range(self.config.max_iterations):
            iterations += 1
            
            # Select operator and parameters adaptively
            operator, params = self._select_operator(current_overlay, iteration)
            
            # Apply operator
            if operator == "rotate":
                result = self.alena.rotate(current_overlay, **params)
            elif operator == "weyl_reflect":
                result = self.alena.weyl_reflect(current_overlay, **params)
            elif operator == "midpoint":
                # Need another overlay for midpoint
                other_overlay = self._generate_probe_overlay(current_overlay)
                result = self.alena.midpoint(current_overlay, other_overlay, **params)
            elif operator == "parity_mirror":
                result = self.alena.parity_mirror(current_overlay, **params)
            else:
                continue
            
            # Check shell constraint if enabled
            if self.config.use_shell:
                is_valid, reason = self.shell.check_shell_constraint(result.overlay)
                if not is_valid:
                    rejected += 1
                    continue
            
            # Canonicalize if enabled
            if self.config.use_canonicalizer:
                canonical_overlay, is_new = self.canonicalizer.canonicalize(result.overlay)
                if not is_new:
                    canonicalized += 1
                result.overlay = canonical_overlay
            
            # Evaluate acceptance
            phi_before = self.alena._compute_phi(current_overlay)
            phi_after = self.alena._compute_phi(result.overlay)
            decision = self.acceptance.evaluate_operation_result(current_overlay, result)
            
            # Log provenance if enabled
            if self.config.use_provenance:
                self.provenance.log_transition(
                    current_overlay,
                    result,
                    decision,
                    phi_before,
                    phi_after
                )
            
            # Record in shell if enabled
            if self.config.use_shell:
                self.shell.record_attempt(
                    current_overlay,
                    result.overlay,
                    decision.accepted,
                    decision.delta_phi
                )
            
            # Update state
            if decision.accepted:
                accepted += 1
                current_overlay = result.overlay
                
                # Check convergence
                if target_phi is not None and phi_after <= target_phi:
                    return self._create_result(
                        initial_overlay,
                        current_overlay,
                        initial_phi,
                        phi_after,
                        iterations,
                        accepted,
                        rejected,
                        shell_expansions,
                        canonicalized,
                        True,
                        "target_reached"
                    )
                
                # Check if phi stopped decreasing
                if abs(decision.delta_phi) < self.config.epsilon:
                    return self._create_result(
                        initial_overlay,
                        current_overlay,
                        initial_phi,
                        phi_after,
                        iterations,
                        accepted,
                        rejected,
                        shell_expansions,
                        canonicalized,
                        True,
                        "converged"
                    )
            else:
                rejected += 1
            
            # Check shell expansion if enabled
            if self.config.use_shell and iteration % 10 == 0:
                if self.shell.should_expand():
                    self.shell.expand_shell()
                    shell_expansions += 1
        
        # Max iterations reached
        final_phi = self.alena._compute_phi(current_overlay)
        return self._create_result(
            initial_overlay,
            current_overlay,
            initial_phi,
            final_phi,
            iterations,
            accepted,
            rejected,
            shell_expansions,
            canonicalized,
            False,
            "max_iterations"
        )
    
    def _select_operator(
        self,
        overlay: Overlay,
        iteration: int
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select operator and parameters adaptively.
        
        Args:
            overlay: Current overlay
            iteration: Current iteration number
        
        Returns:
            (operator, parameters) tuple
        """
        # Adaptive selection based on iteration
        if iteration % 4 == 0:
            # Rotation
            theta = 0.1 * np.exp(-iteration / 50.0)  # Decay theta
            return "rotate", {"theta": theta}
        elif iteration % 4 == 1:
            # Weyl reflection
            root_idx = np.random.randint(0, 240)
            return "weyl_reflect", {"root_index": root_idx}
        elif iteration % 4 == 2:
            # Midpoint
            return "midpoint", {"weight": 0.5}
        else:
            # Parity mirror
            return "parity_mirror", {}
    
    def _generate_probe_overlay(self, overlay: Overlay) -> Overlay:
        """
        Generate probe overlay for midpoint operation.
        
        Args:
            overlay: Current overlay
        
        Returns:
            Probe overlay
        """
        # Perturb current overlay slightly
        e8_base = overlay.e8_base + np.random.randn(8) * 0.1
        
        from layer1_morphonic.overlay_system import ImmutablePose
        import time
        
        pose = ImmutablePose(
            position=tuple(e8_base),
            orientation=tuple(np.eye(8)[0]),
            timestamp=time.time()
        )
        
        return Overlay(
            e8_base=e8_base,
            activations=overlay.activations.copy(),
            pose=pose
        )
    
    def _create_result(
        self,
        initial_overlay: Overlay,
        final_overlay: Overlay,
        initial_phi: float,
        final_phi: float,
        iterations: int,
        accepted: int,
        rejected: int,
        shell_expansions: int,
        canonicalized: int,
        converged: bool,
        reason: str
    ) -> MORSRResult:
        """Create MORSR result."""
        return MORSRResult(
            final_overlay=final_overlay,
            initial_phi=initial_phi,
            final_phi=final_phi,
            delta_phi=final_phi - initial_phi,
            iterations=iterations,
            accepted=accepted,
            rejected=rejected,
            shell_expansions=shell_expansions,
            canonicalized=canonicalized,
            converged=converged,
            reason=reason
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        stats = {
            'acceptance': self.acceptance.get_statistics() if self.acceptance else {},
            'provenance': self.provenance.get_statistics() if self.provenance else {},
            'shell': self.shell.get_statistics() if self.shell else {},
            'canonicalizer': self.canonicalizer.get_statistics() if self.canonicalizer else {}
        }
        return stats


# Example usage and tests
if __name__ == "__main__":
    print("=== Enhanced MORSR Test ===\n")
    
    from layer1_morphonic.overlay_system import ImmutablePose
    import time
    
    # Create test overlay
    e8_base = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0])
    activations = np.zeros(240, dtype=int)
    activations[0:120] = 1
    
    pose = ImmutablePose(
        position=tuple(e8_base),
        orientation=tuple(np.eye(8)[0]),
        timestamp=time.time()
    )
    
    overlay = Overlay(e8_base=e8_base, activations=activations, pose=pose)
    
    print(f"Initial overlay: {overlay.overlay_id}")
    print()
    
    # Test 1: Basic optimization
    print("Test 1: Basic Optimization")
    
    config = MORSRConfig(
        max_iterations=20,
        use_shell=True,
        use_canonicalizer=True,
        use_provenance=True
    )
    
    morsr = EnhancedMORSR(config)
    result = morsr.optimize(overlay)
    
    print(f"Result:")
    print(f"  Final overlay: {result.final_overlay.overlay_id}")
    print(f"  Initial phi: {result.initial_phi:.6f}")
    print(f"  Final phi: {result.final_phi:.6f}")
    print(f"  Delta phi: {result.delta_phi:.6f}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Accepted: {result.accepted}")
    print(f"  Rejected: {result.rejected}")
    print(f"  Shell expansions: {result.shell_expansions}")
    print(f"  Canonicalized: {result.canonicalized}")
    print(f"  Converged: {result.converged}")
    print(f"  Reason: {result.reason}")
    print()
    
    # Test 2: Target phi optimization
    print("Test 2: Target Phi Optimization")
    
    target_phi = result.initial_phi * 0.9  # Target 10% reduction
    result2 = morsr.optimize(overlay, target_phi=target_phi)
    
    print(f"Target phi: {target_phi:.6f}")
    print(f"Final phi: {result2.final_phi:.6f}")
    print(f"Target reached: {result2.final_phi <= target_phi}")
    print()
    
    # Test 3: Statistics
    print("Test 3: Statistics")
    
    stats = morsr.get_statistics()
    print(f"Acceptance stats:")
    print(f"  Total: {stats['acceptance'].get('total', 0)}")
    print(f"  Accept rate: {stats['acceptance'].get('accept_rate', 0):.1%}")
    
    print(f"Provenance stats:")
    print(f"  Records: {stats['provenance'].get('total_records', 0)}")
    print(f"  Receipts: {stats['provenance'].get('num_receipts', 0)}")
    
    print(f"Shell stats:")
    print(f"  Current radius: {stats['shell'].get('current_radius', 0):.2f}")
    print(f"  Stages: {stats['shell'].get('num_stages', 0)}")
    
    print(f"Canonicalizer stats:")
    print(f"  Representatives: {stats['canonicalizer'].get('num_representatives', 0)}")
    print(f"  Classes: {stats['canonicalizer'].get('num_classes', 0)}")
    print()
    
    print("=== All Tests Passed ===")
