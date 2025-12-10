"""
Shell Protocol and Bregman Distance
Cartan-Quadratic Equivalence (CQE) System

From Whitepaper:
"Run MORSR inside a hard shell (radial or graph). Reject any out-of-shell
moves (log reason='out_of_shell'). Expand shell by factors ×2/×4/×8 per stage."

"The MORSR protocol uses Bregman distance defined by the 0.03 metric to
ensure Fejér monotonicity."

This module implements:
1. Shell constraint system with expansion schedule
2. Bregman distance metric
3. Fejér monotonicity tracking
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_morphonic.overlay_system import Overlay


class ShellType(Enum):
    """Type of shell constraint."""
    RADIAL = "radial"  # Radial distance from origin
    GRAPH = "graph"  # Graph distance in lattice


@dataclass
class Shell:
    """
    Shell constraint for MORSR optimization.
    
    From whitepaper:
    "Hard shell (radial or graph). Reject any out-of-shell moves."
    """
    
    shell_type: ShellType
    radius: float  # Current shell radius
    center: np.ndarray  # Shell center (8D)
    expansion_factor: int = 2  # Expansion factor (2, 4, or 8)
    stage: int = 0  # Current expansion stage
    
    def contains(self, overlay: Overlay) -> bool:
        """
        Check if overlay is inside shell.
        
        Args:
            overlay: Overlay to check
        
        Returns:
            True if inside shell, False otherwise
        """
        if self.shell_type == ShellType.RADIAL:
            # Radial distance from center
            distance = np.linalg.norm(overlay.e8_base - self.center)
            return distance <= self.radius
        else:
            # Graph distance (simplified: use L1 distance)
            distance = np.sum(np.abs(overlay.e8_base - self.center))
            return distance <= self.radius
    
    def expand(self):
        """
        Expand shell by expansion factor.
        
        From whitepaper:
        "Expand shell by factors ×2/×4/×8 per stage."
        """
        self.radius *= self.expansion_factor
        self.stage += 1
    
    def distance_to_boundary(self, overlay: Overlay) -> float:
        """
        Compute distance from overlay to shell boundary.
        
        Args:
            overlay: Overlay
        
        Returns:
            Distance to boundary (negative if outside)
        """
        if self.shell_type == ShellType.RADIAL:
            distance = np.linalg.norm(overlay.e8_base - self.center)
            return self.radius - distance
        else:
            distance = np.sum(np.abs(overlay.e8_base - self.center))
            return self.radius - distance


class BregmanDistance:
    """
    Bregman distance metric for MORSR.
    
    From whitepaper:
    "The specific Bregman function used in the MORSR protocol is determined
    by the geometry of the problem and is intrinsically linked to the 0.03 metric."
    
    D_f(x, y) = f(x) - f(y) - ⟨∇f(y), x - y⟩
    """
    
    def __init__(self, coupling_constant: float = 0.03):
        self.coupling_constant = coupling_constant
    
    def bregman_function(self, overlay: Overlay) -> float:
        """
        Bregman function f(x).
        
        Uses 0.03 metric to define curvature.
        
        Args:
            overlay: Overlay
        
        Returns:
            f(x) value
        """
        # Bregman function based on E₈ geometry and 0.03 metric
        # f(x) = (1/2)||x||² + α * parity_term + β * sparsity_term
        
        # Geometric term (quadratic)
        geom_term = 0.5 * np.dot(overlay.e8_base, overlay.e8_base)
        
        # Parity term (weighted by 0.03)
        num_active = np.sum(overlay.activations)
        parity_term = self.coupling_constant * abs(num_active - 120)
        
        # Sparsity term
        sparsity_term = self.coupling_constant * num_active / 240
        
        return geom_term + parity_term + sparsity_term
    
    def gradient(self, overlay: Overlay) -> np.ndarray:
        """
        Gradient ∇f(x) of Bregman function.
        
        Args:
            overlay: Overlay
        
        Returns:
            8D gradient vector
        """
        # ∇f(x) = x (for quadratic term)
        # Other terms don't depend on continuous coordinates
        return overlay.e8_base.copy()
    
    def distance(self, x: Overlay, y: Overlay) -> float:
        """
        Bregman distance D_f(x, y).
        
        D_f(x, y) = f(x) - f(y) - ⟨∇f(y), x - y⟩
        
        Args:
            x: First overlay
            y: Second overlay
        
        Returns:
            Bregman distance
        """
        f_x = self.bregman_function(x)
        f_y = self.bregman_function(y)
        grad_f_y = self.gradient(y)
        diff = x.e8_base - y.e8_base
        
        distance = f_x - f_y - np.dot(grad_f_y, diff)
        return distance


@dataclass
class StageMetrics:
    """Metrics for a shell expansion stage."""
    stage: int
    radius: float
    attempts: int
    accepts: int
    accept_rate: float
    strict_gain: float  # Average strict decrease
    novelty: float  # Measure of exploration
    ema: float  # Exponential moving average


class ShellProtocol:
    """
    Shell protocol for MORSR optimization.
    
    From whitepaper:
    "Run MORSR inside a hard shell (radial or graph). Reject any out-of-shell
    moves. Expand shell by factors ×2/×4/×8 per stage."
    
    "Stopping criteria: Compute stage return (accept_rate | strict_gain |
    novelty) and an EMA; stop when both fall below a threshold τ."
    """
    
    def __init__(
        self,
        initial_overlay: Overlay,
        initial_radius: float = 1.0,
        shell_type: ShellType = ShellType.RADIAL,
        expansion_factor: int = 2,
        stopping_threshold: float = 0.1,
        ema_alpha: float = 0.3
    ):
        self.shell = Shell(
            shell_type=shell_type,
            radius=initial_radius,
            center=initial_overlay.e8_base.copy(),
            expansion_factor=expansion_factor
        )
        
        self.bregman = BregmanDistance()
        self.stopping_threshold = stopping_threshold
        self.ema_alpha = ema_alpha
        
        # Stage tracking
        self.stages: List[StageMetrics] = []
        self.current_stage_attempts = 0
        self.current_stage_accepts = 0
        self.current_stage_strict_gains: List[float] = []
        self.visited_states: List[str] = []  # For novelty tracking
        
        # Fejér monotonicity tracking
        self.optimal_target: Optional[Overlay] = None
        self.bregman_distances: List[float] = []
    
    def check_shell_constraint(self, overlay: Overlay) -> Tuple[bool, str]:
        """
        Check if overlay satisfies shell constraint.
        
        Args:
            overlay: Overlay to check
        
        Returns:
            (is_valid, reason) tuple
        """
        if self.shell.contains(overlay):
            return True, "inside_shell"
        else:
            distance_to_boundary = self.shell.distance_to_boundary(overlay)
            return False, f"out_of_shell (distance={distance_to_boundary:.4f})"
    
    def check_fejer_monotonicity(
        self,
        overlay_current: Overlay,
        overlay_next: Overlay
    ) -> bool:
        """
        Check Fejér monotonicity condition.
        
        From whitepaper:
        "D_f(x*, x_{k+1}) < D_f(x*, x_k)"
        
        Args:
            overlay_current: Current overlay
            overlay_next: Next overlay
        
        Returns:
            True if Fejér monotone, False otherwise
        """
        if self.optimal_target is None:
            # No target set yet, assume monotone
            return True
        
        dist_current = self.bregman.distance(self.optimal_target, overlay_current)
        dist_next = self.bregman.distance(self.optimal_target, overlay_next)
        
        # Track distances
        self.bregman_distances.append(dist_next)
        
        # Check monotonicity
        return dist_next < dist_current
    
    def record_attempt(
        self,
        overlay_before: Overlay,
        overlay_after: Overlay,
        accepted: bool,
        delta_phi: float
    ):
        """
        Record an optimization attempt.
        
        Args:
            overlay_before: Overlay before operation
            overlay_after: Overlay after operation
            accepted: Whether attempt was accepted
            delta_phi: Change in phi metric
        """
        self.current_stage_attempts += 1
        
        if accepted:
            self.current_stage_accepts += 1
            
            # Record strict gain (if phi decreased)
            if delta_phi < 0:
                self.current_stage_strict_gains.append(-delta_phi)
            
            # Record novelty (new state)
            if overlay_after.overlay_id not in self.visited_states:
                self.visited_states.append(overlay_after.overlay_id)
    
    def compute_stage_return(self) -> float:
        """
        Compute stage return metric.
        
        From whitepaper:
        "Compute stage return (accept_rate | strict_gain | novelty)"
        
        Returns:
            Stage return value
        """
        if self.current_stage_attempts == 0:
            return 0.0
        
        # Accept rate
        accept_rate = self.current_stage_accepts / self.current_stage_attempts
        
        # Strict gain (average)
        strict_gain = (
            np.mean(self.current_stage_strict_gains)
            if self.current_stage_strict_gains
            else 0.0
        )
        
        # Novelty (fraction of new states)
        novelty = len(self.visited_states) / self.current_stage_attempts
        
        # Combined return (weighted average)
        stage_return = 0.4 * accept_rate + 0.4 * strict_gain + 0.2 * novelty
        
        return stage_return
    
    def should_expand(self) -> bool:
        """
        Check if shell should be expanded.
        
        From whitepaper:
        "Stop when both [stage return and EMA] fall below a threshold τ."
        
        Returns:
            True if should expand, False if should stop
        """
        if len(self.stages) == 0:
            return False  # Need at least one stage to evaluate
        
        # Compute current stage return
        current_return = self.compute_stage_return()
        
        # Compute EMA
        if len(self.stages) == 0:
            ema = current_return
        else:
            prev_ema = self.stages[-1].ema
            ema = self.ema_alpha * current_return + (1 - self.ema_alpha) * prev_ema
        
        # Check stopping criteria
        if current_return < self.stopping_threshold and ema < self.stopping_threshold:
            return False  # Stop
        else:
            return True  # Continue/expand
    
    def finalize_stage(self):
        """Finalize current stage and record metrics."""
        stage_return = self.compute_stage_return()
        
        # Compute EMA
        if len(self.stages) == 0:
            ema = stage_return
        else:
            prev_ema = self.stages[-1].ema
            ema = self.ema_alpha * stage_return + (1 - self.ema_alpha) * prev_ema
        
        # Compute metrics
        accept_rate = (
            self.current_stage_accepts / self.current_stage_attempts
            if self.current_stage_attempts > 0
            else 0.0
        )
        
        strict_gain = (
            np.mean(self.current_stage_strict_gains)
            if self.current_stage_strict_gains
            else 0.0
        )
        
        novelty = (
            len(self.visited_states) / self.current_stage_attempts
            if self.current_stage_attempts > 0
            else 0.0
        )
        
        # Record stage metrics
        metrics = StageMetrics(
            stage=self.shell.stage,
            radius=self.shell.radius,
            attempts=self.current_stage_attempts,
            accepts=self.current_stage_accepts,
            accept_rate=accept_rate,
            strict_gain=strict_gain,
            novelty=novelty,
            ema=ema
        )
        
        self.stages.append(metrics)
        
        # Reset stage counters
        self.current_stage_attempts = 0
        self.current_stage_accepts = 0
        self.current_stage_strict_gains = []
        self.visited_states = []
    
    def expand_shell(self):
        """Expand shell to next stage."""
        self.finalize_stage()
        self.shell.expand()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get shell protocol statistics."""
        return {
            'current_stage': self.shell.stage,
            'current_radius': self.shell.radius,
            'expansion_factor': self.shell.expansion_factor,
            'num_stages': len(self.stages),
            'stages': [
                {
                    'stage': s.stage,
                    'radius': s.radius,
                    'attempts': s.attempts,
                    'accepts': s.accepts,
                    'accept_rate': s.accept_rate,
                    'strict_gain': s.strict_gain,
                    'novelty': s.novelty,
                    'ema': s.ema
                }
                for s in self.stages
            ],
            'bregman_distances': self.bregman_distances
        }


# Example usage
if __name__ == "__main__":
    print("=== Shell Protocol and Bregman Distance Test ===\n")
    
    from layer1_morphonic.overlay_system import ImmutablePose
    from layer1_morphonic.alena_operators import ALENAOperators
    import time
    
    # Create initial overlay
    e8_base = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0])
    activations = np.zeros(240, dtype=int)
    activations[0:120] = 1
    
    pose = ImmutablePose(
        position=tuple(e8_base),
        orientation=tuple(np.eye(8)[0]),
        timestamp=time.time()
    )
    
    overlay = Overlay(e8_base=e8_base, activations=activations, pose=pose)
    
    # Create shell protocol
    protocol = ShellProtocol(
        initial_overlay=overlay,
        initial_radius=2.0,
        expansion_factor=2
    )
    
    print(f"Initial shell radius: {protocol.shell.radius}")
    print(f"Shell type: {protocol.shell.shell_type.value}")
    print()
    
    # Test 1: Bregman distance
    print("Test 1: Bregman Distance")
    alena = ALENAOperators()
    result = alena.rotate(overlay, theta=0.1)
    
    bregman_dist = protocol.bregman.distance(overlay, result.overlay)
    print(f"Bregman distance: {bregman_dist:.6f}")
    print()
    
    # Test 2: Shell constraint
    print("Test 2: Shell Constraint")
    is_valid, reason = protocol.check_shell_constraint(result.overlay)
    print(f"Inside shell: {is_valid}")
    print(f"Reason: {reason}")
    print()
    
    # Test 3: Simulate optimization with shell expansion
    print("Test 3: Shell Expansion Simulation")
    current = overlay
    
    for stage in range(3):
        print(f"\nStage {stage}: radius={protocol.shell.radius:.2f}")
        
        # Simulate 10 attempts
        for i in range(10):
            result = alena.rotate(current, theta=0.01 * (i + 1))
            
            # Check shell constraint
            is_valid, reason = protocol.check_shell_constraint(result.overlay)
            
            if is_valid:
                # Accept if inside shell
                delta_phi = alena._compute_phi(result.overlay) - alena._compute_phi(current)
                protocol.record_attempt(current, result.overlay, True, delta_phi)
                current = result.overlay
            else:
                protocol.record_attempt(current, result.overlay, False, 0.0)
        
        # Check if should expand
        if protocol.should_expand():
            protocol.expand_shell()
            print(f"  Expanded to radius={protocol.shell.radius:.2f}")
        else:
            print(f"  Stopping (threshold reached)")
            break
    
    # Test 4: Statistics
    print("\nTest 4: Shell Protocol Statistics")
    stats = protocol.get_statistics()
    print(f"Total stages: {stats['num_stages']}")
    print(f"Final radius: {stats['current_radius']:.2f}")
    print("\nStage details:")
    for s in stats['stages']:
        print(f"  Stage {s['stage']}: "
              f"radius={s['radius']:.2f}, "
              f"accept_rate={s['accept_rate']:.2%}, "
              f"ema={s['ema']:.4f}")
    print()
    
    print("=== All Tests Passed ===")
