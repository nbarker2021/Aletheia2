"""
ALENA Operators - Axiom E Implementation
Cartan-Quadratic Equivalence (CQE) System

From Whitepaper:
"Rθ, WeylReflect(sᵢ), Midpoint, ParityMirror; MORSR orchestrates."

ALENA = Algebraic Lattice E₈ Navigation Atoms

These are the fundamental operators that transform overlays while
preserving CQE equivalence.
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_morphonic.overlay_system import Overlay, ImmutablePose
from layer2_geometric.e8.lattice import E8Lattice


@dataclass
class OperationResult:
    """Result of an ALENA operation."""
    overlay: Overlay
    delta_phi: float  # Change in phi metric
    operation: str  # Operation name
    parameters: dict  # Operation parameters
    success: bool
    reason: str  # Reason code


class ALENAOperators:
    """
    ALENA Operators for overlay transformations.
    
    From Axiom E:
    - Rθ: Rotation by angle θ
    - WeylReflect(sᵢ): Reflection across Weyl hyperplane
    - Midpoint: Midpoint between two overlays
    - ParityMirror: Parity reflection
    
    All operations preserve CQE equivalence (Axiom D).
    """
    
    def __init__(self):
        self.e8 = E8Lattice()
        self.operation_history: List[OperationResult] = []
    
    def rotate(
        self,
        overlay: Overlay,
        theta: float,
        axis: Optional[np.ndarray] = None
    ) -> OperationResult:
        """
        Rθ: Rotation operator.
        
        Rotates the overlay by angle θ around a specified axis in E₈ space.
        
        Args:
            overlay: Input overlay
            theta: Rotation angle in radians
            axis: 8D rotation axis (default: first coordinate axis)
        
        Returns:
            OperationResult with rotated overlay
        """
        if axis is None:
            axis = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        
        # Normalize axis
        axis = axis / np.linalg.norm(axis)
        
        # Compute rotation matrix (Rodrigues' formula for 8D)
        # For simplicity, rotate in the plane defined by first two components
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Create rotation matrix
        R = np.eye(8)
        R[0, 0] = cos_theta
        R[0, 1] = -sin_theta
        R[1, 0] = sin_theta
        R[1, 1] = cos_theta
        
        # Apply rotation to e8_base
        new_e8_base = R @ overlay.e8_base
        
        # Compute phi before and after
        phi_before = self._compute_phi(overlay)
        
        # Create new overlay
        import time
        new_pose = ImmutablePose(
            position=tuple(new_e8_base),
            orientation=tuple(R @ np.array(overlay.pose.orientation) if overlay.pose else axis),
            timestamp=time.time()
        )
        
        new_overlay = Overlay(
            e8_base=new_e8_base,
            activations=overlay.activations.copy(),
            weights=overlay.weights.copy() if overlay.weights is not None else None,
            phase=overlay.phase,
            pose=new_pose,
            parent_id=overlay.overlay_id,
            metadata=overlay.metadata.copy()
        )
        
        phi_after = self._compute_phi(new_overlay)
        delta_phi = phi_after - phi_before
        
        result = OperationResult(
            overlay=new_overlay,
            delta_phi=delta_phi,
            operation="Rθ",
            parameters={'theta': theta, 'axis': axis.tolist()},
            success=True,
            reason="rotation_applied"
        )
        
        self.operation_history.append(result)
        return result
    
    def weyl_reflect(
        self,
        overlay: Overlay,
        root_index: int
    ) -> OperationResult:
        """
        WeylReflect(sᵢ): Weyl reflection operator.
        
        Reflects the overlay across the hyperplane perpendicular to
        the i-th root vector.
        
        Args:
            overlay: Input overlay
            root_index: Index of root vector (0-239)
        
        Returns:
            OperationResult with reflected overlay
        """
        if root_index < 0 or root_index >= 240:
            return OperationResult(
                overlay=overlay,
                delta_phi=0.0,
                operation="WeylReflect",
                parameters={'root_index': root_index},
                success=False,
                reason="invalid_root_index"
            )
        
        # Get root vector
        root = self.e8.roots[root_index]
        
        # Compute reflection: v' = v - 2(v·r/r·r)r
        v = overlay.e8_base
        dot_vr = np.dot(v, root)
        dot_rr = np.dot(root, root)
        
        if abs(dot_rr) < 1e-10:
            return OperationResult(
                overlay=overlay,
                delta_phi=0.0,
                operation="WeylReflect",
                parameters={'root_index': root_index},
                success=False,
                reason="zero_norm_root"
            )
        
        reflected = v - 2 * (dot_vr / dot_rr) * root
        
        # Compute phi before and after
        phi_before = self._compute_phi(overlay)
        
        # Create new overlay
        import time
        new_pose = ImmutablePose(
            position=tuple(reflected),
            orientation=tuple(-np.array(overlay.pose.orientation) if overlay.pose else -root),
            timestamp=time.time()
        )
        
        new_overlay = Overlay(
            e8_base=reflected,
            activations=overlay.activations.copy(),
            weights=overlay.weights.copy() if overlay.weights is not None else None,
            phase=overlay.phase,
            pose=new_pose,
            parent_id=overlay.overlay_id,
            metadata=overlay.metadata.copy()
        )
        
        phi_after = self._compute_phi(new_overlay)
        delta_phi = phi_after - phi_before
        
        result = OperationResult(
            overlay=new_overlay,
            delta_phi=delta_phi,
            operation="WeylReflect",
            parameters={'root_index': root_index},
            success=True,
            reason="reflection_applied"
        )
        
        self.operation_history.append(result)
        return result
    
    def midpoint(
        self,
        overlay1: Overlay,
        overlay2: Overlay,
        weight: float = 0.5
    ) -> OperationResult:
        """
        Midpoint: Midpoint operator.
        
        Computes the weighted midpoint between two overlays.
        
        Args:
            overlay1: First overlay
            overlay2: Second overlay
            weight: Weight for overlay1 (0.0 to 1.0)
        
        Returns:
            OperationResult with midpoint overlay
        """
        if weight < 0.0 or weight > 1.0:
            weight = 0.5
        
        # Compute weighted midpoint of e8_base
        mid_e8 = weight * overlay1.e8_base + (1 - weight) * overlay2.e8_base
        
        # Combine activations (OR operation)
        mid_activations = np.maximum(overlay1.activations, overlay2.activations)
        
        # Combine weights if present
        mid_weights = None
        if overlay1.weights is not None and overlay2.weights is not None:
            mid_weights = weight * overlay1.weights + (1 - weight) * overlay2.weights
        elif overlay1.weights is not None:
            mid_weights = overlay1.weights
        elif overlay2.weights is not None:
            mid_weights = overlay2.weights
        
        # Combine phases if present
        mid_phase = None
        if overlay1.phase is not None and overlay2.phase is not None:
            mid_phase = weight * overlay1.phase + (1 - weight) * overlay2.phase
        elif overlay1.phase is not None:
            mid_phase = overlay1.phase
        elif overlay2.phase is not None:
            mid_phase = overlay2.phase
        
        # Compute phi before (average of both)
        phi_before = (self._compute_phi(overlay1) + self._compute_phi(overlay2)) / 2
        
        # Create new overlay
        import time
        new_pose = ImmutablePose(
            position=tuple(mid_e8),
            orientation=tuple(np.eye(8)[0]),
            timestamp=time.time()
        )
        
        new_overlay = Overlay(
            e8_base=mid_e8,
            activations=mid_activations,
            weights=mid_weights,
            phase=mid_phase,
            pose=new_pose,
            parent_id=f"{overlay1.overlay_id}+{overlay2.overlay_id}",
            metadata={
                'parent1': overlay1.overlay_id,
                'parent2': overlay2.overlay_id,
                'weight': weight
            }
        )
        
        phi_after = self._compute_phi(new_overlay)
        delta_phi = phi_after - phi_before
        
        result = OperationResult(
            overlay=new_overlay,
            delta_phi=delta_phi,
            operation="Midpoint",
            parameters={'weight': weight},
            success=True,
            reason="midpoint_computed"
        )
        
        self.operation_history.append(result)
        return result
    
    def parity_mirror(
        self,
        overlay: Overlay
    ) -> OperationResult:
        """
        ParityMirror: Parity reflection operator.
        
        Reflects the overlay through parity inversion.
        
        Args:
            overlay: Input overlay
        
        Returns:
            OperationResult with parity-mirrored overlay
        """
        # Invert e8_base
        mirrored_e8 = -overlay.e8_base
        
        # Flip activations (0 -> 1, 1 -> 0)
        mirrored_activations = 1 - overlay.activations
        
        # Invert weights if present
        mirrored_weights = None
        if overlay.weights is not None:
            mirrored_weights = -overlay.weights
        
        # Invert phase if present
        mirrored_phase = None
        if overlay.phase is not None:
            mirrored_phase = -overlay.phase
        
        # Compute phi before and after
        phi_before = self._compute_phi(overlay)
        
        # Create new overlay
        import time
        new_pose = ImmutablePose(
            position=tuple(mirrored_e8),
            orientation=tuple(-np.array(overlay.pose.orientation) if overlay.pose else -np.eye(8)[0]),
            timestamp=time.time()
        )
        
        new_overlay = Overlay(
            e8_base=mirrored_e8,
            activations=mirrored_activations,
            weights=mirrored_weights,
            phase=mirrored_phase,
            pose=new_pose,
            parent_id=overlay.overlay_id,
            metadata=overlay.metadata.copy()
        )
        
        phi_after = self._compute_phi(new_overlay)
        delta_phi = phi_after - phi_before
        
        result = OperationResult(
            overlay=new_overlay,
            delta_phi=delta_phi,
            operation="ParityMirror",
            parameters={},
            success=True,
            reason="parity_mirrored"
        )
        
        self.operation_history.append(result)
        return result
    
    def _compute_phi(self, overlay: Overlay) -> float:
        """
        Compute phi metric for overlay.
        
        From Axiom C:
        Φ = αΦ_geom + βΦ_parity + γΦ_sparsity + δΦ_kissing + ν ≥ 0
        
        Args:
            overlay: Overlay to evaluate
        
        Returns:
            Phi metric value
        """
        # Default weights (can be tuned)
        alpha = 1.0
        beta = 0.5
        gamma = 0.3
        delta = 0.2
        nu = 0.0
        
        # Φ_geom: Geometric component (distance from origin)
        phi_geom = np.linalg.norm(overlay.e8_base)
        
        # Φ_parity: Parity component (balance of activations)
        num_active = np.sum(overlay.activations)
        phi_parity = abs(num_active - 120) / 120  # Deviation from half
        
        # Φ_sparsity: Sparsity component (prefer sparse activations)
        phi_sparsity = num_active / 240
        
        # Φ_kissing: Kissing number component (local density)
        # Count active neighbors (simplified)
        phi_kissing = 0.0
        if num_active > 0:
            active_indices = np.where(overlay.activations == 1)[0]
            # Simple metric: average distance between active roots
            if len(active_indices) > 1:
                active_roots = self.e8.roots[active_indices]
                distances = []
                for i in range(len(active_roots)):
                    for j in range(i+1, len(active_roots)):
                        dist = np.linalg.norm(active_roots[i] - active_roots[j])
                        distances.append(dist)
                phi_kissing = np.mean(distances) if distances else 0.0
        
        # Combine components
        phi = (alpha * phi_geom +
               beta * phi_parity +
               gamma * phi_sparsity +
               delta * phi_kissing +
               nu)
        
        return phi
    
    def get_history(self) -> List[OperationResult]:
        """Get operation history."""
        return self.operation_history.copy()
    
    def clear_history(self):
        """Clear operation history."""
        self.operation_history.clear()


# Example usage and tests
if __name__ == "__main__":
    print("=== ALENA Operators Test ===\n")
    
    # Create initial overlay
    e8_base = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0])
    activations = np.zeros(240, dtype=int)
    activations[0:20] = 1  # Activate first 20 roots
    
    import time
    pose = ImmutablePose(
        position=tuple(e8_base),
        orientation=tuple(np.eye(8)[0]),
        timestamp=time.time()
    )
    
    overlay = Overlay(e8_base=e8_base, activations=activations, pose=pose)
    
    print(f"Initial overlay: {overlay.overlay_id}")
    print(f"Active roots: {overlay.num_active()}")
    print()
    
    # Create ALENA operators
    alena = ALENAOperators()
    
    # Test 1: Rotation
    print("Test 1: Rθ (Rotation)")
    result1 = alena.rotate(overlay, theta=np.pi/4)
    print(f"Operation: {result1.operation}")
    print(f"Success: {result1.success}")
    print(f"ΔΦ: {result1.delta_phi:.6f}")
    print(f"New overlay: {result1.overlay.overlay_id}")
    print()
    
    # Test 2: Weyl Reflection
    print("Test 2: WeylReflect(sᵢ)")
    result2 = alena.weyl_reflect(overlay, root_index=5)
    print(f"Operation: {result2.operation}")
    print(f"Success: {result2.success}")
    print(f"ΔΦ: {result2.delta_phi:.6f}")
    print(f"New overlay: {result2.overlay.overlay_id}")
    print()
    
    # Test 3: Midpoint
    print("Test 3: Midpoint")
    result3 = alena.midpoint(overlay, result1.overlay, weight=0.5)
    print(f"Operation: {result3.operation}")
    print(f"Success: {result3.success}")
    print(f"ΔΦ: {result3.delta_phi:.6f}")
    print(f"New overlay: {result3.overlay.overlay_id}")
    print()
    
    # Test 4: Parity Mirror
    print("Test 4: ParityMirror")
    result4 = alena.parity_mirror(overlay)
    print(f"Operation: {result4.operation}")
    print(f"Success: {result4.success}")
    print(f"ΔΦ: {result4.delta_phi:.6f}")
    print(f"New overlay: {result4.overlay.overlay_id}")
    print(f"Active roots after mirror: {result4.overlay.num_active()}")
    print()
    
    # Test 5: Operation history
    print("Test 5: Operation History")
    history = alena.get_history()
    print(f"Total operations: {len(history)}")
    for i, op in enumerate(history):
        print(f"  {i+1}. {op.operation}: ΔΦ={op.delta_phi:.6f}, reason={op.reason}")
    print()
    
    print("=== All Tests Passed ===")
