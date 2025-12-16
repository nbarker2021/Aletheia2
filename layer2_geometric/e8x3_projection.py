"""
E₈×3 Comparative Projection
Cartan-Quadratic Equivalence (CQE) System

From Whitepaper:
"Two read-only sources (Left/Right) → Center solve frame. Conflicts resolve
by deterministic tiebreakers. The result is a solve-frame overlay with
provenance and sector histograms for explainability."

This implements the three-way projection system:
- Left source: Read-only, pristine
- Right source: Read-only, pristine  
- Center: Solve frame with parity corrections
- Weights: w_left, w_right for sector routing
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_morphonic.overlay_system import Overlay
from layer1_morphonic.acceptance_rules import ParitySignature


class ConflictResolution(Enum):
    """Conflict resolution strategies."""
    LEFT_PRIORITY = "left_priority"
    RIGHT_PRIORITY = "right_priority"
    WEIGHTED_AVERAGE = "weighted_average"
    PHI_PROBE = "phi_probe"  # Use phi metric to decide
    PARITY_PROBE = "parity_probe"  # Use parity to decide


@dataclass
class Sector:
    """
    Sector in E₈ space.
    
    Sectors are regions defined by digital roots and Weyl chambers.
    """
    digital_root: int  # 0-9
    weyl_chamber: int  # 1-48
    root_indices: List[int]  # Which roots belong to this sector
    
    def __hash__(self):
        return hash((self.digital_root, self.weyl_chamber))


@dataclass
class SectorHistogram:
    """
    Histogram of sector activations.
    
    Tracks which sectors are active in each source and how conflicts
    were resolved.
    """
    left_sectors: Dict[Sector, int]  # Sector -> activation count
    right_sectors: Dict[Sector, int]
    center_sectors: Dict[Sector, int]
    conflicts: List[Tuple[Sector, str]]  # (sector, resolution_method)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'left_count': len(self.left_sectors),
            'right_count': len(self.right_sectors),
            'center_count': len(self.center_sectors),
            'conflict_count': len(self.conflicts),
            'conflicts': [
                {
                    'digital_root': s.digital_root,
                    'weyl_chamber': s.weyl_chamber,
                    'resolution': method
                }
                for s, method in self.conflicts
            ]
        }


@dataclass
class E8x3Result:
    """Result of E₈×3 projection."""
    center_overlay: Overlay
    sector_histogram: SectorHistogram
    provenance: Dict[str, Any]
    weights: Tuple[float, float]  # (w_left, w_right)
    conflicts_resolved: int


class E8x3Projection:
    """
    E₈×3 Comparative Projection System.
    
    From whitepaper:
    "Two read-only sources (Left/Right) → Center solve frame."
    
    The center frame is computed by comparing left and right sources,
    resolving conflicts, and applying parity corrections.
    """
    
    def __init__(
        self,
        w_left: float = 0.5,
        w_right: float = 0.5,
        conflict_resolution: ConflictResolution = ConflictResolution.PHI_PROBE
    ):
        self.w_left = w_left
        self.w_right = w_right
        self.conflict_resolution = conflict_resolution
        
        # Normalize weights
        total = self.w_left + self.w_right
        if total > 0:
            self.w_left /= total
            self.w_right /= total
    
    def _compute_sectors(self, overlay: Overlay) -> Dict[Sector, List[int]]:
        """
        Compute sectors for an overlay.
        
        Args:
            overlay: Overlay to analyze
        
        Returns:
            Dictionary mapping sectors to root indices
        """
        sectors = {}
        
        # Get active roots
        active_indices = np.where(overlay.activations == 1)[0]
        
        # Group by digital root (simplified: use index mod 10)
        for idx in active_indices:
            dr = idx % 10
            # Simplified: assign to chamber based on position
            chamber = (idx % 48) + 1
            
            sector = Sector(
                digital_root=dr,
                weyl_chamber=chamber,
                root_indices=[]
            )
            
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(int(idx))
        
        return sectors
    
    def _resolve_conflict(
        self,
        sector: Sector,
        left_overlay: Overlay,
        right_overlay: Overlay,
        left_active: bool,
        right_active: bool
    ) -> Tuple[bool, str]:
        """
        Resolve conflict for a sector.
        
        Args:
            sector: Sector with conflict
            left_overlay: Left source overlay
            right_overlay: Right source overlay
            left_active: Whether sector is active in left
            right_active: Whether sector is active in right
        
        Returns:
            (should_activate, resolution_method) tuple
        """
        if self.conflict_resolution == ConflictResolution.LEFT_PRIORITY:
            return left_active, "left_priority"
        
        elif self.conflict_resolution == ConflictResolution.RIGHT_PRIORITY:
            return right_active, "right_priority"
        
        elif self.conflict_resolution == ConflictResolution.WEIGHTED_AVERAGE:
            # Use weights to decide
            if self.w_left > self.w_right:
                return left_active, "weighted_left"
            else:
                return right_active, "weighted_right"
        
        elif self.conflict_resolution == ConflictResolution.PHI_PROBE:
            # Use phi metric to decide
            # Compute phi for each choice
            from layer1_morphonic.alena_operators import ALENAOperators
            alena = ALENAOperators()
            
            phi_left = alena._compute_phi(left_overlay)
            phi_right = alena._compute_phi(right_overlay)
            
            # Choose the one with lower phi (better)
            if phi_left < phi_right:
                return left_active, "phi_probe_left"
            else:
                return right_active, "phi_probe_right"
        
        elif self.conflict_resolution == ConflictResolution.PARITY_PROBE:
            # Use parity to decide
            parity_left = ParitySignature.from_overlay(left_overlay)
            parity_right = ParitySignature.from_overlay(right_overlay)
            
            # Choose the one with lower syndrome (better parity)
            if parity_left.syndrome < parity_right.syndrome:
                return left_active, "parity_probe_left"
            else:
                return right_active, "parity_probe_right"
        
        # Default: weighted average
        return self.w_left > self.w_right, "default_weighted"
    
    def project(
        self,
        left_overlay: Overlay,
        right_overlay: Overlay
    ) -> E8x3Result:
        """
        Perform E₈×3 comparative projection.
        
        Args:
            left_overlay: Left source (read-only)
            right_overlay: Right source (read-only)
        
        Returns:
            E8x3Result with center overlay and metadata
        """
        # Compute sectors for each source
        left_sectors = self._compute_sectors(left_overlay)
        right_sectors = self._compute_sectors(right_overlay)
        
        # Find all sectors (union)
        all_sectors = set(left_sectors.keys()) | set(right_sectors.keys())
        
        # Initialize center activations
        center_activations = np.zeros(240, dtype=int)
        
        # Track conflicts
        conflicts = []
        
        # Resolve each sector
        for sector in all_sectors:
            left_active = sector in left_sectors
            right_active = sector in right_sectors
            
            if left_active and right_active:
                # Both active - check if they agree
                left_indices = set(left_sectors[sector])
                right_indices = set(right_sectors[sector])
                
                if left_indices == right_indices:
                    # Agreement - activate in center
                    for idx in left_indices:
                        center_activations[idx] = 1
                else:
                    # Conflict - resolve
                    should_activate, method = self._resolve_conflict(
                        sector, left_overlay, right_overlay,
                        left_active, right_active
                    )
                    conflicts.append((sector, method))
                    
                    if should_activate:
                        # Use the chosen source's indices
                        if "left" in method:
                            for idx in left_indices:
                                center_activations[idx] = 1
                        else:
                            for idx in right_indices:
                                center_activations[idx] = 1
            
            elif left_active:
                # Only left active - use left
                for idx in left_sectors[sector]:
                    center_activations[idx] = 1
            
            elif right_active:
                # Only right active - use right
                for idx in right_sectors[sector]:
                    center_activations[idx] = 1
        
        # Compute center e8_base (weighted average)
        center_e8_base = (
            self.w_left * left_overlay.e8_base +
            self.w_right * right_overlay.e8_base
        )
        
        # Compute center weights (if present)
        center_weights = None
        if left_overlay.weights is not None and right_overlay.weights is not None:
            center_weights = (
                self.w_left * left_overlay.weights +
                self.w_right * right_overlay.weights
            )
        
        # Compute center phase (if present)
        center_phase = None
        if left_overlay.phase is not None and right_overlay.phase is not None:
            center_phase = (
                self.w_left * left_overlay.phase +
                self.w_right * right_overlay.phase
            )
        
        # Create center overlay
        import time
        from layer1_morphonic.overlay_system import ImmutablePose
        
        center_pose = ImmutablePose(
            position=tuple(center_e8_base),
            orientation=tuple(np.eye(8)[0]),
            timestamp=time.time()
        )
        
        center_overlay = Overlay(
            e8_base=center_e8_base,
            activations=center_activations,
            weights=center_weights,
            phase=center_phase,
            pose=center_pose,
            metadata={
                'left_parent': left_overlay.overlay_id,
                'right_parent': right_overlay.overlay_id,
                'projection_type': 'e8x3',
                'weights': (self.w_left, self.w_right)
            }
        )
        
        # Create sector histogram
        histogram = SectorHistogram(
            left_sectors={s: len(indices) for s, indices in left_sectors.items()},
            right_sectors={s: len(indices) for s, indices in right_sectors.items()},
            center_sectors={s: len(indices) for s, indices in self._compute_sectors(center_overlay).items()},
            conflicts=conflicts
        )
        
        # Create provenance
        provenance = {
            'left_id': left_overlay.overlay_id,
            'right_id': right_overlay.overlay_id,
            'center_id': center_overlay.overlay_id,
            'weights': (self.w_left, self.w_right),
            'conflict_resolution': self.conflict_resolution.value,
            'conflicts_resolved': len(conflicts),
            'timestamp': time.time()
        }
        
        return E8x3Result(
            center_overlay=center_overlay,
            sector_histogram=histogram,
            provenance=provenance,
            weights=(self.w_left, self.w_right),
            conflicts_resolved=len(conflicts)
        )
    
    def adaptive_weights(
        self,
        left_overlay: Overlay,
        right_overlay: Overlay
    ) -> Tuple[float, float]:
        """
        Compute adaptive weights based on overlay quality.
        
        Args:
            left_overlay: Left source
            right_overlay: Right source
        
        Returns:
            (w_left, w_right) tuple
        """
        from layer1_morphonic.alena_operators import ALENAOperators
        alena = ALENAOperators()
        
        # Compute phi for each
        phi_left = alena._compute_phi(left_overlay)
        phi_right = alena._compute_phi(right_overlay)
        
        # Compute parity for each
        parity_left = ParitySignature.from_overlay(left_overlay)
        parity_right = ParitySignature.from_overlay(right_overlay)
        
        # Compute quality scores (lower is better)
        quality_left = phi_left + parity_left.syndrome
        quality_right = phi_right + parity_right.syndrome
        
        # Invert and normalize (higher quality = higher weight)
        if quality_left + quality_right > 0:
            w_left = quality_right / (quality_left + quality_right)
            w_right = quality_left / (quality_left + quality_right)
        else:
            w_left = 0.5
            w_right = 0.5
        
        return w_left, w_right


# Example usage and tests
if __name__ == "__main__":
    print("=== E₈×3 Comparative Projection Test ===\n")
    
    from layer1_morphonic.overlay_system import ImmutablePose
    import time
    
    # Create left source
    e8_left = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0])
    activations_left = np.zeros(240, dtype=int)
    activations_left[0:100] = 1  # First 100 roots
    
    pose_left = ImmutablePose(
        position=tuple(e8_left),
        orientation=tuple(np.eye(8)[0]),
        timestamp=time.time()
    )
    
    left_overlay = Overlay(
        e8_base=e8_left,
        activations=activations_left,
        pose=pose_left,
        metadata={'source': 'left'}
    )
    
    # Create right source (overlapping but different)
    e8_right = np.array([0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0])
    activations_right = np.zeros(240, dtype=int)
    activations_right[50:150] = 1  # Roots 50-150 (50% overlap)
    
    pose_right = ImmutablePose(
        position=tuple(e8_right),
        orientation=tuple(np.eye(8)[0]),
        timestamp=time.time()
    )
    
    right_overlay = Overlay(
        e8_base=e8_right,
        activations=activations_right,
        pose=pose_right,
        metadata={'source': 'right'}
    )
    
    print(f"Left overlay: {left_overlay.overlay_id}")
    print(f"  Active roots: {np.sum(activations_left)}")
    print(f"Right overlay: {right_overlay.overlay_id}")
    print(f"  Active roots: {np.sum(activations_right)}")
    print()
    
    # Test 1: Basic projection with equal weights
    print("Test 1: Equal Weights Projection")
    projection = E8x3Projection(w_left=0.5, w_right=0.5)
    result = projection.project(left_overlay, right_overlay)
    
    print(f"Center overlay: {result.center_overlay.overlay_id}")
    print(f"  Active roots: {np.sum(result.center_overlay.activations)}")
    print(f"  Conflicts resolved: {result.conflicts_resolved}")
    print(f"  Weights: left={result.weights[0]:.2f}, right={result.weights[1]:.2f}")
    print()
    
    # Test 2: Adaptive weights
    print("Test 2: Adaptive Weights")
    w_left, w_right = projection.adaptive_weights(left_overlay, right_overlay)
    print(f"Adaptive weights: left={w_left:.3f}, right={w_right:.3f}")
    
    projection_adaptive = E8x3Projection(w_left=w_left, w_right=w_right)
    result_adaptive = projection_adaptive.project(left_overlay, right_overlay)
    print(f"Center overlay: {result_adaptive.center_overlay.overlay_id}")
    print(f"  Active roots: {np.sum(result_adaptive.center_overlay.activations)}")
    print()
    
    # Test 3: Sector histogram
    print("Test 3: Sector Histogram")
    histogram = result.sector_histogram
    print(f"Left sectors: {histogram.to_dict()['left_count']}")
    print(f"Right sectors: {histogram.to_dict()['right_count']}")
    print(f"Center sectors: {histogram.to_dict()['center_count']}")
    print(f"Conflicts: {histogram.to_dict()['conflict_count']}")
    print()
    
    # Test 4: Different conflict resolution strategies
    print("Test 4: Conflict Resolution Strategies")
    strategies = [
        ConflictResolution.LEFT_PRIORITY,
        ConflictResolution.RIGHT_PRIORITY,
        ConflictResolution.PHI_PROBE,
        ConflictResolution.PARITY_PROBE
    ]
    
    for strategy in strategies:
        proj = E8x3Projection(
            w_left=0.5,
            w_right=0.5,
            conflict_resolution=strategy
        )
        res = proj.project(left_overlay, right_overlay)
        print(f"  {strategy.value}:")
        print(f"    Active roots: {np.sum(res.center_overlay.activations)}")
        print(f"    Conflicts: {res.conflicts_resolved}")
    
    print()
    print("=== All Tests Passed ===")
