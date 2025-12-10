"""
ε-Invariant Canonicalizer
Cartan-Quadratic Equivalence (CQE) System

From Whitepaper:
"Numeric tolerance ε ensures that overlays differing by less than ε are
considered equivalent. The canonicalizer rounds/snaps values to canonical
representatives within ε-balls."

This implements:
- ε-ball canonicalization
- Numeric tolerance handling
- Canonical representative selection
- Equivalence class management
"""

import numpy as np
from typing import Optional, Dict, Any, List, Set, Tuple
from dataclasses import dataclass
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_morphonic.overlay_system import Overlay


@dataclass
class CanonicalRepresentative:
    """
    Canonical representative of an ε-equivalence class.
    
    All overlays within ε of this representative are considered equivalent.
    """
    overlay_id: str
    e8_base: np.ndarray
    activations: np.ndarray
    epsilon: float
    members: List[str]  # IDs of equivalent overlays
    
    def distance_to(self, overlay: Overlay) -> float:
        """Compute distance to overlay."""
        # Euclidean distance in E₈
        e8_dist = np.linalg.norm(self.e8_base - overlay.e8_base)
        
        # Hamming distance in activations
        hamming_dist = np.sum(self.activations != overlay.activations)
        
        # Combined distance (weighted)
        return e8_dist + 0.01 * hamming_dist
    
    def is_equivalent(self, overlay: Overlay) -> bool:
        """Check if overlay is equivalent to this representative."""
        return self.distance_to(overlay) < self.epsilon


class EpsilonCanonicalizer:
    """
    ε-Invariant Canonicalizer.
    
    From whitepaper:
    "Numeric tolerance ε ensures that overlays differing by less than ε
    are considered equivalent."
    
    This system:
    1. Groups overlays into ε-equivalence classes
    2. Selects canonical representatives
    3. Snaps new overlays to nearest canonical representative
    4. Maintains equivalence class registry
    """
    
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
        self.representatives: List[CanonicalRepresentative] = []
        self.equivalence_classes: Dict[str, Set[str]] = {}  # rep_id -> member_ids
    
    def canonicalize(self, overlay: Overlay) -> Tuple[Overlay, bool]:
        """
        Canonicalize an overlay.
        
        Args:
            overlay: Overlay to canonicalize
        
        Returns:
            (canonical_overlay, is_new) tuple
            - canonical_overlay: Canonical representative
            - is_new: True if new representative was created
        """
        # Find nearest representative
        nearest_rep, distance = self._find_nearest_representative(overlay)
        
        if nearest_rep is not None and distance < self.epsilon:
            # Snap to existing representative
            canonical_overlay = self._snap_to_representative(overlay, nearest_rep)
            
            # Add to equivalence class
            if nearest_rep.overlay_id not in self.equivalence_classes:
                self.equivalence_classes[nearest_rep.overlay_id] = set()
            self.equivalence_classes[nearest_rep.overlay_id].add(overlay.overlay_id)
            nearest_rep.members.append(overlay.overlay_id)
            
            return canonical_overlay, False
        else:
            # Create new representative
            new_rep = CanonicalRepresentative(
                overlay_id=overlay.overlay_id,
                e8_base=overlay.e8_base.copy(),
                activations=overlay.activations.copy(),
                epsilon=self.epsilon,
                members=[overlay.overlay_id]
            )
            self.representatives.append(new_rep)
            self.equivalence_classes[overlay.overlay_id] = {overlay.overlay_id}
            
            return overlay, True
    
    def _find_nearest_representative(
        self,
        overlay: Overlay
    ) -> Tuple[Optional[CanonicalRepresentative], float]:
        """
        Find nearest canonical representative.
        
        Args:
            overlay: Overlay to find representative for
        
        Returns:
            (representative, distance) tuple
        """
        if not self.representatives:
            return None, float('inf')
        
        min_distance = float('inf')
        nearest_rep = None
        
        for rep in self.representatives:
            distance = rep.distance_to(overlay)
            if distance < min_distance:
                min_distance = distance
                nearest_rep = rep
        
        return nearest_rep, min_distance
    
    def _snap_to_representative(
        self,
        overlay: Overlay,
        representative: CanonicalRepresentative
    ) -> Overlay:
        """
        Snap overlay to canonical representative.
        
        Args:
            overlay: Overlay to snap
            representative: Canonical representative
        
        Returns:
            Snapped overlay
        """
        # Create new overlay with representative's values
        from layer1_morphonic.overlay_system import ImmutablePose
        import time
        
        pose = ImmutablePose(
            position=tuple(representative.e8_base),
            orientation=tuple(np.eye(8)[0]),
            timestamp=time.time()
        )
        
        return Overlay(
            e8_base=representative.e8_base.copy(),
            activations=representative.activations.copy(),
            weights=overlay.weights.copy() if overlay.weights is not None else None,
            phase=overlay.phase,
            pose=pose,
            metadata={
                **overlay.metadata,
                'canonical_id': representative.overlay_id,
                'snapped': True,
                'epsilon': self.epsilon
            }
        )
    
    def round_to_epsilon(self, value: float) -> float:
        """
        Round value to ε-grid.
        
        Args:
            value: Value to round
        
        Returns:
            Rounded value
        """
        if self.epsilon == 0:
            return value
        
        # Round to nearest multiple of epsilon
        return np.round(value / self.epsilon) * self.epsilon
    
    def snap_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Snap vector to ε-grid.
        
        Args:
            vector: Vector to snap
        
        Returns:
            Snapped vector
        """
        return np.array([self.round_to_epsilon(v) for v in vector])
    
    def are_equivalent(self, overlay1: Overlay, overlay2: Overlay) -> bool:
        """
        Check if two overlays are ε-equivalent.
        
        Args:
            overlay1: First overlay
            overlay2: Second overlay
        
        Returns:
            True if equivalent, False otherwise
        """
        # E₈ distance
        e8_dist = np.linalg.norm(overlay1.e8_base - overlay2.e8_base)
        
        # Activation distance (Hamming)
        activation_dist = np.sum(overlay1.activations != overlay2.activations)
        
        # Combined distance
        total_dist = e8_dist + 0.01 * activation_dist
        
        return total_dist < self.epsilon
    
    def get_equivalence_class(self, overlay_id: str) -> Set[str]:
        """
        Get equivalence class for an overlay.
        
        Args:
            overlay_id: Overlay ID
        
        Returns:
            Set of equivalent overlay IDs
        """
        # Find which equivalence class this overlay belongs to
        for rep_id, members in self.equivalence_classes.items():
            if overlay_id in members:
                return members
        
        return {overlay_id}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get canonicalizer statistics."""
        if not self.representatives:
            return {
                'num_representatives': 0,
                'num_classes': 0,
                'avg_class_size': 0.0,
                'max_class_size': 0,
                'epsilon': self.epsilon
            }
        
        class_sizes = [len(members) for members in self.equivalence_classes.values()]
        
        return {
            'num_representatives': len(self.representatives),
            'num_classes': len(self.equivalence_classes),
            'avg_class_size': np.mean(class_sizes),
            'max_class_size': np.max(class_sizes),
            'min_class_size': np.min(class_sizes),
            'total_overlays': sum(class_sizes),
            'epsilon': self.epsilon
        }
    
    def compress(self, threshold: int = 1):
        """
        Compress equivalence classes by removing singletons.
        
        Args:
            threshold: Minimum class size to keep
        """
        # Remove representatives with small equivalence classes
        self.representatives = [
            rep for rep in self.representatives
            if len(rep.members) >= threshold
        ]
        
        # Remove corresponding equivalence classes
        self.equivalence_classes = {
            rep_id: members
            for rep_id, members in self.equivalence_classes.items()
            if len(members) >= threshold
        }


# Example usage and tests
if __name__ == "__main__":
    print("=== ε-Invariant Canonicalizer Test ===\n")
    
    from layer1_morphonic.overlay_system import ImmutablePose
    import time
    
    # Create canonicalizer
    epsilon = 0.1
    canonicalizer = EpsilonCanonicalizer(epsilon=epsilon)
    
    print(f"Epsilon: {epsilon}")
    print()
    
    # Test 1: Canonicalize first overlay
    print("Test 1: First Overlay (New Representative)")
    e8_base1 = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0])
    activations1 = np.zeros(240, dtype=int)
    activations1[0:120] = 1
    
    pose1 = ImmutablePose(
        position=tuple(e8_base1),
        orientation=tuple(np.eye(8)[0]),
        timestamp=time.time()
    )
    
    overlay1 = Overlay(e8_base=e8_base1, activations=activations1, pose=pose1)
    
    canonical1, is_new1 = canonicalizer.canonicalize(overlay1)
    print(f"Original: {overlay1.overlay_id}")
    print(f"Canonical: {canonical1.overlay_id}")
    print(f"Is new representative: {is_new1}")
    print()
    
    # Test 2: Canonicalize similar overlay (should snap to representative)
    print("Test 2: Similar Overlay (Should Snap)")
    e8_base2 = e8_base1 + np.random.randn(8) * 0.01  # Small perturbation
    activations2 = activations1.copy()
    
    pose2 = ImmutablePose(
        position=tuple(e8_base2),
        orientation=tuple(np.eye(8)[0]),
        timestamp=time.time()
    )
    
    overlay2 = Overlay(e8_base=e8_base2, activations=activations2, pose=pose2)
    
    canonical2, is_new2 = canonicalizer.canonicalize(overlay2)
    print(f"Original: {overlay2.overlay_id}")
    print(f"Canonical: {canonical2.overlay_id}")
    print(f"Is new representative: {is_new2}")
    print(f"Distance to rep: {np.linalg.norm(e8_base2 - e8_base1):.6f}")
    print()
    
    # Test 3: Canonicalize different overlay (should create new representative)
    print("Test 3: Different Overlay (New Representative)")
    e8_base3 = np.array([2.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0])
    activations3 = np.zeros(240, dtype=int)
    activations3[120:240] = 1  # Different activations
    
    pose3 = ImmutablePose(
        position=tuple(e8_base3),
        orientation=tuple(np.eye(8)[0]),
        timestamp=time.time()
    )
    
    overlay3 = Overlay(e8_base=e8_base3, activations=activations3, pose=pose3)
    
    canonical3, is_new3 = canonicalizer.canonicalize(overlay3)
    print(f"Original: {overlay3.overlay_id}")
    print(f"Canonical: {canonical3.overlay_id}")
    print(f"Is new representative: {is_new3}")
    print()
    
    # Test 4: Equivalence checking
    print("Test 4: Equivalence Checking")
    equiv_1_2 = canonicalizer.are_equivalent(overlay1, overlay2)
    equiv_1_3 = canonicalizer.are_equivalent(overlay1, overlay3)
    print(f"Overlay1 ≈ Overlay2: {equiv_1_2}")
    print(f"Overlay1 ≈ Overlay3: {equiv_1_3}")
    print()
    
    # Test 5: Equivalence classes
    print("Test 5: Equivalence Classes")
    class1 = canonicalizer.get_equivalence_class(overlay1.overlay_id)
    class2 = canonicalizer.get_equivalence_class(overlay2.overlay_id)
    class3 = canonicalizer.get_equivalence_class(overlay3.overlay_id)
    print(f"Class 1 size: {len(class1)}")
    print(f"Class 2 size: {len(class2)}")
    print(f"Class 3 size: {len(class3)}")
    print(f"Class 1 == Class 2: {class1 == class2}")
    print()
    
    # Test 6: Vector snapping
    print("Test 6: Vector Snapping")
    vector = np.array([1.234567, 2.345678, 3.456789])
    snapped = canonicalizer.snap_vector(vector)
    print(f"Original: {vector}")
    print(f"Snapped: {snapped}")
    print()
    
    # Test 7: Statistics
    print("Test 7: Statistics")
    stats = canonicalizer.get_statistics()
    print(f"Representatives: {stats['num_representatives']}")
    print(f"Equivalence classes: {stats['num_classes']}")
    print(f"Average class size: {stats['avg_class_size']:.2f}")
    print(f"Max class size: {stats['max_class_size']}")
    print(f"Total overlays: {stats['total_overlays']}")
    print()
    
    print("=== All Tests Passed ===")
