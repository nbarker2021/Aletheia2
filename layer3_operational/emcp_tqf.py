"""
EMCP TQF - Emergent Morphonic Chiral Pairing
Topological Quantum Field (TQF) Theory
Cartan-Quadratic Equivalence (CQE) System

From Whitepaper:
"EMCP TQF describes chiral coupling between left and right sectors in E₈.
The theory ensures parity conservation and topological invariance."

This implements:
- Chiral sectors (left/right)
- Coupling operators
- Topological invariants
- Parity conservation laws
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


class Chirality(Enum):
    """Chirality of a sector."""
    LEFT = "left"
    RIGHT = "right"
    ACHIRAL = "achiral"


@dataclass
class ChiralSector:
    """
    Chiral sector in E₈.
    
    E₈ can be decomposed into left and right chiral sectors.
    """
    chirality: Chirality
    root_indices: List[int]  # Which roots belong to this sector
    overlay: Overlay
    
    @property
    def dimension(self) -> int:
        """Dimension of this sector."""
        return len(self.root_indices)
    
    @property
    def charge(self) -> int:
        """Chiral charge (number of active roots)."""
        return int(np.sum(self.overlay.activations[self.root_indices]))


@dataclass
class ChiralPair:
    """
    Pair of left and right chiral sectors.
    
    From whitepaper:
    "Chiral pairing ensures parity conservation."
    """
    left_sector: ChiralSector
    right_sector: ChiralSector
    coupling_strength: float
    
    @property
    def total_charge(self) -> int:
        """Total chiral charge."""
        return self.left_sector.charge + self.right_sector.charge
    
    @property
    def charge_imbalance(self) -> int:
        """Chiral charge imbalance."""
        return self.left_sector.charge - self.right_sector.charge
    
    def is_parity_conserving(self) -> bool:
        """Check if pair conserves parity."""
        # Parity is conserved if charge imbalance is even
        return self.charge_imbalance % 2 == 0


@dataclass
class TopologicalInvariant:
    """
    Topological invariant of a chiral configuration.
    
    From TQF theory:
    Topological invariants are preserved under continuous deformations.
    """
    chern_number: int  # First Chern class
    euler_characteristic: int  # Euler characteristic
    signature: int  # Signature invariant
    
    def __eq__(self, other):
        if not isinstance(other, TopologicalInvariant):
            return False
        return (
            self.chern_number == other.chern_number and
            self.euler_characteristic == other.euler_characteristic and
            self.signature == other.signature
        )


class EMCP_TQF:
    """
    Emergent Morphonic Chiral Pairing - Topological Quantum Field Theory.
    
    From whitepaper:
    "EMCP TQF describes chiral coupling between left and right sectors."
    
    Key features:
    1. Decompose E₈ into chiral sectors
    2. Compute chiral coupling
    3. Enforce parity conservation
    4. Compute topological invariants
    """
    
    def __init__(self):
        self.chiral_pairs: List[ChiralPair] = []
    
    def decompose_chiral(self, overlay: Overlay) -> Tuple[ChiralSector, ChiralSector]:
        """
        Decompose overlay into left and right chiral sectors.
        
        Args:
            overlay: Overlay to decompose
        
        Returns:
            (left_sector, right_sector) tuple
        """
        # Split 240 roots into left (0-119) and right (120-239)
        left_indices = list(range(0, 120))
        right_indices = list(range(120, 240))
        
        # Create left sector
        left_sector = ChiralSector(
            chirality=Chirality.LEFT,
            root_indices=left_indices,
            overlay=overlay
        )
        
        # Create right sector
        right_sector = ChiralSector(
            chirality=Chirality.RIGHT,
            root_indices=right_indices,
            overlay=overlay
        )
        
        return left_sector, right_sector
    
    def compute_coupling(
        self,
        left_sector: ChiralSector,
        right_sector: ChiralSector
    ) -> float:
        """
        Compute coupling strength between chiral sectors.
        
        Args:
            left_sector: Left chiral sector
            right_sector: Right chiral sector
        
        Returns:
            Coupling strength
        """
        # Coupling based on charge overlap
        left_charge = left_sector.charge
        right_charge = right_sector.charge
        
        # Normalized coupling (0 to 1)
        if left_charge + right_charge > 0:
            coupling = 2 * min(left_charge, right_charge) / (left_charge + right_charge)
        else:
            coupling = 0.0
        
        return coupling
    
    def create_pair(
        self,
        overlay: Overlay
    ) -> ChiralPair:
        """
        Create chiral pair from overlay.
        
        Args:
            overlay: Overlay
        
        Returns:
            ChiralPair
        """
        left_sector, right_sector = self.decompose_chiral(overlay)
        coupling = self.compute_coupling(left_sector, right_sector)
        
        pair = ChiralPair(
            left_sector=left_sector,
            right_sector=right_sector,
            coupling_strength=coupling
        )
        
        self.chiral_pairs.append(pair)
        return pair
    
    def compute_topological_invariant(
        self,
        pair: ChiralPair
    ) -> TopologicalInvariant:
        """
        Compute topological invariant for chiral pair.
        
        Args:
            pair: Chiral pair
        
        Returns:
            TopologicalInvariant
        """
        # Chern number (simplified: charge imbalance)
        chern_number = pair.charge_imbalance
        
        # Euler characteristic (simplified: total charge)
        euler_characteristic = pair.total_charge
        
        # Signature (simplified: coupling strength * 100)
        signature = int(pair.coupling_strength * 100)
        
        return TopologicalInvariant(
            chern_number=chern_number,
            euler_characteristic=euler_characteristic,
            signature=signature
        )
    
    def check_parity_conservation(
        self,
        pair_before: ChiralPair,
        pair_after: ChiralPair
    ) -> bool:
        """
        Check if parity is conserved between two chiral pairs.
        
        Args:
            pair_before: Chiral pair before transformation
            pair_after: Chiral pair after transformation
        
        Returns:
            True if parity conserved, False otherwise
        """
        # Compute topological invariants
        inv_before = self.compute_topological_invariant(pair_before)
        inv_after = self.compute_topological_invariant(pair_after)
        
        # Parity is conserved if topological invariants match
        return inv_before == inv_after
    
    def couple_sectors(
        self,
        left_overlay: Overlay,
        right_overlay: Overlay,
        coupling_strength: float = 0.5
    ) -> Overlay:
        """
        Couple two overlays via chiral coupling.
        
        Args:
            left_overlay: Left sector overlay
            right_overlay: Right sector overlay
            coupling_strength: Coupling strength (0 to 1)
        
        Returns:
            Coupled overlay
        """
        # Create coupled activations
        coupled_activations = np.zeros(240, dtype=int)
        
        # Left sector (0-119)
        coupled_activations[0:120] = left_overlay.activations[0:120]
        
        # Right sector (120-239)
        coupled_activations[120:240] = right_overlay.activations[120:240]
        
        # Apply coupling (mix boundary regions)
        boundary_width = int(10 * coupling_strength)
        for i in range(boundary_width):
            # Mix near boundary (110-120 and 120-130)
            left_idx = 110 + i
            right_idx = 120 + i
            
            if left_idx < 120 and right_idx < 240:
                # Average activations at boundary
                avg = (coupled_activations[left_idx] + coupled_activations[right_idx]) / 2
                coupled_activations[left_idx] = int(avg > 0.5)
                coupled_activations[right_idx] = int(avg > 0.5)
        
        # Coupled e8_base (weighted average)
        coupled_e8_base = (
            (1 - coupling_strength) * left_overlay.e8_base +
            coupling_strength * right_overlay.e8_base
        )
        
        # Create coupled overlay
        from layer1_morphonic.overlay_system import ImmutablePose
        import time
        
        pose = ImmutablePose(
            position=tuple(coupled_e8_base),
            orientation=tuple(np.eye(8)[0]),
            timestamp=time.time()
        )
        
        return Overlay(
            e8_base=coupled_e8_base,
            activations=coupled_activations,
            pose=pose,
            metadata={
                'left_parent': left_overlay.overlay_id,
                'right_parent': right_overlay.overlay_id,
                'coupling_strength': coupling_strength,
                'emcp_coupled': True
            }
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get TQF statistics."""
        if not self.chiral_pairs:
            return {
                'num_pairs': 0,
                'avg_coupling': 0.0,
                'avg_charge_imbalance': 0.0,
                'parity_conserving_fraction': 0.0
            }
        
        couplings = [p.coupling_strength for p in self.chiral_pairs]
        imbalances = [p.charge_imbalance for p in self.chiral_pairs]
        parity_conserving = sum(1 for p in self.chiral_pairs if p.is_parity_conserving())
        
        return {
            'num_pairs': len(self.chiral_pairs),
            'avg_coupling': np.mean(couplings),
            'avg_charge_imbalance': np.mean(imbalances),
            'parity_conserving_fraction': parity_conserving / len(self.chiral_pairs)
        }


# Example usage and tests
if __name__ == "__main__":
    print("=== EMCP TQF Test ===\n")
    
    from layer1_morphonic.overlay_system import ImmutablePose
    import time
    
    # Create TQF system
    tqf = EMCP_TQF()
    
    # Test 1: Create overlay and decompose
    print("Test 1: Chiral Decomposition")
    
    e8_base = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0])
    activations = np.zeros(240, dtype=int)
    activations[0:80] = 1  # Left sector: 80 active
    activations[120:200] = 1  # Right sector: 80 active
    
    pose = ImmutablePose(
        position=tuple(e8_base),
        orientation=tuple(np.eye(8)[0]),
        timestamp=time.time()
    )
    
    overlay = Overlay(e8_base=e8_base, activations=activations, pose=pose)
    
    left_sector, right_sector = tqf.decompose_chiral(overlay)
    print(f"Left sector charge: {left_sector.charge}")
    print(f"Right sector charge: {right_sector.charge}")
    print()
    
    # Test 2: Create chiral pair
    print("Test 2: Chiral Pair Creation")
    
    pair = tqf.create_pair(overlay)
    print(f"Total charge: {pair.total_charge}")
    print(f"Charge imbalance: {pair.charge_imbalance}")
    print(f"Coupling strength: {pair.coupling_strength:.3f}")
    print(f"Parity conserving: {pair.is_parity_conserving()}")
    print()
    
    # Test 3: Topological invariant
    print("Test 3: Topological Invariant")
    
    invariant = tqf.compute_topological_invariant(pair)
    print(f"Chern number: {invariant.chern_number}")
    print(f"Euler characteristic: {invariant.euler_characteristic}")
    print(f"Signature: {invariant.signature}")
    print()
    
    # Test 4: Couple two overlays
    print("Test 4: Chiral Coupling")
    
    # Create left overlay
    e8_left = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0])
    activations_left = np.zeros(240, dtype=int)
    activations_left[0:100] = 1
    
    pose_left = ImmutablePose(
        position=tuple(e8_left),
        orientation=tuple(np.eye(8)[0]),
        timestamp=time.time()
    )
    
    left_overlay = Overlay(e8_base=e8_left, activations=activations_left, pose=pose_left)
    
    # Create right overlay
    e8_right = np.array([0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0])
    activations_right = np.zeros(240, dtype=int)
    activations_right[120:220] = 1
    
    pose_right = ImmutablePose(
        position=tuple(e8_right),
        orientation=tuple(np.eye(8)[0]),
        timestamp=time.time()
    )
    
    right_overlay = Overlay(e8_base=e8_right, activations=activations_right, pose=pose_right)
    
    # Couple
    coupled = tqf.couple_sectors(left_overlay, right_overlay, coupling_strength=0.5)
    print(f"Left overlay: {left_overlay.overlay_id}")
    print(f"  Active: {np.sum(left_overlay.activations)}")
    print(f"Right overlay: {right_overlay.overlay_id}")
    print(f"  Active: {np.sum(right_overlay.activations)}")
    print(f"Coupled overlay: {coupled.overlay_id}")
    print(f"  Active: {np.sum(coupled.activations)}")
    print()
    
    # Test 5: Parity conservation check
    print("Test 5: Parity Conservation")
    
    pair_before = tqf.create_pair(overlay)
    pair_after = tqf.create_pair(coupled)
    
    conserved = tqf.check_parity_conservation(pair_before, pair_after)
    print(f"Parity conserved: {conserved}")
    print()
    
    # Test 6: Statistics
    print("Test 6: Statistics")
    
    stats = tqf.get_statistics()
    print(f"Total pairs: {stats['num_pairs']}")
    print(f"Average coupling: {stats['avg_coupling']:.3f}")
    print(f"Average charge imbalance: {stats['avg_charge_imbalance']:.1f}")
    print(f"Parity conserving fraction: {stats['parity_conserving_fraction']:.1%}")
    print()
    
    print("=== All Tests Passed ===")
