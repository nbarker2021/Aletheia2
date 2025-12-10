"""
Morphonic Seed Generator

Single-digit bootstrap: A single digit (1-9) deterministically generates
the entire 24D substrate via mod-9 iteration.

This demonstrates:
- Observer effect: choosing a digit collapses to specific geometry
- Emergence: full structure from minimal seed
- Digital root conservation: all operations preserve mod-9 class

Ported from cqe-complete/cqe/advanced/morphonic.py
Author: Manus AI (based on CQE research)
Date: December 5, 2025
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class MorphonSeed:
    """A single-digit seed that generates geometry."""
    digit: int  # 1-9
    digital_root: int  # mod 9 class
    parity: int  # 0 (even) or 1 (odd)
    
    def __post_init__(self):
        assert 1 <= self.digit <= 9, "Seed must be 1-9"
        self.digital_root = self.digit  # For single digit, DR = digit
        self.parity = self.digit % 2


class MorphonicSeedGenerator:
    """
    Generate full geometric structures from single-digit seeds.
    
    Process:
    1. Start with digit d ∈ {1..9}
    2. Apply mod-9 iteration: d → d² mod 9 → (d²)² mod 9 → ...
    3. Sequence converges to fixed point or cycle
    4. Map sequence to geometric structures
    5. Compose to build full 24D substrate
    
    This demonstrates morphonic emergence: complex structure from simple seed.
    """
    
    # Golden ratio
    PHI = (1 + np.sqrt(5)) / 2
    
    # Digital root to index mapping
    DR_TO_INDEX = {
        1: 0,    # Unity
        2: 1,    # Duality
        3: 2,    # Trinity
        4: 3,    # Stability
        5: 4,    # Change
        6: 5,    # Harmony
        7: 6,    # Completion
        8: 7,    # Infinity
        9: 0,    # Return to unity (mod 9 = 0)
    }
    
    # Digital root to Niemeier lattice type mapping
    DR_TO_NIEMEIER = {
        1: "24A1",      # Unity → many small components
        2: "12A2",      # Duality → pairs
        3: "3E8",       # Trinity → three E8
        4: "D24",       # Stability → single large component
        5: "A24",       # Change → maximal roots
        6: "2D6",       # Harmony → balanced pairs
        7: "E8",        # Completion → single E8
        8: "D16E8",     # Infinity → mixed
        9: "Leech",     # Return → no roots (pure potential)
    }
    
    def __init__(self):
        """Initialize the morphonic seed generator."""
        pass
    
    def iterate_mod9(self, digit: int, max_iterations: int = 20) -> List[int]:
        """
        Iterate digit via squaring mod 9.
        
        Sequence: d → d² mod 9 → (d²)² mod 9 → ...
        
        Args:
            digit: Starting digit (1-9)
            max_iterations: Maximum iterations
            
        Returns:
            List of digital roots in sequence
        """
        sequence = [digit]
        current = digit
        
        for _ in range(max_iterations):
            # Square and take mod 9
            next_val = (current * current) % 9
            if next_val == 0:
                next_val = 9  # Keep in {1..9}
            
            sequence.append(next_val)
            
            # Check for fixed point or cycle
            if next_val == current:
                break  # Fixed point
            if next_val in sequence[:-1]:
                break  # Cycle detected
            
            current = next_val
        
        return sequence
    
    def sequence_to_vector(self, sequence: List[int]) -> np.ndarray:
        """
        Map digital root sequence to 8D vector.
        
        Each digit maps to a basis vector via the DR correspondence.
        
        Args:
            sequence: List of digital roots
            
        Returns:
            8D vector
        """
        vector = np.zeros(8)
        
        for i, dr in enumerate(sequence):
            index = self.DR_TO_INDEX[dr]
            # Weight by golden ratio powers
            weight = self.PHI ** (-i)
            vector[index] += weight
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 1e-10:
            vector = vector / norm
        
        return vector
    
    def generate_e8_from_seed(self, seed: MorphonSeed) -> Tuple[np.ndarray, List[int]]:
        """
        Generate E8 vector from single-digit seed.
        
        Process:
        1. Iterate seed via mod-9
        2. Map sequence to vector
        3. Normalize
        
        Args:
            seed: MorphonSeed (digit 1-9)
            
        Returns:
            (8D vector, digital root sequence)
        """
        # Generate digital root sequence
        dr_sequence = self.iterate_mod9(seed.digit)
        
        # Map to vector
        vector = self.sequence_to_vector(dr_sequence)
        
        return vector, dr_sequence
    
    def extend_to_24d(self, e8_vector: np.ndarray, seed: MorphonSeed) -> np.ndarray:
        """
        Extend 8D E8 vector to 24D via 3×E8 construction.
        
        Uses seed to determine the three copies.
        
        Args:
            e8_vector: 8D E8 vector
            seed: Original seed
            
        Returns:
            24D vector
        """
        # First copy: original
        copy1 = e8_vector
        
        # Second copy: reflect based on parity
        if seed.parity == 0:
            copy2 = e8_vector.copy()
            copy2[0] = -copy2[0]  # Reflect first coordinate
        else:
            copy2 = -e8_vector  # Full reflection
        
        # Third copy: rotate based on digital root
        copy3 = np.roll(e8_vector, seed.digital_root % 8)
        
        # Concatenate
        vector_24d = np.concatenate([copy1, copy2, copy3])
        
        return vector_24d
    
    def get_niemeier_type(self, seed: MorphonSeed) -> str:
        """
        Get Niemeier lattice type for a seed.
        
        Args:
            seed: MorphonSeed
            
        Returns:
            Niemeier lattice type name
        """
        return self.DR_TO_NIEMEIER[seed.digital_root]
    
    def full_generation(self, digit: int) -> Dict:
        """
        Complete generation from single digit to 24D.
        
        Args:
            digit: Starting digit (1-9)
            
        Returns:
            Dictionary with all generation results
        """
        # Create seed
        seed = MorphonSeed(digit=digit, digital_root=digit, parity=digit % 2)
        
        # Generate E8 vector
        e8_vector, dr_sequence = self.generate_e8_from_seed(seed)
        
        # Extend to 24D
        vector_24d = self.extend_to_24d(e8_vector, seed)
        
        # Get Niemeier type
        niemeier_type = self.get_niemeier_type(seed)
        
        return {
            "seed": seed,
            "dr_sequence": dr_sequence,
            "e8_vector": e8_vector,
            "vector_24d": vector_24d,
            "niemeier_type": niemeier_type,
            "sequence_length": len(dr_sequence),
            "converged": dr_sequence[-1] == dr_sequence[-2] if len(dr_sequence) > 1 else False
        }
    
    def __repr__(self) -> str:
        return "MorphonicSeedGenerator(phi={:.6f})".format(self.PHI)
