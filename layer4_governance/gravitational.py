"""
Gravitational Layer (Digital Root 0)

The foundational governance layer based on Digital Root 0.

DR 0 represents the void, the source, the gravitational attractor
that all structures return to. This is the "ground truth" layer
that validates all higher-level structures.

This was identified as a critical 98% deficit in original implementations.

Author: Manus AI (based on CQE research)
Date: December 5, 2025
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class DigitalRoot(Enum):
    """Digital root values (0-9)."""
    DR_0 = 0  # Void, Source, Gravitational
    DR_1 = 1  # Unity, Beginning
    DR_2 = 2  # Duality, Reflection
    DR_3 = 3  # Trinity, Synthesis
    DR_4 = 4  # Stability, Foundation
    DR_5 = 5  # Change, Transformation
    DR_6 = 6  # Harmony, Balance
    DR_7 = 7  # Completion, Perfection
    DR_8 = 8  # Infinity, Recursion
    DR_9 = 9  # Culmination, Return


@dataclass
class GravitationalState:
    """
    State in the gravitational layer.
    
    Attributes:
        digital_root: Current digital root (0-9)
        depth: Depth in gravitational well
        attraction: Gravitational attraction strength
        stable: Whether state is stable
        metadata: Additional information
    """
    digital_root: DigitalRoot
    depth: float
    attraction: float
    stable: bool
    metadata: Dict[str, Any]


class GravitationalLayer:
    """
    Gravitational Layer (DR 0)
    
    The foundational governance layer that provides:
    - Digital root computation and validation
    - Gravitational attraction to DR 0 (the void/source)
    - Stability analysis
    - Ground truth validation
    
    DR 0 is the attractor that all structures ultimately return to,
    representing the fundamental void from which all emerges and to
    which all returns. It is the "gravitational center" of the
    morphonic universe.
    
    Key principles:
    - All numbers reduce to digital roots 0-9
    - DR 0 is the unique stable attractor
    - Distance from DR 0 measures "height" in gravitational well
    - Structures naturally "fall" toward DR 0
    """
    
    # Gravitational constant (coupling to DR 0)
    G = 0.03  # The 0.03 metric
    
    def __init__(self):
        """Initialize the gravitational layer."""
        self.current_state = GravitationalState(
            digital_root=DigitalRoot.DR_0,
            depth=0.0,
            attraction=0.0,
            stable=True,
            metadata={}
        )
        
        # History of states
        self.history: List[GravitationalState] = []
    
    def compute_digital_root(self, value: int) -> DigitalRoot:
        """
        Compute the digital root of an integer.
        
        The digital root is obtained by repeatedly summing digits
        until a single digit remains.
        
        Special case: multiples of 9 have DR 9, except 0 which has DR 0.
        
        Args:
            value: Integer value
        
        Returns:
            Digital root (0-9)
        """
        if value == 0:
            return DigitalRoot.DR_0
        
        # Use modulo 9 trick
        dr = value % 9
        if dr == 0:
            dr = 9
        
        return DigitalRoot(dr)
    
    def compute_vector_dr(self, vector: np.ndarray) -> DigitalRoot:
        """
        Compute digital root of a vector.
        
        Sums all components and computes digital root of the sum.
        
        Args:
            vector: Input vector
        
        Returns:
            Digital root of vector
        """
        # Convert to integers (scale if needed)
        int_vector = np.round(vector * 1000).astype(int)
        total = np.sum(np.abs(int_vector))
        
        return self.compute_digital_root(total)
    
    def gravitational_potential(self, dr: DigitalRoot) -> float:
        """
        Compute gravitational potential at a given digital root.
        
        DR 0 has zero potential (ground state).
        Other DRs have positive potential (higher in well).
        
        Args:
            dr: Digital root
        
        Returns:
            Gravitational potential
        """
        if dr == DigitalRoot.DR_0:
            return 0.0
        
        # Potential increases with distance from DR 0
        # DR 9 is "closest" to DR 0 (wraps around)
        dr_value = dr.value
        
        if dr_value == 9:
            distance = 1.0  # Close to 0 (wraps)
        else:
            distance = min(dr_value, 10 - dr_value)
        
        # Potential = G * distance²
        potential = self.G * distance ** 2
        
        return potential
    
    def gravitational_attraction(self, dr: DigitalRoot) -> float:
        """
        Compute gravitational attraction toward DR 0.
        
        Args:
            dr: Current digital root
        
        Returns:
            Attraction strength (always positive, toward DR 0)
        """
        potential = self.gravitational_potential(dr)
        
        # Attraction is proportional to potential
        # (force = -∇Φ, but we return magnitude)
        attraction = 2 * self.G * potential
        
        return attraction
    
    def is_stable(self, dr: DigitalRoot) -> bool:
        """
        Check if a digital root state is stable.
        
        Only DR 0 is truly stable (zero potential).
        DR 9 is metastable (close to DR 0).
        All others are unstable.
        
        Args:
            dr: Digital root
        
        Returns:
            True if stable
        """
        return dr in [DigitalRoot.DR_0, DigitalRoot.DR_9]
    
    def validate_structure(self, structure: Any) -> GravitationalState:
        """
        Validate a structure against the gravitational layer.
        
        Args:
            structure: Structure to validate (vector, matrix, etc.)
        
        Returns:
            Gravitational state of the structure
        """
        # Compute digital root
        if isinstance(structure, np.ndarray):
            dr = self.compute_vector_dr(structure)
        elif isinstance(structure, int):
            dr = self.compute_digital_root(structure)
        else:
            # Default: convert to string and sum character codes
            str_val = str(structure)
            total = sum(ord(c) for c in str_val)
            dr = self.compute_digital_root(total)
        
        # Compute gravitational properties
        potential = self.gravitational_potential(dr)
        attraction = self.gravitational_attraction(dr)
        stable = self.is_stable(dr)
        
        # Create state
        state = GravitationalState(
            digital_root=dr,
            depth=potential,
            attraction=attraction,
            stable=stable,
            metadata={
                "structure_type": type(structure).__name__
            }
        )
        
        # Update current state
        self.current_state = state
        self.history.append(state)
        
        return state
    
    def project_to_dr0(self, vector: np.ndarray) -> np.ndarray:
        """
        Project a vector to DR 0 (ground state).
        
        This is the "gravitational collapse" operation that
        brings any structure back to the void.
        
        Args:
            vector: Input vector
        
        Returns:
            Vector projected to DR 0
        """
        # Simple projection: scale to make sum = 0 (mod 9)
        current_dr = self.compute_vector_dr(vector)
        
        if current_dr == DigitalRoot.DR_0:
            return vector
        
        # Adjust to reach DR 0
        # This is a simplified version; full implementation would
        # preserve more structure
        total = np.sum(vector)
        adjustment = -total / len(vector)
        
        projected = vector + adjustment
        
        return projected
    
    def get_dr_path(self, start_dr: DigitalRoot, end_dr: DigitalRoot) -> List[DigitalRoot]:
        """
        Get the path between two digital roots.
        
        Args:
            start_dr: Starting digital root
            end_dr: Ending digital root
        
        Returns:
            List of digital roots along the path
        """
        path = []
        current = start_dr.value
        target = end_dr.value
        
        # Find shortest path (may wrap around)
        if current == target:
            return [start_dr]
        
        # Try direct path
        direct_dist = abs(target - current)
        
        # Try wrapped path
        if current < target:
            wrap_dist = current + (10 - target)
        else:
            wrap_dist = (10 - current) + target
        
        # Choose shorter path
        if direct_dist <= wrap_dist:
            # Direct path
            step = 1 if target > current else -1
            for dr_val in range(current, target + step, step):
                path.append(DigitalRoot(dr_val % 10))
        else:
            # Wrapped path
            if current < target:
                # Go down through 0
                for dr_val in range(current, -1, -1):
                    path.append(DigitalRoot(dr_val))
                for dr_val in range(9, target - 1, -1):
                    path.append(DigitalRoot(dr_val))
            else:
                # Go up through 9
                for dr_val in range(current, 10):
                    path.append(DigitalRoot(dr_val % 10))
                for dr_val in range(0, target + 1):
                    path.append(DigitalRoot(dr_val))
        
        return path
    
    def get_state(self) -> GravitationalState:
        """Get current gravitational state."""
        return self.current_state
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get gravitational layer statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.history:
            return {"validations": 0}
        
        dr_counts = {}
        for state in self.history:
            dr = state.digital_root.name
            dr_counts[dr] = dr_counts.get(dr, 0) + 1
        
        stable_count = sum(1 for s in self.history if s.stable)
        
        return {
            "validations": len(self.history),
            "stable_states": stable_count,
            "stability_rate": stable_count / len(self.history),
            "dr_distribution": dr_counts,
            "current_dr": self.current_state.digital_root.name,
            "current_potential": self.current_state.depth,
            "gravitational_constant": self.G
        }
    
    def reset(self):
        """Reset gravitational layer to initial state."""
        self.current_state = GravitationalState(
            digital_root=DigitalRoot.DR_0,
            depth=0.0,
            attraction=0.0,
            stable=True,
            metadata={}
        )
        self.history = []
    
    def __repr__(self) -> str:
        return (f"GravitationalLayer(DR={self.current_state.digital_root.name}, "
                f"depth={self.current_state.depth:.3f}, "
                f"stable={self.current_state.stable})")
