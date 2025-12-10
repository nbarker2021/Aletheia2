"""
Conservation Law Enforcer

Implements the fundamental CQE conservation law: ΔΦ ≤ 0

This is the core principle that ensures all transformations in the
CQE system are non-increasing in potential energy.

Author: Manus AI (based on CQE research)
Date: December 5, 2025
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class ConservationResult:
    """
    Result of a conservation law check.
    
    Attributes:
        valid: Whether the transformation satisfies ΔΦ ≤ 0
        delta_phi: Change in potential energy
        phi_initial: Initial potential
        phi_final: Final potential
        metadata: Additional information
    """
    valid: bool
    delta_phi: float
    phi_initial: float
    phi_final: float
    metadata: dict


class ConservationEnforcer:
    """
    Conservation Law Enforcer
    
    Ensures all transformations satisfy the fundamental conservation law:
    ΔΦ ≤ 0 (non-increasing potential energy)
    
    This is the "second law of morphonics" - the system naturally
    evolves toward lower potential energy states.
    """
    
    # Golden ratio (φ) - fundamental constant
    PHI = (1 + np.sqrt(5)) / 2
    
    # Coupling constant (0.03 metric)
    COUPLING = np.log(PHI) / 16
    
    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize the conservation enforcer.
        
        Args:
            tolerance: Numerical tolerance for ΔΦ ≤ 0 check
        """
        self.tolerance = tolerance
        self.violation_count = 0
        self.enforcement_count = 0
    
    def compute_potential(self, state: np.ndarray) -> float:
        """
        Compute the potential energy Φ of a state.
        
        The potential is computed as:
        Φ = ||state||² + coupling * (state · golden_vector)
        
        Args:
            state: State vector
        
        Returns:
            Potential energy Φ
        """
        # Quadratic term (kinetic-like)
        quadratic = np.dot(state, state)
        
        # Golden ratio coupling term
        golden_vector = np.array([self.PHI ** i for i in range(len(state))])
        golden_vector /= np.linalg.norm(golden_vector)
        coupling_term = self.COUPLING * np.dot(state, golden_vector)
        
        # Total potential
        phi = quadratic + coupling_term
        
        return phi
    
    def check_transformation(self, 
                            initial_state: np.ndarray,
                            final_state: np.ndarray) -> ConservationResult:
        """
        Check if a transformation satisfies ΔΦ ≤ 0.
        
        Args:
            initial_state: State before transformation
            final_state: State after transformation
        
        Returns:
            ConservationResult indicating validity
        """
        # Compute potentials
        phi_initial = self.compute_potential(initial_state)
        phi_final = self.compute_potential(final_state)
        
        # Compute change
        delta_phi = phi_final - phi_initial
        
        # Check conservation law
        valid = delta_phi <= self.tolerance
        
        # Update statistics
        self.enforcement_count += 1
        if not valid:
            self.violation_count += 1
        
        return ConservationResult(
            valid=valid,
            delta_phi=delta_phi,
            phi_initial=phi_initial,
            phi_final=phi_final,
            metadata={
                "tolerance": self.tolerance,
                "enforcement_count": self.enforcement_count,
                "violation_count": self.violation_count
            }
        )
    
    def enforce_transformation(self,
                              initial_state: np.ndarray,
                              transformation: Callable[[np.ndarray], np.ndarray],
                              max_attempts: int = 10) -> Tuple[np.ndarray, ConservationResult]:
        """
        Apply a transformation and enforce conservation law.
        
        If the transformation violates ΔΦ ≤ 0, attempt to correct it
        by scaling or projecting to a valid state.
        
        Args:
            initial_state: Initial state
            transformation: Transformation function
            max_attempts: Maximum correction attempts
        
        Returns:
            Tuple of (final_state, conservation_result)
        """
        # Apply transformation
        final_state = transformation(initial_state)
        
        # Check conservation
        result = self.check_transformation(initial_state, final_state)
        
        if result.valid:
            return final_state, result
        
        # Attempt correction
        for attempt in range(max_attempts):
            # Scale down the transformation
            scale = 0.9 ** (attempt + 1)
            corrected_state = initial_state + scale * (final_state - initial_state)
            
            # Check corrected transformation
            result = self.check_transformation(initial_state, corrected_state)
            
            if result.valid:
                result.metadata["corrected"] = True
                result.metadata["correction_attempts"] = attempt + 1
                return corrected_state, result
        
        # If all corrections fail, return initial state (no transformation)
        result = self.check_transformation(initial_state, initial_state)
        result.metadata["correction_failed"] = True
        return initial_state, result
    
    def get_statistics(self) -> dict:
        """
        Get conservation enforcement statistics.
        
        Returns:
            Dictionary with statistics
        """
        violation_rate = (self.violation_count / self.enforcement_count 
                         if self.enforcement_count > 0 else 0)
        
        return {
            "enforcement_count": self.enforcement_count,
            "violation_count": self.violation_count,
            "violation_rate": violation_rate,
            "tolerance": self.tolerance,
            "coupling_constant": self.COUPLING,
            "golden_ratio": self.PHI
        }
    
    def reset_statistics(self):
        """Reset enforcement statistics."""
        self.violation_count = 0
        self.enforcement_count = 0
    
    def __repr__(self) -> str:
        return (f"ConservationEnforcer(enforcements={self.enforcement_count}, "
                f"violations={self.violation_count})")
