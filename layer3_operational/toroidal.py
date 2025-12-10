"""
Toroidal Geometry and Flow

Implements toroidal closure and temporal flow via four rotation modes.
The toroidal structure ensures lossless generation through closed loops.

Four rotation modes:
- Poloidal: Around minor circle (electromagnetic, DR 1,4,7)
- Toroidal: Around major circle (weak nuclear, DR 2,5,8)
- Meridional: Along meridian (strong nuclear, DR 3,6,9)
- Helical: Spiral motion (gravitational, DR 0)

Ported from cqe-complete/cqe/advanced/toroidal.py
Author: Manus AI (based on CQE research)
Date: December 5, 2025
"""

import numpy as np
from typing import Tuple, List
from dataclasses import dataclass


# Constants
COUPLING = 0.03  # The 0.03 metric
MAJOR_RADIUS = 1.0
MINOR_RADIUS = 0.3  # 10 × coupling (0.3 = 10 × 0.03)
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio


@dataclass
class ToroidalState:
    """State on toroidal manifold."""
    poloidal_angle: float  # θ ∈ [0, 2π) - around minor circle
    toroidal_angle: float  # φ ∈ [0, 2π) - around major circle
    meridional_phase: float  # ψ ∈ [0, 2π) - along meridian
    helical_phase: float  # ω ∈ [0, 2π) - helical (gravitational)
    
    e8_embedding: np.ndarray  # (8,) E8 coordinates
    timestamp: float  # Time in seconds
    
    def to_cartesian(self, R: float = MAJOR_RADIUS, 
                    r: float = MINOR_RADIUS) -> Tuple[float, float, float]:
        """Convert to 3D Cartesian coordinates."""
        x = (R + r * np.cos(self.poloidal_angle)) * np.cos(self.toroidal_angle)
        y = (R + r * np.cos(self.poloidal_angle)) * np.sin(self.toroidal_angle)
        z = r * np.sin(self.poloidal_angle)
        return x, y, z
    
    def __repr__(self) -> str:
        return (f"ToroidalState(θ={self.poloidal_angle:.3f}, φ={self.toroidal_angle:.3f}, "
                f"t={self.timestamp:.3f})")


class ToroidalFlow:
    """
    Toroidal flow engine for temporal evolution.
    
    Implements the four fundamental rotation modes that govern
    temporal flow in the CQE framework. The toroidal structure
    ensures closure - no information leaks out.
    """
    
    def __init__(self, 
                 coupling: float = COUPLING, 
                 major_radius: float = MAJOR_RADIUS,
                 minor_radius: float = MINOR_RADIUS):
        """
        Initialize toroidal flow engine.
        
        Args:
            coupling: Coupling constant (default 0.03)
            major_radius: Major radius R (default 1.0)
            minor_radius: Minor radius r (default 0.3)
        """
        self.coupling = coupling
        self.R = major_radius
        self.r = minor_radius
        
    def _rotation_matrix_2d(self, angle: float) -> np.ndarray:
        """2D rotation matrix."""
        return np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
    
    def _rotation_matrix_e8(self, axis1: int, axis2: int, 
                           angle: float) -> np.ndarray:
        """E8 rotation matrix in specified plane."""
        R = np.eye(8)
        R2d = self._rotation_matrix_2d(angle)
        R[axis1, axis1] = R2d[0, 0]
        R[axis1, axis2] = R2d[0, 1]
        R[axis2, axis1] = R2d[1, 0]
        R[axis2, axis2] = R2d[1, 1]
        return R
    
    def rotate_poloidal(self, e8_state: np.ndarray, dt: float) -> np.ndarray:
        """
        Poloidal rotation (around minor circle).
        Maps to electromagnetic force (DR 1, 4, 7).
        """
        angle = dt * 2 * np.pi
        # Rotate in 0-1 plane
        R = self._rotation_matrix_e8(0, 1, angle)
        return R @ e8_state
    
    def rotate_toroidal(self, e8_state: np.ndarray, dt: float) -> np.ndarray:
        """
        Toroidal rotation (around major circle).
        Maps to weak nuclear force (DR 2, 5, 8).
        """
        angle = dt * 2 * np.pi
        # Rotate in 2-3 plane
        R = self._rotation_matrix_e8(2, 3, angle)
        return R @ e8_state
    
    def rotate_meridional(self, e8_state: np.ndarray, dt: float) -> np.ndarray:
        """
        Meridional rotation (along meridian).
        Maps to strong nuclear force (DR 3, 6, 9).
        """
        angle = dt * 2 * np.pi
        # Rotate in 4-5 plane
        R = self._rotation_matrix_e8(4, 5, angle)
        return R @ e8_state
    
    def rotate_helical(self, e8_state: np.ndarray, dt: float) -> np.ndarray:
        """
        Helical rotation (spiral motion).
        Maps to gravitational force (DR 0).
        This is the unifying rotation mode.
        """
        # Combine all three rotations with golden ratio weighting
        poloidal = self.rotate_poloidal(e8_state, dt / PHI)
        toroidal = self.rotate_toroidal(e8_state, dt / PHI**2)
        meridional = self.rotate_meridional(e8_state, dt / PHI**3)
        
        # Helical = weighted combination
        helical = (poloidal + toroidal + meridional) / 3
        
        # Normalize
        norm = np.linalg.norm(helical)
        if norm > 1e-10:
            helical = helical / norm * np.sqrt(2)
        
        return helical
    
    def evolve_state(self, e8_state: np.ndarray, dt: float = None) -> np.ndarray:
        """
        Evolve E8 state by one timestep using all four rotation modes.
        This is the core temporal flow operation.
        
        Args:
            e8_state: Current E8 state (8D vector)
            dt: Time step (default: coupling constant)
        
        Returns:
            Next E8 state after evolution
        """
        if dt is None:
            dt = self.coupling
        
        # Apply all four rotation modes
        poloidal = self.rotate_poloidal(e8_state, dt)
        toroidal = self.rotate_toroidal(e8_state, dt)
        meridional = self.rotate_meridional(e8_state, dt)
        helical = self.rotate_helical(e8_state, dt)
        
        # Combine with equal weighting
        next_state = (
            poloidal * 0.25 +
            toroidal * 0.25 +
            meridional * 0.25 +
            helical * 0.25
        ) * dt
        
        # Add current state (Euler integration)
        next_state = e8_state + next_state
        
        # Project to toroidal manifold
        next_state = self.project_to_torus(next_state)
        
        return next_state
    
    def project_to_torus(self, e8_state: np.ndarray) -> np.ndarray:
        """
        Project E8 state to toroidal manifold.
        Ensures closure - no information leaks out.
        
        Args:
            e8_state: E8 state to project
        
        Returns:
            Projected state on torus
        """
        # Extract toroidal coordinates from E8
        x, y = e8_state[0], e8_state[1]
        z = e8_state[2]
        
        # Compute angles
        phi = np.arctan2(y, x)  # Toroidal angle
        rho = np.sqrt(x**2 + y**2)  # Distance from z-axis
        
        # Ensure rho is in valid range
        if rho < self.R - self.r:
            rho = self.R - self.r
        elif rho > self.R + self.r:
            rho = self.R + self.r
        
        # Compute poloidal angle
        theta = np.arccos(np.clip((rho - self.R) / self.r, -1, 1))
        
        # Reconstruct on torus
        new_x = (self.R + self.r * np.cos(theta)) * np.cos(phi)
        new_y = (self.R + self.r * np.cos(theta)) * np.sin(phi)
        new_z = self.r * np.sin(theta)
        
        # Update E8 state
        projected = e8_state.copy()
        projected[0] = new_x
        projected[1] = new_y
        projected[2] = new_z
        
        # Normalize to maintain E8 norm
        norm = np.linalg.norm(projected)
        if norm > 1e-10:
            projected = projected / norm * np.sqrt(2)
        
        return projected
    
    def extract_toroidal_state(self, e8_state: np.ndarray, 
                              timestamp: float) -> ToroidalState:
        """
        Extract toroidal state from E8 embedding.
        
        Args:
            e8_state: E8 state vector
            timestamp: Current timestamp
        
        Returns:
            ToroidalState with all angles
        """
        x, y = e8_state[0], e8_state[1]
        z = e8_state[2]
        
        # Compute angles
        phi = np.arctan2(y, x)
        rho = np.sqrt(x**2 + y**2)
        theta = np.arccos(np.clip((rho - self.R) / self.r, -1, 1))
        
        # Meridional and helical phases from remaining coordinates
        psi = np.arctan2(e8_state[5], e8_state[4])
        omega = np.arctan2(e8_state[7], e8_state[6])
        
        return ToroidalState(
            poloidal_angle=theta,
            toroidal_angle=phi,
            meridional_phase=psi,
            helical_phase=omega,
            e8_embedding=e8_state,
            timestamp=timestamp
        )
    
    def compute_flow_velocity(self, e8_state: np.ndarray) -> float:
        """
        Compute flow velocity at current state.
        
        Args:
            e8_state: Current E8 state
        
        Returns:
            Flow velocity (normalized)
        """
        # Velocity is proportional to distance from center
        rho = np.sqrt(e8_state[0]**2 + e8_state[1]**2)
        velocity = (rho - self.R) / self.r  # Normalized [-1, 1]
        return velocity * self.coupling
    
    def check_closure(self, trajectory: List[np.ndarray]) -> bool:
        """
        Check if trajectory forms a closed loop (toroidal closure).
        True lossless generation requires closure.
        
        Args:
            trajectory: List of E8 states forming a trajectory
        
        Returns:
            True if trajectory closes within coupling threshold
        """
        if len(trajectory) < 2:
            return False
        
        start = trajectory[0]
        end = trajectory[-1]
        
        # Check if end state is close to start state
        distance = np.linalg.norm(end - start)
        
        # Closure threshold: one coupling unit
        return distance < self.coupling
    
    def generate_trajectory(self, 
                          initial_state: np.ndarray, 
                          num_steps: int = 100) -> List[np.ndarray]:
        """
        Generate a trajectory through toroidal flow.
        
        Args:
            initial_state: Starting E8 state
            num_steps: Number of evolution steps
        
        Returns:
            List of E8 states forming trajectory
        """
        trajectory = [initial_state.copy()]
        current = initial_state.copy()
        
        for _ in range(num_steps):
            current = self.evolve_state(current)
            trajectory.append(current.copy())
        
        return trajectory
    
    def __repr__(self) -> str:
        return (f"ToroidalFlow(R={self.R:.2f}, r={self.r:.2f}, "
                f"coupling={self.coupling:.4f})")
