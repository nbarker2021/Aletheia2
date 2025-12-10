"""
Universal Morphon (M₀) Implementation

The Universal Morphon is the fundamental unit of morphonic geometry.
All structures are observations of M₀ through different functors.

Based on CQE morphonic theory and category-theoretic foundations.

Author: Manus AI (based on CQE research)
Date: December 5, 2025
"""

import numpy as np
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class ObservationType(Enum):
    """Types of observations that can be made on the Universal Morphon."""
    GEOMETRIC = "geometric"  # Observe as geometric structure (E8, Leech, etc.)
    ALGEBRAIC = "algebraic"  # Observe as algebraic structure (group, ring, etc.)
    TOPOLOGICAL = "topological"  # Observe as topological space
    COMPUTATIONAL = "computational"  # Observe as computation/lambda term
    PHYSICAL = "physical"  # Observe as physical phenomenon


@dataclass
class Observation:
    """
    An observation of the Universal Morphon through a specific functor.
    
    Attributes:
        type: Type of observation
        data: Observed data (structure-dependent)
        functor: Name of the observation functor used
        metadata: Additional metadata about the observation
    """
    type: ObservationType
    data: Any
    functor: str
    metadata: Dict[str, Any]


class UniversalMorphon:
    """
    Universal Morphon (M₀)
    
    The fundamental object in morphonic geometry. All mathematical structures
    are observations of M₀ through different functors.
    
    Key properties:
    - Category-theoretic foundation (M₀ is the initial object)
    - All structures emerge via observation functors
    - Preserves morphonic operations (⊕, ⊗, ∇)
    - Supports composition and transformation
    
    The Universal Morphon embodies the principle that "geometry is fundamental"
    and all other structures are derived observations.
    """
    
    def __init__(self, state: Optional[np.ndarray] = None):
        """
        Initialize the Universal Morphon.
        
        Args:
            state: Optional initial state (default: zero state)
        """
        # Internal state (abstract representation)
        self.state = state if state is not None else np.zeros(8)
        
        # Observation history
        self.observations: List[Observation] = []
        
        # Registered functors
        self.functors: Dict[str, Callable] = {}
        
        # Register default functors
        self._register_default_functors()
    
    def _register_default_functors(self):
        """Register default observation functors."""
        self.register_functor("identity", self._identity_functor)
        self.register_functor("geometric_e8", self._geometric_e8_functor)
        self.register_functor("algebraic", self._algebraic_functor)
    
    def _identity_functor(self, state: np.ndarray) -> Any:
        """Identity functor: observe state directly."""
        return state
    
    def _geometric_e8_functor(self, state: np.ndarray) -> Any:
        """Geometric functor: observe as E8 lattice point."""
        # This would use the E8Lattice from layer2_geometric
        # For now, return the state as-is
        return state
    
    def _algebraic_functor(self, state: np.ndarray) -> Any:
        """Algebraic functor: observe as algebraic structure."""
        # Convert to algebraic representation
        return {"coefficients": state.tolist()}
    
    def register_functor(self, name: str, functor: Callable):
        """
        Register a new observation functor.
        
        Args:
            name: Name of the functor
            functor: Callable that takes state and returns observed structure
        """
        self.functors[name] = functor
    
    def observe(self, 
                functor_name: str, 
                obs_type: ObservationType,
                metadata: Optional[Dict] = None) -> Observation:
        """
        Observe the morphon through a specific functor.
        
        Args:
            functor_name: Name of the observation functor
            obs_type: Type of observation
            metadata: Optional metadata
        
        Returns:
            Observation object containing the observed structure
        """
        if functor_name not in self.functors:
            raise ValueError(f"Unknown functor: {functor_name}")
        
        # Apply functor to current state
        functor = self.functors[functor_name]
        observed_data = functor(self.state)
        
        # Create observation
        obs = Observation(
            type=obs_type,
            data=observed_data,
            functor=functor_name,
            metadata=metadata or {}
        )
        
        # Record observation
        self.observations.append(obs)
        
        return obs
    
    def morphonic_add(self, other: 'UniversalMorphon') -> 'UniversalMorphon':
        """
        Morphonic addition (⊕).
        
        Args:
            other: Another Universal Morphon
        
        Returns:
            New Universal Morphon representing the sum
        """
        new_state = self.state + other.state
        return UniversalMorphon(state=new_state)
    
    def morphonic_multiply(self, other: 'UniversalMorphon') -> 'UniversalMorphon':
        """
        Morphonic multiplication (⊗).
        
        Args:
            other: Another Universal Morphon
        
        Returns:
            New Universal Morphon representing the product
        """
        # Element-wise multiplication (Hadamard product)
        new_state = self.state * other.state
        return UniversalMorphon(state=new_state)
    
    def morphonic_gradient(self) -> np.ndarray:
        """
        Morphonic gradient (∇).
        
        Returns:
            Gradient vector of the morphon state
        """
        # Compute discrete gradient
        # For now, use finite differences
        gradient = np.gradient(self.state)
        return gradient
    
    def transform(self, transformation: Callable[[np.ndarray], np.ndarray]) -> 'UniversalMorphon':
        """
        Apply a transformation to the morphon state.
        
        Args:
            transformation: Function that transforms the state
        
        Returns:
            New Universal Morphon with transformed state
        """
        new_state = transformation(self.state)
        return UniversalMorphon(state=new_state)
    
    def get_state(self) -> np.ndarray:
        """Get current morphon state."""
        return self.state.copy()
    
    def set_state(self, state: np.ndarray):
        """Set morphon state."""
        self.state = state.copy()
    
    def __repr__(self) -> str:
        return f"UniversalMorphon(state_dim={len(self.state)}, observations={len(self.observations)})"
    
    def __add__(self, other: 'UniversalMorphon') -> 'UniversalMorphon':
        """Operator overload for morphonic addition."""
        return self.morphonic_add(other)
    
    def __mul__(self, other: 'UniversalMorphon') -> 'UniversalMorphon':
        """Operator overload for morphonic multiplication."""
        return self.morphonic_multiply(other)
