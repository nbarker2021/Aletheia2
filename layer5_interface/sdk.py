"""
CQE Native SDK

Provides a clean, user-friendly interface to the CQE Unified Runtime.

This is the primary API for interacting with the morphonic-native
geometric operating system.

Author: Manus AI (based on CQE research)
Date: December 5, 2025
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

# Import all layers
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_morphonic import UniversalMorphon, ObservationType, MGLCEngine, LambdaLevel
from layer2_geometric import E8Lattice, LeechLattice
from layer3_operational import ConservationEnforcer, MORSRExplorer
from layer4_governance import GravitationalLayer, SevenWitness


@dataclass
class CQEResult:
    """
    Result from a CQE operation.
    
    Attributes:
        success: Whether operation succeeded
        data: Result data
        metadata: Additional metadata
        validation: Validation results
    """
    success: bool
    data: Any
    metadata: Dict[str, Any]
    validation: Optional[Any] = None


class CQESDK:
    """
    CQE Native SDK
    
    Provides high-level interface to the CQE Unified Runtime.
    
    Key features:
    - Simple API for geometric operations
    - Automatic validation and governance
    - Conservation law enforcement
    - Multi-layer integration
    
    Example usage:
        >>> sdk = CQESDK()
        >>> result = sdk.embed_to_e8([1, 2, 3, 4, 5, 6, 7, 8])
        >>> print(result.success)
        True
    """
    
    def __init__(self, 
                 enable_validation: bool = True,
                 enable_conservation: bool = True):
        """
        Initialize the CQE SDK.
        
        Args:
            enable_validation: Enable Seven Witness validation
            enable_conservation: Enable conservation law enforcement
        """
        # Configuration
        self.enable_validation = enable_validation
        self.enable_conservation = enable_conservation
        
        # Initialize all layers
        self._init_layers()
        
        # Statistics
        self.operation_count = 0
        self.validation_count = 0
        self.conservation_violations = 0
    
    def _init_layers(self):
        """Initialize all runtime layers."""
        # Layer 1: Morphonic Foundation
        self.morphon = UniversalMorphon()
        self.mglc = MGLCEngine()
        
        # Layer 2: Geometric Engine
        self.e8 = E8Lattice()
        self.leech = LeechLattice()
        
        # Layer 3: Operational Systems
        self.conservation = ConservationEnforcer()
        self.morsr = MORSRExplorer()
        
        # Layer 4: Governance
        self.gravitational = GravitationalLayer()
        self.seven_witness = SevenWitness()
    
    def embed_to_e8(self, vector: Union[List, np.ndarray]) -> CQEResult:
        """
        Embed a vector into the E8 lattice.
        
        Args:
            vector: 8D input vector
        
        Returns:
            CQEResult with embedded vector
        """
        self.operation_count += 1
        
        # Convert to numpy array
        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.float64)
        
        # Validate input
        if len(vector) != 8:
            return CQEResult(
                success=False,
                data=None,
                metadata={"error": f"Expected 8D vector, got {len(vector)}D"},
                validation=None
            )
        
        # Project to E8
        projected = self.e8.project(vector)
        
        # Validate if enabled
        validation = None
        if self.enable_validation:
            validation = self.seven_witness.validate(projected)
            self.validation_count += 1
        
        return CQEResult(
            success=True,
            data=projected,
            metadata={
                "original_norm": float(np.linalg.norm(vector)),
                "projected_norm": float(np.linalg.norm(projected)),
                "distance": float(self.e8.distance_to_lattice(vector))
            },
            validation=validation
        )
    
    def expand_to_leech(self, e8_vector: Union[List, np.ndarray]) -> CQEResult:
        """
        Expand an E8 vector to the Leech lattice.
        
        Args:
            e8_vector: 8D E8 lattice vector
        
        Returns:
            CQEResult with 24D Leech vector
        """
        self.operation_count += 1
        
        # Convert to numpy array
        if isinstance(e8_vector, list):
            e8_vector = np.array(e8_vector, dtype=np.float64)
        
        # Validate input
        if len(e8_vector) != 8:
            return CQEResult(
                success=False,
                data=None,
                metadata={"error": f"Expected 8D vector, got {len(e8_vector)}D"},
                validation=None
            )
        
        # Embed to Leech
        leech_vector = self.leech.embed_e8(e8_vector)
        
        # Validate if enabled
        validation = None
        if self.enable_validation:
            validation = self.seven_witness.validate(leech_vector)
            self.validation_count += 1
        
        return CQEResult(
            success=True,
            data=leech_vector,
            metadata={
                "e8_norm": float(np.linalg.norm(e8_vector)),
                "leech_norm": float(np.linalg.norm(leech_vector)),
                "is_minimal": self.leech.is_minimal(leech_vector)
            },
            validation=validation
        )
    
    def validate_structure(self, structure: Any) -> CQEResult:
        """
        Validate a structure through Seven Witness and Gravitational Layer.
        
        Args:
            structure: Structure to validate
        
        Returns:
            CQEResult with validation verdict
        """
        self.operation_count += 1
        self.validation_count += 1
        
        # Gravitational validation
        grav_state = self.gravitational.validate_structure(structure)
        
        # Seven Witness validation
        verdict = self.seven_witness.validate(
            structure,
            context={"gravitational_state": grav_state}
        )
        
        return CQEResult(
            success=verdict.valid,
            data=verdict,
            metadata={
                "consensus": verdict.consensus,
                "digital_root": grav_state.digital_root.name,
                "gravitational_depth": grav_state.depth,
                "stable": grav_state.stable
            },
            validation=verdict
        )
    
    def explore(self, 
                target: Any,
                max_iterations: int = 100) -> CQEResult:
        """
        Explore a target using MORSR.
        
        Args:
            target: Target to explore
            max_iterations: Maximum exploration iterations
        
        Returns:
            CQEResult with exploration results
        """
        self.operation_count += 1
        
        # Run MORSR exploration
        results = self.morsr.explore(target, max_iterations=max_iterations)
        
        return CQEResult(
            success=results.get("converged", False),
            data=results,
            metadata={
                "iterations": results["iterations"],
                "max_depth": results["max_depth_reached"],
                "final_quality": results["final_quality"]
            },
            validation=None
        )
    
    def check_conservation(self, 
                          initial: np.ndarray,
                          final: np.ndarray) -> CQEResult:
        """
        Check if a transformation satisfies ΔΦ ≤ 0.
        
        Args:
            initial: Initial state
            final: Final state
        
        Returns:
            CQEResult with conservation check
        """
        self.operation_count += 1
        
        result = self.conservation.check_transformation(initial, final)
        
        if not result.valid:
            self.conservation_violations += 1
        
        return CQEResult(
            success=result.valid,
            data=result,
            metadata={
                "delta_phi": result.delta_phi,
                "phi_initial": result.phi_initial,
                "phi_final": result.phi_final
            },
            validation=None
        )
    
    def create_morphon(self, state: Optional[np.ndarray] = None) -> UniversalMorphon:
        """
        Create a new Universal Morphon.
        
        Args:
            state: Optional initial state
        
        Returns:
            Universal Morphon instance
        """
        self.operation_count += 1
        return UniversalMorphon(state=state)
    
    def parse_lambda(self, expression: str, level: LambdaLevel = LambdaLevel.LAMBDA_0) -> CQEResult:
        """
        Parse and evaluate a lambda expression.
        
        Args:
            expression: Lambda expression string
            level: Lambda calculus level
        
        Returns:
            CQEResult with parsed term
        """
        self.operation_count += 1
        
        try:
            term = self.mglc.parse(expression, level)
            normalized = self.mglc.normalize(term)
            
            return CQEResult(
                success=True,
                data=normalized,
                metadata={
                    "level": level.name,
                    "original": expression
                },
                validation=None
            )
        except Exception as e:
            return CQEResult(
                success=False,
                data=None,
                metadata={"error": str(e)},
                validation=None
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get SDK usage statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "operations": self.operation_count,
            "validations": self.validation_count,
            "conservation_violations": self.conservation_violations,
            "conservation_stats": self.conservation.get_statistics(),
            "gravitational_stats": self.gravitational.get_statistics(),
            "witness_stats": self.seven_witness.get_statistics()
        }
    
    def __repr__(self) -> str:
        return f"CQESDK(operations={self.operation_count}, validations={self.validation_count})"
