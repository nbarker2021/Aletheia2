"""
Weyl Chamber Navigation for E8 Lattice

Weyl chambers are regions in E8 space separated by reflection hyperplanes.
The E8 Weyl group has order 696,729,600 and acts on the root system.

This module provides:
- Chamber determination (which chamber contains a vector)
- Chamber projection (move vector to specified chamber)
- Chamber-aware distance metrics
- Weyl group reflections

Ported from cqe_unified/cqe/L0_geometric.py
Author: Manus AI (based on CQE research)
Date: December 5, 2025
"""

import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class ChamberInfo:
    """Information about a Weyl chamber."""
    signature: str  # Binary signature (e.g., "11111111")
    inner_products: np.ndarray  # Inner products with simple roots
    is_fundamental: bool  # Is this the fundamental chamber?
    depth: float  # Distance to nearest chamber wall


class WeylChamberNavigator:
    """
    Navigate Weyl chambers in E8 space.
    
    The Weyl group W(E8) has order 696,729,600 and partitions E8 space
    into chambers separated by reflection hyperplanes.
    
    The fundamental chamber is defined by all inner products with simple
    roots being non-negative.
    """
    
    # E8 Weyl group order
    WEYL_GROUP_ORDER = 696729600
    
    def __init__(self, simple_roots: np.ndarray):
        """
        Initialize Weyl chamber navigator.
        
        Args:
            simple_roots: 8×8 array of simple roots (rows are roots)
        """
        if simple_roots.shape != (8, 8):
            raise ValueError(f"Expected 8×8 simple roots, got {simple_roots.shape}")
        
        self.simple_roots = simple_roots
        self.dimension = 8
        
        # Precompute root norms squared for efficiency
        self.root_norms_sq = np.sum(simple_roots ** 2, axis=1)
    
    def determine_chamber(self, vector: np.ndarray) -> ChamberInfo:
        """
        Determine which Weyl chamber contains the vector.
        
        Args:
            vector: 8D vector
            
        Returns:
            ChamberInfo with chamber details
        """
        if len(vector) != self.dimension:
            raise ValueError(f"Vector must be {self.dimension}D")
        
        # Calculate inner products with simple roots
        inner_products = np.dot(self.simple_roots, vector)
        
        # Determine chamber by sign pattern
        signs = np.sign(inner_products)
        
        # Create chamber signature
        signature = ''.join(['1' if s >= 0 else '0' for s in signs])
        
        # Fundamental chamber: all inner products ≥ 0
        is_fundamental = np.all(signs >= 0)
        
        # Chamber depth: distance to nearest wall
        depth = float(np.min(np.abs(inner_products)))
        
        return ChamberInfo(
            signature=signature,
            inner_products=inner_products,
            is_fundamental=is_fundamental,
            depth=depth
        )
    
    def reflect_across_root(self, vector: np.ndarray, root_index: int) -> np.ndarray:
        """
        Reflect vector across the hyperplane orthogonal to a simple root.
        
        Uses the Weyl reflection formula:
        s_α(v) = v - 2⟨v,α⟩/⟨α,α⟩ α
        
        Args:
            vector: 8D vector to reflect
            root_index: Index of simple root (0-7)
            
        Returns:
            Reflected vector
        """
        if not 0 <= root_index < 8:
            raise ValueError(f"Root index must be 0-7, got {root_index}")
        
        simple_root = self.simple_roots[root_index]
        root_norm_sq = self.root_norms_sq[root_index]
        
        if root_norm_sq < 1e-10:
            return vector.copy()
        
        # Reflection formula
        inner_prod = np.dot(vector, simple_root)
        reflected = vector - 2 * inner_prod / root_norm_sq * simple_root
        
        return reflected
    
    def project_to_chamber(self, vector: np.ndarray, target_signature: str = "11111111") -> np.ndarray:
        """
        Project vector to specified Weyl chamber.
        
        Default target is the fundamental chamber (all positive).
        
        Args:
            vector: 8D vector
            target_signature: Binary signature of target chamber
            
        Returns:
            Projected vector in target chamber
        """
        if len(vector) != self.dimension:
            raise ValueError(f"Vector must be {self.dimension}D")
        
        if len(target_signature) != 8:
            raise ValueError(f"Chamber signature must be 8 bits")
        
        # Get current chamber
        current_info = self.determine_chamber(vector)
        
        if current_info.signature == target_signature:
            return vector.copy()
        
        # Reflect across hyperplanes to reach target chamber
        projected = vector.copy()
        
        for i, (current_bit, target_bit) in enumerate(zip(current_info.signature, target_signature)):
            if current_bit != target_bit:
                projected = self.reflect_across_root(projected, i)
        
        return projected
    
    def project_to_fundamental(self, vector: np.ndarray) -> np.ndarray:
        """
        Project vector to the fundamental chamber.
        
        This is the most commonly used projection.
        
        Args:
            vector: 8D vector
            
        Returns:
            Vector in fundamental chamber
        """
        return self.project_to_chamber(vector, "11111111")
    
    def chamber_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate chamber-aware distance between vectors.
        
        Projects both vectors to fundamental chamber before measuring distance.
        
        Args:
            vec1: First 8D vector
            vec2: Second 8D vector
            
        Returns:
            Distance in fundamental chamber
        """
        proj1 = self.project_to_fundamental(vec1)
        proj2 = self.project_to_fundamental(vec2)
        
        return np.linalg.norm(proj1 - proj2)
    
    def chamber_quality(self, vector: np.ndarray) -> Dict[str, float]:
        """
        Assess quality of vector's chamber placement.
        
        Args:
            vector: 8D vector
            
        Returns:
            Dictionary with quality metrics
        """
        chamber_info = self.determine_chamber(vector)
        
        return {
            "chamber_signature": chamber_info.signature,
            "is_fundamental": chamber_info.is_fundamental,
            "chamber_depth": chamber_info.depth,
            "symmetry_score": float(np.std(chamber_info.inner_products)),
            "min_inner_product": float(np.min(chamber_info.inner_products)),
            "max_inner_product": float(np.max(chamber_info.inner_products))
        }
    
    def generate_chamber_samples(self, signature: str, count: int = 10) -> np.ndarray:
        """
        Generate random samples from specified Weyl chamber.
        
        Args:
            signature: Binary signature of chamber
            count: Number of samples to generate
            
        Returns:
            Array of shape (count, 8) with samples
        """
        samples = []
        
        attempts = 0
        max_attempts = count * 10
        
        while len(samples) < count and attempts < max_attempts:
            # Generate random vector
            vec = np.random.randn(self.dimension)
            
            # Project to desired chamber
            projected = self.project_to_chamber(vec, signature)
            
            # Verify it's in the right chamber
            chamber_info = self.determine_chamber(projected)
            if chamber_info.signature == signature:
                samples.append(projected)
            
            attempts += 1
        
        if len(samples) < count:
            print(f"Warning: Only generated {len(samples)}/{count} samples")
        
        return np.array(samples)
    
    def list_adjacent_chambers(self, signature: str) -> List[str]:
        """
        List chambers adjacent to the given chamber.
        
        Adjacent chambers differ by exactly one reflection.
        
        Args:
            signature: Binary signature of chamber
            
        Returns:
            List of adjacent chamber signatures
        """
        adjacent = []
        
        for i in range(8):
            # Flip bit i
            new_sig = list(signature)
            new_sig[i] = '0' if new_sig[i] == '1' else '1'
            adjacent.append(''.join(new_sig))
        
        return adjacent
    
    def __repr__(self) -> str:
        return f"WeylChamberNavigator(E8, {self.WEYL_GROUP_ORDER:,} chambers)"
