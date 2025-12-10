"""
Enhanced Vector Operations for CQE

Provides optimized vector operations commonly used in CQE:
- Gram-Schmidt orthogonalization
- Vector projections
- Angle computations
- Norm operations
- Digital root calculations

Author: Manus AI
Date: December 5, 2025
"""

import numpy as np
from typing import List, Tuple, Optional


class VectorOperations:
    """Collection of enhanced vector operations for CQE."""
    
    # Golden ratio
    PHI = (1 + np.sqrt(5)) / 2
    
    @staticmethod
    def digital_root(n: int) -> int:
        """
        Compute digital root (repeated digit sum until single digit).
        
        Args:
            n: Input integer
        
        Returns:
            Digital root (0-9)
        """
        n = abs(int(n))
        if n == 0:
            return 0
        return 1 + ((n - 1) % 9)
    
    @staticmethod
    def vector_digital_root(vec: np.ndarray) -> int:
        """
        Compute digital root of vector sum.
        
        Args:
            vec: Input vector
        
        Returns:
            Digital root of sum of components
        """
        vec_sum = int(np.sum(np.abs(vec)))
        return VectorOperations.digital_root(vec_sum)
    
    @staticmethod
    def gram_schmidt(vectors: List[np.ndarray]) -> List[np.ndarray]:
        """
        Gram-Schmidt orthogonalization.
        
        Args:
            vectors: List of vectors to orthogonalize
        
        Returns:
            List of orthogonal vectors
        """
        if not vectors:
            return []
        
        orthogonal = []
        
        for v in vectors:
            # Subtract projections onto all previous orthogonal vectors
            u = v.copy()
            for o in orthogonal:
                proj = np.dot(u, o) / np.dot(o, o) * o
                u = u - proj
            
            # Normalize
            norm = np.linalg.norm(u)
            if norm > 1e-10:
                u = u / norm
                orthogonal.append(u)
        
        return orthogonal
    
    @staticmethod
    def project_onto(v: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Project vector v onto vector u.
        
        Args:
            v: Vector to project
            u: Target vector
        
        Returns:
            Projection of v onto u
        """
        u_norm_sq = np.dot(u, u)
        if u_norm_sq < 1e-10:
            return np.zeros_like(v)
        
        return np.dot(v, u) / u_norm_sq * u
    
    @staticmethod
    def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute angle between two vectors in radians.
        
        Args:
            v1: First vector
            v2: Second vector
        
        Returns:
            Angle in radians [0, π]
        """
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm < 1e-10 or v2_norm < 1e-10:
            return 0.0
        
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1, 1)
        
        return np.arccos(cos_angle)
    
    @staticmethod
    def normalize(vec: np.ndarray, target_norm: float = 1.0) -> np.ndarray:
        """
        Normalize vector to target norm.
        
        Args:
            vec: Input vector
            target_norm: Target norm (default 1.0)
        
        Returns:
            Normalized vector
        """
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return vec.copy()
        
        return vec * (target_norm / norm)
    
    @staticmethod
    def normalize_e8(vec: np.ndarray) -> np.ndarray:
        """
        Normalize to E8 standard norm (√2).
        
        Args:
            vec: 8D vector
        
        Returns:
            Vector with norm √2
        """
        return VectorOperations.normalize(vec, target_norm=np.sqrt(2))
    
    @staticmethod
    def is_orthogonal(v1: np.ndarray, v2: np.ndarray, 
                     tolerance: float = 1e-6) -> bool:
        """
        Check if two vectors are orthogonal.
        
        Args:
            v1: First vector
            v2: Second vector
            tolerance: Numerical tolerance
        
        Returns:
            True if vectors are orthogonal
        """
        return abs(np.dot(v1, v2)) < tolerance
    
    @staticmethod
    def is_parallel(v1: np.ndarray, v2: np.ndarray, 
                   tolerance: float = 1e-6) -> bool:
        """
        Check if two vectors are parallel.
        
        Args:
            v1: First vector
            v2: Second vector
            tolerance: Numerical tolerance
        
        Returns:
            True if vectors are parallel
        """
        angle = VectorOperations.angle_between(v1, v2)
        return angle < tolerance or abs(angle - np.pi) < tolerance
    
    @staticmethod
    def golden_ratio_split(vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split vector using golden ratio.
        
        Returns two vectors that sum to original, with norms in golden ratio.
        
        Args:
            vec: Input vector
        
        Returns:
            (major, minor) tuple where ||major||/||minor|| ≈ φ
        """
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return vec.copy(), np.zeros_like(vec)
        
        # Major part: φ/(1+φ) = 1/φ of the norm
        major_norm = norm / VectorOperations.PHI
        minor_norm = norm - major_norm
        
        direction = vec / norm
        major = direction * major_norm
        minor = direction * minor_norm
        
        return major, minor
    
    @staticmethod
    def reflect_across(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """
        Reflect vector across hyperplane with given normal.
        
        Uses formula: v' = v - 2⟨v,n⟩/⟨n,n⟩ n
        
        Args:
            v: Vector to reflect
            normal: Normal vector of hyperplane
        
        Returns:
            Reflected vector
        """
        normal_norm_sq = np.dot(normal, normal)
        if normal_norm_sq < 1e-10:
            return v.copy()
        
        proj = np.dot(v, normal) / normal_norm_sq
        return v - 2 * proj * normal
    
    @staticmethod
    def component_wise_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        Hadamard (component-wise) product.
        
        Args:
            v1: First vector
            v2: Second vector
        
        Returns:
            Component-wise product
        """
        return v1 * v2
    
    @staticmethod
    def l1_norm(vec: np.ndarray) -> float:
        """L1 (Manhattan) norm."""
        return float(np.sum(np.abs(vec)))
    
    @staticmethod
    def l2_norm(vec: np.ndarray) -> float:
        """L2 (Euclidean) norm."""
        return float(np.linalg.norm(vec))
    
    @staticmethod
    def linf_norm(vec: np.ndarray) -> float:
        """L∞ (maximum) norm."""
        return float(np.max(np.abs(vec)))
    
    @staticmethod
    def all_norms(vec: np.ndarray) -> dict:
        """Compute all common norms."""
        return {
            'l1': VectorOperations.l1_norm(vec),
            'l2': VectorOperations.l2_norm(vec),
            'linf': VectorOperations.linf_norm(vec)
        }
    
    @staticmethod
    def pairwise_distances(vectors: List[np.ndarray]) -> np.ndarray:
        """
        Compute pairwise distances between vectors.
        
        Args:
            vectors: List of vectors
        
        Returns:
            Distance matrix
        """
        n = len(vectors)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(vectors[i] - vectors[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    @staticmethod
    def centroid(vectors: List[np.ndarray]) -> np.ndarray:
        """
        Compute centroid of vectors.
        
        Args:
            vectors: List of vectors
        
        Returns:
            Centroid vector
        """
        if not vectors:
            raise ValueError("Cannot compute centroid of empty list")
        
        return np.mean(vectors, axis=0)
    
    @staticmethod
    def variance(vectors: List[np.ndarray]) -> float:
        """
        Compute variance of vectors from their centroid.
        
        Args:
            vectors: List of vectors
        
        Returns:
            Total variance
        """
        if not vectors:
            return 0.0
        
        center = VectorOperations.centroid(vectors)
        return float(np.mean([np.linalg.norm(v - center)**2 for v in vectors]))
