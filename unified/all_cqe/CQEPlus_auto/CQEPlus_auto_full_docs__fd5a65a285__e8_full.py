"""
Complete E8 Root System Implementation

Generates all 240 roots of E8 Lie algebra from 8 simple roots.
Implements reflection, projection, and Weyl group operations.

Based on morphonic principles:
- Geometry emerges from seeds (8 simple roots)
- All operations preserve invariants (norm, parity)
- Receipts generated for all operations
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import hashlib
import json

@dataclass
class E8Root:
    """Single E8 root vector"""
    vector: np.ndarray
    norm_squared: float
    parity: int  # 0 = even, 1 = odd
    root_type: str  # "simple", "positive", "negative"
    
    def __post_init__(self):
        """Compute derived properties"""
        self.norm_squared = float(np.dot(self.vector, self.vector))
        self.parity = int(np.sum(self.vector)) % 2
        
    def to_dict(self) -> dict:
        """Convert to dictionary for receipts"""
        return {
            "vector": self.vector.tolist(),
            "norm_squared": self.norm_squared,
            "parity": self.parity,
            "root_type": self.root_type
        }

class E8Full:
    """
    Complete E8 root system with all 240 roots.
    
    Root Types:
    - Type 1: (±1, ±1, 0, 0, 0, 0, 0, 0) and permutations (112 roots)
    - Type 2: (±1/2)^8 with even number of minus signs (128 roots)
    
    Total: 240 roots, all with norm² = 2, all with even parity
    """
    
    def __init__(self):
        """Initialize E8 root system"""
        self.dimension = 8
        self.simple_roots = self._generate_simple_roots()
        self.all_roots = self._generate_all_roots()
        self.root_bank = {self._hash_vector(r.vector): r for r in self.all_roots}
        
        # Verify invariants
        self._verify_invariants()
        
    def _generate_simple_roots(self) -> List[E8Root]:
        """
        Generate 8 simple roots of E8.
        
        Using standard basis:
        α₁ = (1, -1, 0, 0, 0, 0, 0, 0)
        α₂ = (0, 1, -1, 0, 0, 0, 0, 0)
        α₃ = (0, 0, 1, -1, 0, 0, 0, 0)
        α₄ = (0, 0, 0, 1, -1, 0, 0, 0)
        α₅ = (0, 0, 0, 0, 1, -1, 0, 0)
        α₆ = (0, 0, 0, 0, 0, 1, -1, 0)
        α₇ = (0, 0, 0, 0, 0, 1, 1, 0)
        α₈ = (-1/2, -1/2, -1/2, -1/2, -1/2, -1/2, -1/2, -1/2)
        """
        simple = []
        
        # α₁ through α₆
        for i in range(6):
            v = np.zeros(8)
            v[i] = 1
            v[i+1] = -1
            simple.append(E8Root(v, 0, 0, "simple"))
            
        # α₇ = (0, 0, 0, 0, 0, 1, 1, 0)
        v = np.zeros(8)
        v[5] = 1
        v[6] = 1
        simple.append(E8Root(v, 0, 0, "simple"))
        
        # α₈ = (-1/2)^8
        v = np.full(8, -0.5)
        simple.append(E8Root(v, 0, 0, "simple"))
        
        return simple
    
    def _generate_all_roots(self) -> List[E8Root]:
        """
        Generate all 240 roots from simple roots.
        
        Uses two constructions:
        1. Type 1: (±1, ±1, 0, 0, 0, 0, 0, 0) and permutations
        2. Type 2: (±1/2)^8 with even parity
        """
        roots = []
        
        # Type 1: (±1, ±1, 0, 0, 0, 0, 0, 0) and permutations
        # Choose 2 positions out of 8: C(8,2) = 28
        # Each can be (±1, ±1): 4 choices
        # Total: 28 × 4 = 112 roots
        
        for i in range(8):
            for j in range(i+1, 8):
                for sign_i in [1, -1]:
                    for sign_j in [1, -1]:
                        v = np.zeros(8)
                        v[i] = sign_i
                        v[j] = sign_j
                        roots.append(E8Root(v.copy(), 0, 0, "positive"))
        
        # Type 2: (±1/2)^8 with even number of minus signs
        # 2^8 = 256 total sign patterns
        # Half (128) have even parity
        
        for pattern in range(256):
            v = np.zeros(8)
            minus_count = 0
            for bit in range(8):
                if (pattern >> bit) & 1:
                    v[bit] = 0.5
                else:
                    v[bit] = -0.5
                    minus_count += 1
            
            # Only keep even parity (even number of minus signs)
            if minus_count % 2 == 0:
                roots.append(E8Root(v.copy(), 0, 0, "positive"))
        
        return roots
    
    def _verify_invariants(self):
        """Verify E8 invariants"""
        assert len(self.all_roots) == 240, f"Expected 240 roots, got {len(self.all_roots)}"
        
        for root in self.all_roots:
            # All roots have norm² = 2
            assert abs(root.norm_squared - 2.0) < 1e-10, \
                f"Root {root.vector} has norm² = {root.norm_squared}, expected 2"
            
            # All roots have even parity
            assert root.parity == 0, \
                f"Root {root.vector} has odd parity"
    
    def reflect(self, vector: np.ndarray, root: E8Root) -> np.ndarray:
        """
        Reflect vector across hyperplane perpendicular to root.
        
        Formula: v' = v - 2(v·α)/(α·α) α
        
        For E8 roots, α·α = 2, so: v' = v - (v·α)α
        """
        dot_product = np.dot(vector, root.vector)
        reflected = vector - dot_product * root.vector
        return reflected
    
    def project(self, vector: np.ndarray) -> Tuple[E8Root, float]:
        """
        Project arbitrary vector to nearest E8 root.
        
        Returns:
            (nearest_root, distance)
        """
        min_dist = float('inf')
        nearest = None
        
        for root in self.all_roots:
            dist = np.linalg.norm(vector - root.vector)
            if dist < min_dist:
                min_dist = dist
                nearest = root
        
        return nearest, min_dist
    
    def inner_product(self, root1: E8Root, root2: E8Root) -> float:
        """Compute inner product between two roots"""
        return float(np.dot(root1.vector, root2.vector))
    
    def is_positive_root(self, root: E8Root) -> bool:
        """Check if root is positive (first non-zero coordinate is positive)"""
        for coord in root.vector:
            if abs(coord) > 1e-10:
                return coord > 0
        return True
    
    def cartan_matrix(self) -> np.ndarray:
        """
        Compute Cartan matrix from simple roots.
        
        C_ij = 2(α_i · α_j)/(α_i · α_i)
        
        For E8, all simple roots have norm² = 2, so:
        C_ij = α_i · α_j
        """
        n = len(self.simple_roots)
        cartan = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                cartan[i, j] = self.inner_product(
                    self.simple_roots[i],
                    self.simple_roots[j]
                )
        
        return cartan
    
    def _hash_vector(self, v: np.ndarray) -> str:
        """Hash vector for lookup"""
        return hashlib.sha256(v.tobytes()).hexdigest()[:16]
    
    def lookup_root(self, vector: np.ndarray) -> Optional[E8Root]:
        """Look up root by vector"""
        h = self._hash_vector(vector)
        return self.root_bank.get(h)
    
    def generate_receipt(self, operation: str, **kwargs) -> dict:
        """Generate receipt for E8 operation"""
        receipt = {
            "operation": operation,
            "timestamp": np.datetime64('now').astype(str),
            "dimension": self.dimension,
            "root_count": len(self.all_roots),
            **kwargs
        }
        
        # Add hash
        receipt_str = json.dumps(receipt, sort_keys=True)
        receipt["hash"] = hashlib.sha256(receipt_str.encode()).hexdigest()[:16]
        
        return receipt
    
    def to_dict(self) -> dict:
        """Export E8 structure to dictionary"""
        return {
            "dimension": self.dimension,
            "simple_roots": [r.to_dict() for r in self.simple_roots],
            "root_count": len(self.all_roots),
            "cartan_matrix": self.cartan_matrix().tolist(),
            "invariants": {
                "all_norms_equal_2": all(abs(r.norm_squared - 2.0) < 1e-10 for r in self.all_roots),
                "all_even_parity": all(r.parity == 0 for r in self.all_roots),
                "count_240": len(self.all_roots) == 240
            }
        }
    
    def save_root_bank(self, filepath: str):
        """Save all roots to file"""
        data = {
            "roots": [r.to_dict() for r in self.all_roots],
            "metadata": self.to_dict()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(self.all_roots)} roots to {filepath}")


if __name__ == "__main__":
    # Test E8Full implementation
    print("Initializing E8 root system...")
    e8 = E8Full()
    
    print(f"\n✓ Generated {len(e8.all_roots)} roots")
    print(f"✓ All roots have norm² = 2: {all(abs(r.norm_squared - 2.0) < 1e-10 for r in e8.all_roots)}")
    print(f"✓ All roots have even parity: {all(r.parity == 0 for r in e8.all_roots)}")
    
    print("\nCartan matrix:")
    print(e8.cartan_matrix())
    
    print("\nSimple roots:")
    for i, root in enumerate(e8.simple_roots):
        print(f"α_{i+1} = {root.vector}")
    
    # Test reflection
    test_vector = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    reflected = e8.reflect(test_vector, e8.simple_roots[0])
    print(f"\nReflection test:")
    print(f"Original: {test_vector}")
    print(f"Reflected: {reflected}")
    
    # Test projection
    random_vector = np.random.randn(8)
    nearest, dist = e8.project(random_vector)
    print(f"\nProjection test:")
    print(f"Random vector: {random_vector}")
    print(f"Nearest root: {nearest.vector}")
    print(f"Distance: {dist:.4f}")
    
    # Generate receipt
    receipt = e8.generate_receipt(
        "e8_initialization",
        root_count=len(e8.all_roots),
        invariants_verified=True
    )
    print(f"\nReceipt: {json.dumps(receipt, indent=2)}")
    
    # Save root bank
    e8.save_root_bank(str(Path(__file__).parent / "e8_root_bank.json"))
    
    print("\n✓ E8Full implementation complete and verified")

