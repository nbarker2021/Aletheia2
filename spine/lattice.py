"""
Lattice Systems - E8, Leech, and Niemeier

Integrated lattice operations for the spine.
Provides geometric substrate for all CQE operations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from spine.speedlight import get_speedlight, receipted


@dataclass
class LatticePoint:
    """A point in a lattice."""
    vector: np.ndarray
    index: int
    distance: float
    chamber: str = ""


class E8Lattice:
    """
    E8 Lattice - 8-dimensional lattice with 240 roots.
    
    The universal geometric substrate for CQE.
    """
    
    def __init__(self):
        self.dim = 8
        self.roots = self._generate_roots()
        self.simple_roots = self.roots[:8]
        self.speedlight = get_speedlight()
    
    def _generate_roots(self) -> np.ndarray:
        """Generate the 240 E8 root vectors."""
        roots = []
        
        # Type 1: All permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
        # 8 choose 2 * 2^2 = 112 roots
        for i in range(8):
            for j in range(i+1, 8):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        v = np.zeros(8)
                        v[i] = s1
                        v[j] = s2
                        roots.append(v)
        
        # Type 2: (±1/2, ±1/2, ..., ±1/2) with even number of minus signs
        # 2^7 = 128 roots
        for bits in range(256):
            signs = [1 if (bits >> i) & 1 else -1 for i in range(8)]
            if sum(1 for s in signs if s == -1) % 2 == 0:
                roots.append(np.array(signs) * 0.5)
        
        return np.array(roots)
    
    @receipted("e8_nearest")
    def nearest_root(self, vector: np.ndarray) -> LatticePoint:
        """Find the nearest E8 root to the given vector."""
        if len(vector) != 8:
            vector = np.pad(vector, (0, 8 - len(vector)))[:8]
        
        distances = np.linalg.norm(self.roots - vector, axis=1)
        idx = np.argmin(distances)
        
        return LatticePoint(
            vector=self.roots[idx],
            index=idx,
            distance=float(distances[idx]),
            chamber=self.determine_chamber(vector)
        )
    
    def determine_chamber(self, vector: np.ndarray) -> str:
        """Determine which Weyl chamber contains the vector."""
        if len(vector) != 8:
            vector = np.pad(vector, (0, 8 - len(vector)))[:8]
        
        inner_products = np.dot(self.simple_roots, vector)
        signs = np.sign(inner_products)
        return ''.join(['1' if s >= 0 else '0' for s in signs])
    
    def project(self, vector: np.ndarray) -> np.ndarray:
        """Project a vector onto the E8 lattice."""
        point = self.nearest_root(vector)
        return point.vector
    
    def coset_margin(self, vector: np.ndarray) -> float:
        """Calculate the coset margin (distance to nearest root)."""
        point = self.nearest_root(vector)
        return point.distance


class LeechLattice:
    """
    Leech Lattice - 24-dimensional lattice with 196560 minimal vectors.
    
    Connected to the Monster group and Moonshine.
    """
    
    KISSING_NUMBER = 196560
    
    def __init__(self, simplified: bool = True):
        self.dim = 24
        self.simplified = simplified
        self.speedlight = get_speedlight()
        
        if simplified:
            # Use a simplified representation for efficiency
            self.minimal_vectors = self._generate_simplified()
        else:
            # Full implementation would generate all 196560 vectors
            self.minimal_vectors = self._generate_simplified()
    
    def _generate_simplified(self) -> np.ndarray:
        """Generate a simplified set of Leech lattice vectors."""
        # Generate a representative subset
        vectors = []
        
        # Type 1: Scaled E8 embeddings (3 copies)
        e8 = E8Lattice()
        for offset in [0, 8, 16]:
            for root in e8.roots[:80]:  # Subset
                v = np.zeros(24)
                v[offset:offset+8] = root * 2
                vectors.append(v)
        
        # Normalize to Leech lattice norm
        vectors = np.array(vectors)
        if len(vectors) > 0:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vectors = vectors / norms * np.sqrt(8)  # Leech minimal norm
        
        return vectors
    
    @receipted("leech_nearest")
    def nearest_vector(self, vector: np.ndarray) -> LatticePoint:
        """Find the nearest Leech lattice vector."""
        if len(vector) != 24:
            vector = np.pad(vector, (0, 24 - len(vector)))[:24]
        
        if len(self.minimal_vectors) == 0:
            return LatticePoint(vector=np.zeros(24), index=0, distance=float('inf'))
        
        distances = np.linalg.norm(self.minimal_vectors - vector, axis=1)
        idx = np.argmin(distances)
        
        return LatticePoint(
            vector=self.minimal_vectors[idx],
            index=idx,
            distance=float(distances[idx])
        )
    
    def monster_metrics(self, vector: np.ndarray) -> Dict[str, Any]:
        """Calculate Monster group related metrics."""
        vector = np.array(vector)
        if len(vector) < 24:
            vector = np.pad(vector, (0, 24 - len(vector)))
        vector = vector[:24]
        
        # Quadratic form values
        q2 = float(np.sum(vector ** 2))
        q4 = float(np.sum(vector ** 4))
        
        # Mod 7 residue (Monster group connection)
        mod7 = int(np.sum(np.abs(vector) * 1000)) % 7
        
        return {
            "q2": q2,
            "q4": q4,
            "mod7": mod7,
            "kissing_number": self.KISSING_NUMBER
        }


class NiemeierLattice:
    """
    Niemeier Lattices - 24 even unimodular lattices in dimension 24.
    
    Each corresponds to a root system.
    """
    
    LATTICE_NAMES = [
        "D24", "D16+E8", "3E8", "A24", "2D12", "A17+E7", "D10+2E7",
        "A15+D9", "3D8", "2A12", "A11+D7+E6", "4E6", "2A9+D6",
        "4D6", "3A8", "2A7+2D5", "4A6", "6D4", "6A4", "8A3",
        "12A2", "24A1", "Leech", "0"  # Leech is special
    ]
    
    def __init__(self, name: str = "Leech"):
        self.name = name
        self.dim = 24
        
        if name == "Leech":
            self._lattice = LeechLattice()
        else:
            # Other Niemeier lattices use simplified construction
            self._lattice = self._construct(name)
    
    def _construct(self, name: str) -> LeechLattice:
        """Construct a Niemeier lattice by name."""
        # Simplified: all non-Leech Niemeier lattices use same base
        return LeechLattice(simplified=True)
    
    def nearest(self, vector: np.ndarray) -> LatticePoint:
        """Find nearest lattice point."""
        return self._lattice.nearest_vector(vector)


class LatticeEngine:
    """
    Unified Lattice Engine - Manages all lattice operations.
    
    Provides a single interface for E8, Leech, and Niemeier lattices.
    """
    
    def __init__(self):
        self.e8 = E8Lattice()
        self.leech = LeechLattice()
        self.niemeier: Dict[str, NiemeierLattice] = {}
        self.speedlight = get_speedlight()
    
    def get_e8(self) -> E8Lattice:
        """Get the E8 lattice."""
        return self.e8
    
    def get_leech(self) -> LeechLattice:
        """Get the Leech lattice."""
        return self.leech
    
    def get_niemeier(self, name: str) -> NiemeierLattice:
        """Get a Niemeier lattice by name."""
        if name not in self.niemeier:
            self.niemeier[name] = NiemeierLattice(name)
        return self.niemeier[name]
    
    def snap_to_e8(self, vector: np.ndarray) -> np.ndarray:
        """Snap a vector to the nearest E8 root."""
        return self.e8.project(vector)
    
    def get_status(self) -> Dict[str, Any]:
        """Get lattice engine status."""
        return {
            "e8_roots": len(self.e8.roots),
            "leech_vectors": len(self.leech.minimal_vectors),
            "niemeier_loaded": list(self.niemeier.keys())
        }


# Global instance
_engine: Optional[LatticeEngine] = None

def get_lattice_engine() -> LatticeEngine:
    """Get the global lattice engine."""
    global _engine
    if _engine is None:
        _engine = LatticeEngine()
    return _engine
