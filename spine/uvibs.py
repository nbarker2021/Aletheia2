"""
UVIBS - Universal Vector Identity Bucket System

Provides geometric metrics and validation for CQE atoms.
Integrates with Monster group projections.
"""

import math
import random
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import numpy as np

from spine.kernel import CQEAtom
from spine.speedlight import get_speedlight, receipted


@dataclass
class UVIBSMetrics:
    """Complete UVIBS metrics for an atom."""
    w4_geometry: float  # Plane concentration score
    w80_parity_stability: float  # Stability under noise
    wexp_sparsity: float  # Sparsity score
    monster_q2: float  # 24D quadratic form
    monster_q4: float  # 24D quartic form
    monster_mod7: int  # Mod 7 residue
    monster_pass: float  # Overall Monster validation score
    bucket: int  # UVIBS bucket assignment


def project_to_planes(v: List[float]) -> Dict[str, float]:
    """Project vector to orthogonal plane pairs."""
    planes = {}
    for i, (a, b) in enumerate(((0, 1), (2, 3), (4, 5), (6, 7))):
        if a < len(v) and b < len(v):
            planes[f"P{i}"] = math.sqrt(v[a] * v[a] + v[b] * v[b])
        else:
            planes[f"P{i}"] = 0.0
    return planes


def w4_geometry(v: List[float]) -> float:
    """
    W4 Geometry Score.
    
    Measures plane energy concentration (higher = more concentrated).
    """
    planes = project_to_planes(v)
    total = sum(planes.values()) or 1.0
    return max(planes.values()) / total


def w80_parity_stability(v: List[float], trials: int = 50, sigma: float = 1e-3) -> float:
    """
    W80 Parity Stability Score.
    
    Measures stability of sign pattern under small perturbations.
    """
    base_signs = tuple(1 if x >= 0 else -1 for x in v)
    same = 0
    for _ in range(trials):
        vp = [x + random.uniform(-sigma, sigma) for x in v]
        signs = tuple(1 if x >= 0 else -1 for x in vp)
        if signs == base_signs:
            same += 1
    return same / trials


def wexp_sparsity(v: List[float]) -> float:
    """
    Sparsity Score.
    
    Measures concentration in top-k coordinates (higher = sparser).
    """
    absvals = sorted([abs(x) for x in v], reverse=True)
    topk = sum(absvals[:2])
    total = sum(absvals) or 1.0
    return topk / total


def monster_24d_projection(v: List[float]) -> Dict[str, float]:
    """
    Monster 24D Projection.
    
    Expands 8D vector to 24D with phase shifts and computes modular invariants.
    """
    V = []
    for k in range(3):
        for i in range(min(8, len(v))):
            V.append(v[i] * math.cos((k + 1) * (i + 1) / 10.0))
    
    # Pad to 24D if needed
    while len(V) < 24:
        V.append(0.0)
    
    q2 = sum(x * x for x in V)
    q4 = sum(x ** 4 for x in V)
    mod7 = sum(int(abs(x) * 1e6) % 7 for x in V) % 7
    
    return {"q2": q2, "q4": q4, "mod7": mod7}


def monster_pass(inv: Dict[str, float]) -> float:
    """
    Monster Pass Score.
    
    Heuristic validation against Monster group constraints.
    """
    score = 0.0
    if 1.0 <= inv["q2"] <= 4.0:
        score += 0.4
    if inv["q4"] <= 10.0:
        score += 0.4
    if inv["mod7"] in (3, 4):
        score += 0.2
    return score


def assign_bucket(metrics: UVIBSMetrics) -> int:
    """
    Assign UVIBS bucket based on metrics.
    
    Buckets 1-8 based on geometric properties.
    """
    # Simple bucket assignment based on w4 geometry
    if metrics.w4_geometry > 0.8:
        return 1  # Highly concentrated
    elif metrics.w4_geometry > 0.6:
        return 2
    elif metrics.w4_geometry > 0.4:
        return 3
    elif metrics.monster_pass > 0.8:
        return 4  # Monster-aligned
    elif metrics.monster_pass > 0.5:
        return 5
    elif metrics.w80_parity_stability > 0.8:
        return 6  # Parity-stable
    elif metrics.wexp_sparsity > 0.7:
        return 7  # Sparse
    else:
        return 8  # General


class UVIBSEngine:
    """
    UVIBS Engine - Computes and manages UVIBS metrics.
    """
    
    def __init__(self):
        self.speedlight = get_speedlight()
    
    @receipted("uvibs_compute")
    def compute(self, atom: CQEAtom) -> UVIBSMetrics:
        """Compute full UVIBS metrics for an atom."""
        v = atom.lanes.tolist()
        
        # Core metrics
        w4 = w4_geometry(v)
        w80 = w80_parity_stability(v)
        wexp = wexp_sparsity(v)
        
        # Monster projection
        monster = monster_24d_projection(v)
        mp = monster_pass(monster)
        
        metrics = UVIBSMetrics(
            w4_geometry=w4,
            w80_parity_stability=w80,
            wexp_sparsity=wexp,
            monster_q2=monster["q2"],
            monster_q4=monster["q4"],
            monster_mod7=monster["mod7"],
            monster_pass=mp,
            bucket=0  # Will be assigned
        )
        
        # Assign bucket
        metrics.bucket = assign_bucket(metrics)
        
        return metrics
    
    def validate(self, atom: CQEAtom, threshold: float = 0.5) -> Tuple[bool, str]:
        """Validate an atom against UVIBS constraints."""
        metrics = self.compute(atom)
        
        if metrics.monster_pass < threshold:
            return False, f"Monster pass failed: {metrics.monster_pass:.2f} < {threshold}"
        
        if metrics.w80_parity_stability < 0.3:
            return False, f"Parity unstable: {metrics.w80_parity_stability:.2f}"
        
        return True, f"UVIBS valid: bucket={metrics.bucket}, monster={metrics.monster_pass:.2f}"
    
    def get_status(self) -> Dict[str, Any]:
        """Get UVIBS engine status."""
        return {
            "active": True,
            "buckets": list(range(1, 9))
        }


# Global instance
_uvibs: UVIBSEngine = None

def get_uvibs() -> UVIBSEngine:
    """Get the global UVIBS engine."""
    global _uvibs
    if _uvibs is None:
        _uvibs = UVIBSEngine()
    return _uvibs
