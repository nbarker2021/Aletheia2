
from typing import Dict, Any, List, Tuple
import math, random

def _cos(a: List[float], b: List[float]) -> float:
    na = math.sqrt(sum(x*x for x in a)) or 1.0
    nb = math.sqrt(sum(x*x for x in b)) or 1.0
    return sum(x*y for x,y in zip(a,b))/(na*nb)

def project_to_planes(v: List[float]) -> Dict[str, float]:
    # Simple orthogonal plane pairs: (0,1),(2,3),(4,5),(6,7)
    planes = {}
    for i,(a,b) in enumerate(((0,1),(2,3),(4,5),(6,7))):
        planes[f"P{i}"] = math.sqrt(v[a]*v[a] + v[b]*v[b])
    return planes

def w4_geometry(v: List[float]) -> float:
    # Local geometry score: max plane energy share (encourages concentration)
    planes = project_to_planes(v)
    total = sum(planes.values()) or 1.0
    return max(planes.values())/total

def w80_parity_stability(v: List[float], trials: int=50, sigma: float=1e-3) -> float:
    # Stability under small gaussian noise (higher is better)
    import random
    base_signs = tuple(1 if x>=0 else -1 for x in v)
    same = 0
    for _ in range(trials):
        vp = [x + random.uniform(-sigma, sigma) for x in v]
        signs = tuple(1 if x>=0 else -1 for x in vp)
        if signs == base_signs:
            same += 1
    return same/trials

def wexp_sparsity(v: List[float]) -> float:
    # Encourage few large coordinates
    absvals = sorted([abs(x) for x in v], reverse=True)
    topk = sum(absvals[:2])
    total = sum(absvals) or 1.0
    return topk/total  # closer to 1 means sparse

def monster_24D_projection(v: List[float]) -> Dict[str,float]:
    # Expand 8D to 24D by repeating and slight phase shifts; compute simple modular invariants
    import math
    V = []
    for k in range(3):
        for i in range(8):
            # slight rotation-like shift
            V.append(v[i] * math.cos((k+1)*(i+1)/10.0))
    # invariants
    q2 = sum(x*x for x in V)
    q4 = sum(x**4 for x in V)
    mod7 = sum(int(abs(x)*1e6)%7 for x in V) % 7
    return {"q2": q2, "q4": q4, "mod7": mod7}

def monster_pass(inv: Dict[str,float]) -> float:
    # Heuristic: normalized q2 in [1.5, 3.5], q4 not extreme, mod7 near middle
    score = 0.0
    if 1.0 <= inv["q2"] <= 4.0: score += 0.4
    if inv["q4"] <= 10.0: score += 0.4
    if inv["mod7"] in (3,4): score += 0.2
    return score  # in [0,1]
