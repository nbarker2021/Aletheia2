
from typing import Dict
import math

def compute_phi(features: Dict[str, float], weights: Dict[str, float]) -> float:
    # Simple weighted sum with sqrt squashing to [0,1]-ish
    s = 0.0
    for k, w in weights.items():
        v = float(features.get(k, 0.0))
        s += w * v
    return 1.0 / (1.0 + math.exp(-s))  # logistic squash

def feature_pack(geom: float, parity: float, sparsity: float, kissing: float) -> Dict[str, float]:
    return {"geom": geom, "parity": parity, "sparsity": sparsity, "kissing": kissing}
