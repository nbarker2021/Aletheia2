
from typing import Dict, Any
import statistics

def compute_v_total(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    num = 0.0
    den = 0.0
    for k, w in weights.items():
        num += w * float(scores.get(k, 0.0))
        den += w
    return num/den if den else 0.0

def band_for(v: float) -> str:
    if v >= 0.80: return "BREAKTHROUGH"
    if v >= 0.60: return "PEER_READY"
    return "EXPLORATORY"
