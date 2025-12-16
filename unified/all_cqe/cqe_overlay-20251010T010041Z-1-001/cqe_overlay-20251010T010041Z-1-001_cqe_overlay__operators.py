
from __future__ import annotations
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

@dataclass
class OpResult:
    new_state: Dict[str, Any]
    delta_phi: float
    accepted: bool
    reason_code: str  # strict_decrease, parity_strict, escrow_uphill, delta_increase, plateau

def _copy(s: Dict[str, Any]) -> Dict[str, Any]:
    return {k:(v[:] if isinstance(v, list) else v) for k,v in s.items()}

def rotate_Rtheta(state: Dict[str, Any]) -> Dict[str, Any]:
    s = _copy(state)
    lanes = s.get("lanes", [0.0]*8)
    if lanes:
        lanes = lanes[1:] + lanes[:1]
    s["lanes"] = lanes
    s["geom_err"] = max(0.0, s.get("geom_err", 0.0) - 0.1)
    return s

def weyl_reflect(state: Dict[str, Any], i: int = 0) -> Dict[str, Any]:
    s = _copy(state)
    lanes = s.get("lanes", [0.0]*8)
    lanes[i] = -lanes[i]
    s["lanes"] = lanes
    return s

def midpoint_expand(state: Dict[str, Any]) -> Dict[str, Any]:
    s = _copy(state)
    w = s.get("weights", [])
    if not w:
        w = [0.0]
    mid = len(w)//2
    w.insert(mid, 0.0)
    s["weights"] = w
    return s

def parity_mirror(state: Dict[str, Any]) -> Dict[str, Any]:
    s = _copy(state)
    lanes = s.get("lanes", [0.0]*8)
    lanes = [lanes[0], -lanes[1], lanes[2], -lanes[3], lanes[4], -lanes[5], lanes[6], -lanes[7]]
    s["lanes"] = lanes
    return s

def single_insert(state: Dict[str, Any]) -> Dict[str, Any]:
    s = _copy(state)
    w = s.get("weights", [])
    w.append(0.0)
    s["weights"] = w
    return s
