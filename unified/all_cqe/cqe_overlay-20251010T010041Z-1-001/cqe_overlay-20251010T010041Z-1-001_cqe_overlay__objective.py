
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class ObjectiveWeights:
    alpha: float = 1.0  # geometry
    beta: float  = 1.0  # parity
    gamma: float = 0.1  # sparsity
    delta: float = 0.1  # kissing

@dataclass
class PhiTerms:
    geom: float
    parity: float
    sparsity: float
    kissing: float

def phi_geom(state: Dict[str, Any]) -> float:
    # Placeholder: Coxeter smoothness / ring variance proxy
    return float(state.get("geom_err", 0.0))

def phi_parity(state: Dict[str, Any]) -> float:
    # Placeholder: simple even-parity check over 8 lanes
    lanes = state.get("lanes", [0]*8)
    synd = sum(int(abs(x*10)) & 1 for x in lanes)  # fake syndrome count
    return float(synd)

def phi_sparsity(state: Dict[str, Any]) -> float:
    # L1 on active weights (placeholder)
    w = state.get("weights", [])
    return float(sum(abs(x) for x in w))

def phi_kissing(state: Dict[str, Any]) -> float:
    # Neighbor deviation proxy (placeholder)
    return float(state.get("neighbor_dev", 0.0))

def compute_phi(state: Dict[str, Any], w: ObjectiveWeights) -> (float, PhiTerms):
    terms = PhiTerms(
        geom=phi_geom(state),
        parity=phi_parity(state),
        sparsity=phi_sparsity(state),
        kissing=phi_kissing(state)
    )
    total = w.alpha*terms.geom + w.beta*terms.parity + w.gamma*terms.sparsity + w.delta*terms.kissing
    return total, terms
