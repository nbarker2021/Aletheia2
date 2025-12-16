
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import math

class Phi:
    """Lightweight Φ objective with 4 components:
       - geom: radial smoothness and Coxeter-like angular acceleration proxy
       - parity: penalty if state.parity_ok is False
       - sparsity: L1 norm
       - lattice: distance to nearest pseudo-lattice point (integer grid)
    """
    def __init__(self, w_geom=1.0, w_parity=1.0, w_sparsity=0.1, w_lattice=0.2):
        self.w_geom = w_geom
        self.w_parity = w_parity
        self.w_sparsity = w_sparsity
        self.w_lattice = w_lattice

    @staticmethod
    def _radial(v: List[float]) -> float:
        return math.sqrt(sum(x*x for x in v)) + 1e-12

    @staticmethod
    def _angular_accel_proxy(v: List[float]) -> float:
        # simple 2nd-difference energy on lanes — rotational smoothness proxy
        if len(v) < 3:
            return 0.0
        acc = 0.0
        for i in range(1, len(v)-1):
            acc += abs(v[i-1] - 2*v[i] + v[i+1])
        # wrap-around terms to impose circularity
        acc += abs(v[-2] - 2*v[-1] + v[0])
        acc += abs(v[-1] - 2*v[0] + v[1])
        return acc

    @staticmethod
    def _lattice_distance(v: List[float]) -> float:
        # distance to nearest integer grid
        return sum((x - round(x))**2 for x in v)

    @staticmethod
    def _l1(v: List[float]) -> float:
        return sum(abs(x) for x in v)

    def components(self, lanes: List[float], parity_ok: bool) -> Dict[str, float]:
        r = self._radial(lanes)
        geom = self._angular_accel_proxy(lanes) / (1.0 + r)
        parity = 0.0 if parity_ok else 8.0  # flat penalty if parity broken
        sparsity = self._l1(lanes)
        lattice = self._lattice_distance(lanes)
        return {
            "geom": geom,
            "parity": parity,
            "sparsity": sparsity,
            "lattice": lattice,
        }

    def total(self, lanes: List[float], parity_ok: bool) -> float:
        c = self.components(lanes, parity_ok)
        return (self.w_geom*c["geom"] +
                self.w_parity*c["parity"] +
                self.w_sparsity*c["sparsity"] +
                self.w_lattice*c["lattice"])
