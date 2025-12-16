
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import random, math

class OperatorLibrary:
    """A small, clean set of operators to mutate lanes with parity awareness.
    All operators return (new_lanes, new_parity).
    """
    def __init__(self, seed: int = 0):
        self.rs = random.Random(seed)

    def R_theta(self, lanes: List[float], parity_ok: bool, k:int=1) -> Tuple[List[float], bool]:
        # cyclic rotate lanes by k
        k = k % len(lanes)
        new = lanes[-k:] + lanes[:-k]
        # parity preserved
        return new, parity_ok

    def Weyl_reflect(self, lanes: List[float], parity_ok: bool, idx:int=0) -> Tuple[List[float], bool]:
        # reflect sign on one lane; flip neighboring lane slightly
        new = list(lanes)
        if 0 <= idx < len(new):
            new[idx] = -new[idx]
            left = (idx-1) % len(new)
            right = (idx+1) % len(new)
            new[left] *= 0.99
            new[right] *= 0.99
        # parity toggles 10% of time to emulate difficult fixes
        flip = self.rs.random() < 0.10
        return new, (parity_ok if not flip else not parity_ok)

    def Midpoint(self, lanes: List[float], parity_ok: bool) -> Tuple[List[float], bool]:
        # pull each lane toward average of neighbors (smoothing); parity often improves
        new = list(lanes)
        L = len(new)
        if L >= 3:
            sm = []
            for i in range(L):
                sm.append(0.5*new[i] + 0.25*new[(i-1)%L] + 0.25*new[(i+1)%L])
            new = sm
        # 70% chance of fixing parity if it was broken
        if not parity_ok and self.rs.random() < 0.70:
            parity_ok = True
        return new, parity_ok

    def ECC_parity(self, lanes: List[float], parity_ok: bool) -> Tuple[List[float], bool]:
        # quantize lanes a bit toward integers; flip parity to True with high prob
        new = [x*0.9 + round(x)*0.1 for x in lanes]
        # 80% chance to resolve parity
        if not parity_ok and self.rs.random() < 0.80:
            parity_ok = True
        return new, parity_ok

    def SingleInsert(self, lanes: List[float], parity_ok: bool, idx:int=0, val:float=0.0) -> Tuple[List[float], bool]:
        # small perturbation at idx
        new = list(lanes)
        if 0 <= idx < len(new):
            new[idx] += val
        # parity unchanged
        return new, parity_ok

    def ParityMirror(self, lanes: List[float], parity_ok: bool, strength:float=0.2) -> Tuple[List[float], bool]:
        # try to mirror lanes across midpoint; may worsen geom, may fix parity
        L = len(lanes)
        new = list(lanes)
        for i in range(L//2):
            j = L-1-i
            m = 0.5*(lanes[i] + lanes[j])
            new[i] = (1-strength)*lanes[i] + strength*m
            new[j] = (1-strength)*lanes[j] + strength*m
        # 50% chance to flip parity to OK
        if self.rs.random() < 0.5:
            parity_ok = True
        return new, parity_ok

    def shortlist(self) -> List[str]:
        return [
            "R_theta", "Weyl_reflect", "Midpoint", "ECC_parity", "SingleInsert", "ParityMirror"
        ]
