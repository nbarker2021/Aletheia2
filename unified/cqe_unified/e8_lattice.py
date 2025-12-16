
from typing import List, Tuple
import itertools, math

def _roots_type1():
    # 112 roots: permutations of (±1, ±1, 0^6) with even number of negative signs
    roots = set()
    base = [1,1,0,0,0,0,0,0]
    for signs in itertools.product([1,-1], repeat=2):
        if (signs.count(-1) % 2) != 0:
            continue
        vec = [signs[0], signs[1]] + [0]*6
        for perm in set(itertools.permutations(vec, 8)):
            roots.add(perm)
    return list(roots)  # length 112

def _roots_type2():
    # 128 roots: all 8-tuples with entries ±1/2, even number of negatives
    roots = []
    for signs in itertools.product([1,-1], repeat=8):
        if (signs.count(-1) % 2) != 0:
            continue
        roots.append(tuple(s*0.5 for s in signs))
    return roots  # length 128

def e8_roots() -> List[Tuple[float,...]]:
    # Generate full E8 root system (length 240), not normalized
    r1 = _roots_type1()
    r2 = _roots_type2()
    roots = r1 + r2
    assert len(roots) == 240
    return roots

def normalize(v):
    n = math.sqrt(sum(x*x for x in v)) or 1.0
    return tuple(x/n for x in v)

def normalized_e8_roots() -> List[Tuple[float,...]]:
    return [normalize(r) for r in e8_roots()]

def chamber_signature(v) -> str:
    # Simple sign-pattern chamber proxy (not a formal Weyl chamber)
    return "".join("+" if x>=0 else "-" for x in v)

