
from typing import List, Tuple, Dict
import math, hashlib, struct, random

# Minimal E8 root subset (demo). Real system should load full 240 roots.
E8_ROOTS = [
    # Simple orthonormal-like demo roots (NOT actual E8 full set!) — placeholder for nearest-root demo
    (1, -1, 0, 0, 0, 0, 0, 0),
    (0, 1, -1, 0, 0, 0, 0, 0),
    (0, 0, 1, -1, 0, 0, 0, 0),
    (0, 0, 0, 1, -1, 0, 0, 0),
    (0, 0, 0, 0, 1, -1, 0, 0),
    (0, 0, 0, 0, 0, 1, -1, 0),
    (0, 0, 0, 0, 0, 0, 1, -1),
    (-1, 0, 0, 0, 0, 0, 0, 1),
]

def _hash_to_8floats(s: str, seed: int=0) -> List[float]:
    h = hashlib.sha256((s + f"|{seed}").encode("utf-8")).digest()
    # unpack 8 doubles from 64 bytes via 8 chunks of 8 bytes (little endian IEEE754-ish using ints→normalize)
    vals = []
    for i in range(0, 64, 8):
        chunk = h[i:i+8]
        x = int.from_bytes(chunk, "little") / (2**64 - 1)
        vals.append(2.0 * x - 1.0)  # map to [-1, 1]
    return vals[:8]

def _normalize(v: List[float]) -> List[float]:
    n = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x/n for x in v]

def nearest_root(v: List[float]) -> Tuple[Tuple[int, ...], float]:
    # cosine similarity as proxy → distance = 1 - cos
    def cos(a, b):
        na = math.sqrt(sum(x*x for x in a)) or 1.0
        nb = math.sqrt(sum(x*x for x in b)) or 1.0
        dot = sum(x*y for x,y in zip(a,b))
        return dot/(na*nb)
    best = None
    best_d = 1e9
    for r in E8_ROOTS:
        c = cos(v, r)
        d = 1.0 - c
        if d < best_d:
            best = r; best_d = d
    return best, best_d

def embed_string(s: str, seed: int=0) -> Dict:
    v = _hash_to_8floats(s, seed)
    v = _normalize(v)
    root, dist = nearest_root(v)
    return {"vector": v, "nearest_root": root, "root_distance": dist}


def cosine(a, b):
    na = math.sqrt(sum(x*x for x in a)) or 1.0
    nb = math.sqrt(sum(x*x for x in b)) or 1.0
    return sum(x*y for x,y in zip(a,b)) / (na*nb)
