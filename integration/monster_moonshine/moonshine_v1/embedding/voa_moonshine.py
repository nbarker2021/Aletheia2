
from typing import List, Dict, Any

J_COEFFS = {
    -1: 1,
    0: 0,
    1: 196884,
    2: 21493760,
    3: 864299970,
    4: 20245856256,
}

MT_1A = [1, 196884, 21493760, 864299970]
MT_2A = [1, 4372, 96256, 1240002]
MT_3A = [1, 783, 8672, 65367]

def pad(v: List[float], n: int) -> List[float]:
    return v + [0.0]*(n-len(v)) if len(v) < n else v[:n]

def moonshine_feature(dim: int=32) -> List[float]:
    a = pad([float(J_COEFFS.get(i, 0)) for i in range(-1, 8)], 10)
    b = pad([float(x) for x in MT_1A], 10)
    c = pad([float(x) for x in MT_2A], 10)
    d = pad([float(x) for x in MT_3A], 10)
    def scale(v):
        m = (max(v) or 1.0)
        return [x/m for x in v]
    feat = scale(a) + scale(b) + scale(c) + scale(d)
    return feat[:dim] if len(feat) >= dim else feat + [0.0]*(dim-len(feat))

def fuse(features: Dict[str, Any]) -> List[float]:
    out: List[float] = []
    for k in sorted(features.keys()):
        v = features[k]
        out.extend(v)
    return out
