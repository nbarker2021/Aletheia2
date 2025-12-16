
from typing import Dict, Any
import math, random

def digital_root(n: int) -> int:
    n = abs(n)
    while n >= 10:
        s = 0
        while n > 0:
            s += n % 10
            n //= 10
        n = s
    return n

def classify_rotational_pattern(dr: int) -> str:
    # Simple mapping demo (placeholder)
    if dr in (1,4,7): return "INWARD"
    if dr in (2,5,8): return "OUTWARD"
    if dr in (3,6,9): return "TRANSFORM"
    return "NEUTRAL"

def generate_toroidal_shell(n_points: int=256, R: float=1.0, r: float=0.3, seed: int=0):
    random.seed(seed)
    pts = []
    for i in range(n_points):
        theta = 2*math.pi*random.random()
        phi   = 2*math.pi*random.random()
        x = (R + r*math.cos(theta))*math.cos(phi)
        y = (R + r*math.cos(theta))*math.sin(phi)
        z = r*math.sin(theta)
        dr = digital_root(int(abs(x*1e3)+abs(y*1e3)+abs(z*1e3)) or 0)
        pattern = classify_rotational_pattern(dr)
        pts.append({"x":x,"y":y,"z":z,"digital_root":dr,"pattern":pattern})
    return pts
