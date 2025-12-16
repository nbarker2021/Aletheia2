
import math
from typing import Tuple
from cqe_math import Vector, l2_norm

def _rot2(x: float, y: float, theta: float) -> Tuple[float, float]:
    c, s = math.cos(theta), math.sin(theta)
    return c*x - s*y, s*x + c*y

def toroidal_step(x: Vector, base_coupling: float = 0.03, tol: float = 1e-10) -> Tuple[Vector, bool]:
    assert len(x) == 8, "toroidal_step expects 8D"
    xs = list(x)
    norm0 = l2_norm(x)
    for k, (i,j) in enumerate([(0,1),(2,3),(4,5),(6,7)]):
        theta = base_coupling * 2.0*math.pi * (k+1)
        xs[i], xs[j] = _rot2(xs[i], xs[j], theta)
    norm1 = l2_norm(tuple(xs))
    closed = abs(norm1 - norm0) <= tol
    return tuple(xs), closed
