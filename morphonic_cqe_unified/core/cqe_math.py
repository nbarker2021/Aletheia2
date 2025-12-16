
import math
from typing import List, Tuple

Vector = Tuple[float, ...]

def _dot(a: Vector, b: Vector) -> float:
    return sum(x*y for x,y in zip(a,b))

def _add(a: Vector, b: Vector) -> Vector:
    return tuple(x+y for x,y in zip(a,b))

def _sub(a: Vector, b: Vector) -> Vector:
    return tuple(x-b for x,b in zip(a,b))

def _scale(a: Vector, s: float) -> Vector:
    return tuple(s*x for x in a)

def generate_e8_roots() -> List[Vector]:
    roots = []
    n = 8
    # Family 1: D8 roots (±1, ±1, 0^6), 112 roots
    for i in range(n):
        for j in range(i+1, n):
            for s1 in (+1.0, -1.0):
                for s2 in (+1.0, -1.0):
                    v = [0.0]*n
                    v[i] = s1
                    v[j] = s2
                    roots.append(tuple(v))
    # Family 2: (±1/2)^8 with even number of + (128 roots)
    from itertools import product
    for signs in product((-0.5, 0.5), repeat=8):
        plus = sum(1 for s in signs if s > 0)
        if plus % 2 == 0:
            roots.append(tuple(signs))
    assert len(roots) == 240, f"Expected 240 roots, got {len(roots)}"
    for r in roots:
        l2 = _dot(r, r)
        if abs(l2 - 2.0) > 1e-9:
            raise ValueError("Root has wrong length^2: {}".format(l2))
    return roots

def simple_roots_e8() -> List[Vector]:
    def e(i):
        v = [0.0]*8
        v[i] = 1.0
        return tuple(v)
    e1,e2,e3,e4,e5,e6,e7,e8 = e(0),e(1),e(2),e(3),e(4),e(5),e(6),e(7)
    a1 = tuple(0.5*(e1[i]+e8[i]) - 0.5*(e2[i]+e3[i]+e4[i]+e5[i]+e6[i]+e7[i]) for i in range(8))
    a2 = tuple(e1[i]+e2[i] for i in range(8))
    a3 = tuple(-e1[i]+e2[i] for i in range(8))
    a4 = tuple(-e2[i]+e3[i] for i in range(8))
    a5 = tuple(-e3[i]+e4[i] for i in range(8))
    a6 = tuple(-e4[i]+e5[i] for i in range(8))
    a7 = tuple(-e5[i]+e6[i] for i in range(8))
    a8 = tuple(-e6[i]+e7[i] for i in range(8))
    S = [a1,a2,a3,a4,a5,a6,a7,a8]
    for a in S:
        if abs(_dot(a,a) - 2.0) > 1e-9:
            raise ValueError("Simple root length^2 not 2")
    return S

def cartan_from_simple_roots(S: List[Vector]) -> List[List[float]]:
    C = []
    for ai in S:
        row = []
        for aj in S:
            num = 2.0 * _dot(ai, aj)
            den = _dot(aj, aj)
            row.append(num/den)
        C.append(row)
    return C

def reflect(v: Vector, root: Vector) -> Vector:
    denom = _dot(root, root)
    return _sub(v, _scale(root, 2.0*_dot(v, root)/denom))

def project_to_fundamental_chamber(v: Vector, S: List[Vector], max_iter: int = 1024, tol: float = 1e-12):
    cur = tuple(v)
    reflections = 0
    for _ in range(max_iter):
        alpha_dots = [sum(cur[i]*a[i] for i in range(8)) for a in S]
        if min(alpha_dots) >= -tol:
            return cur, alpha_dots, reflections, True
        i = min(range(len(S)), key=lambda k: alpha_dots[k])
        if alpha_dots[i] >= -tol:
            return cur, alpha_dots, reflections, True
        cur = reflect(cur, S[i])
        reflections += 1
    alpha_dots = [sum(cur[i]*a[i] for i in range(8)) for a in S]
    return cur, alpha_dots, reflections, False

def metric_A_from_cartan(C: List[List[float]], scale: float = 1.0) -> List[List[float]]:
    n = len(C)
    A = [[scale*C[i][j] for j in range(n)] for i in range(n)]
    return A

def phi(A: List[List[float]], x: Vector) -> float:
    n = len(A)
    y = [sum(A[i][j]*x[j] for j in range(n)) for i in range(n)]
    return sum(x[i]*y[i] for i in range(n))

def try_internal_step(A: List[List[float]], x: Vector, delta: Vector, max_backtracks: int = 20, shrink: float = 0.5):
    base = phi(A, x)
    s = 1.0
    for k in range(max_backtracks+1):
        trial = tuple(x[i] + s*delta[i] for i in range(8))
        dphi = phi(A, trial) - base
        if dphi <= 1e-12:
            return trial, True, k+1
        s *= shrink
    return x, False, max_backtracks+1

def l2_norm(x: Vector) -> float:
    return math.sqrt(sum(xi*xi for xi in x))
