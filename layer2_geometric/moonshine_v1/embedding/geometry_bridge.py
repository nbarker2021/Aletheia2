
from typing import List, Tuple
import math

Vec = Tuple[float, float]

def centroid(ps: List[Vec]) -> Vec:
    n = max(1, len(ps))
    return (sum(p[0] for p in ps)/n, sum(p[1] for p in ps)/n)

def v_sub(a: Vec, b: Vec) -> Vec: return (a[0]-b[0], a[1]-b[1])
def v_norm(a: Vec) -> float: return math.hypot(a[0], a[1])
def angle(a: Vec) -> float: return math.atan2(a[1], a[0])

def radial_angle_hist(pts: List[Vec], rbins=16, abins=16) -> list:
    if not pts: return [0.0]*(rbins+abins+4)
    c = centroid(pts)
    rs, ths = [], []
    for p in pts:
        d = v_sub(p, c)
        rs.append(v_norm(d))
        ths.append((angle(d)%(2*math.pi)))
    R = max(1e-9, max(rs))
    rh = [0]*rbins; ah = [0]*abins
    for r, th in zip(rs, ths):
        ri = min(rbins-1, int(rbins * (r / R)))
        ai = min(abins-1, int(abins * (th /(2*math.pi))))
        rh[ri] += 1; ah[ai] += 1
    rh = [x/len(pts) for x in rh]
    ah = [x/len(pts) for x in ah]
    return rh + ah + [float(len(pts)), R, sum(rs)/len(rs), 0.0]
