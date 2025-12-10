
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coherence/Decoherence Metrics (pure stdlib)
-------------------------------------------
Geometry-first measures + embedding-alignment + dPhi guard.
This defines how we measure coherence, decoherence, and collapse events.
"""
from __future__ import annotations
import math, json
from typing import List, Tuple, Dict, Any

Vec = Tuple[float, float]

def centroid(ps: List[Vec]) -> Vec:
    n = max(1, len(ps))
    return (sum(p[0] for p in ps)/n, sum(p[1] for p in ps)/n)

def v_sub(a: Vec, b: Vec) -> Vec: return (a[0]-b[0], a[1]-b[1])
def v_norm(a: Vec) -> float: return math.hypot(a[0], a[1])
def angle(a: Vec) -> float: return math.atan2(a[1], a[0])

def angular_coherence(points: List[Vec]) -> float:
    """Circular statistic Rbar in [0,1]: 1 means perfect phase alignment."""
    if not points: return 0.0
    c = centroid(points)
    cs = 0.0; ss = 0.0; n = 0
    for p in points:
        d = v_sub(p, c)
        th = angle(d)
        cs += math.cos(th); ss += math.sin(th); n += 1
    if n == 0: return 0.0
    R = math.sqrt((cs/n)**2 + (ss/n)**2)
    return R

def radial_coherence(points: List[Vec]) -> float:
    """1 - Coefficient of variation of radii, clamped to [0,1]."""
    if not points: return 0.0
    c = centroid(points)
    rs = [v_norm(v_sub(p, c)) for p in points]
    mu = sum(rs)/len(rs)
    if mu == 0: return 1.0
    var = sum((r-mu)*(r-mu) for r in rs)/len(rs)
    cv = math.sqrt(var)/abs(mu)
    score = 1.0 - min(1.0, cv)
    return max(0.0, min(1.0, score))

def spectral_entropy(series: List[float]) -> float:
    """Normalized spectral entropy of a real series via naive DFT. Returns 0..1 (higher = more decoherence)."""
    n = len(series)
    if n == 0: return 0.0
    import cmath
    mags = []
    for k in range(n):
        s = 0j
        for t, x in enumerate(series):
            s += x * cmath.exp(-2j*math.pi*k*t/n)
        mags.append((s.real*s.real + s.imag*s.imag))
    total = sum(mags) or 1.0
    p = [m/total for m in mags]
    H = -sum(pi*math.log(pi+1e-12) for pi in p)
    Hmax = math.log(n) if n>0 else 1.0
    return float(H/Hmax) if Hmax>0 else 0.0

def embedding_alignment(a: List[float], b: List[float]) -> float:
    """Cosine similarity in [-1,1] mapped to [0,1]."""
    if not a or not b or len(a)!=len(b): return 0.0
    da = sum(x*x for x in a); db = sum(y*y for y in b)
    if da==0 or db==0: return 0.0
    dot = sum(x*y for x,y in zip(a,b))
    cos = dot / math.sqrt(da*db)
    return 0.5*(cos+1.0)

def delta_phi(before_points: List[Vec], after_points: List[Vec]) -> float:
    """Average squared displacement between two point sets (index-aligned)."""
    if not before_points or not after_points: return 0.0
    n = min(len(before_points), len(after_points))
    s = 0.0
    for i in range(n):
        dx = after_points[i][0]-before_points[i][0]
        dy = after_points[i][1]-before_points[i][1]
        s += dx*dx + dy*dy
    return s/max(1,n)

def composite_coherence(points: List[Vec]) -> Dict[str,float]:
    ac = angular_coherence(points)
    rc = radial_coherence(points)
    c = centroid(points)
    series = [v_norm(v_sub(p, c)) for p in points]
    se = spectral_entropy(series)
    se_score = 1.0 - se
    comp = 0.5*ac + 0.3*rc + 0.2*se_score
    return {"angular": ac, "radial": rc, "spectral_entropy": se, "score": comp}

def collapse_detector(prev_points: List[Vec], curr_points: List[Vec], *, thresh_drop=0.25, dphi_thresh=0.05) -> Dict[str,Any]:
    prev = composite_coherence(prev_points)
    curr = composite_coherence(curr_points)
    dscore = curr["score"] - prev["score"]
    dphi = delta_phi(prev_points, curr_points)
    collapsed = (dscore <= -thresh_drop) or (dphi <= dphi_thresh and curr["score"] < 0.3)
    reason = "score_drop" if dscore <= -thresh_drop else ("frozen_low_score" if dphi <= dphi_thresh and curr["score"]<0.3 else "no")
    return {"collapsed": bool(collapsed), "reason": reason, "delta_score": dscore, "dphi": dphi, "prev": prev, "curr": curr}
