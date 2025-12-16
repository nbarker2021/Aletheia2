
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import math

@dataclass
class ContextScore:
    name: str
    score: float  # 0..1
    evidence: str = ""

def _agg(values: List[float], method: str = "mean", weights: Dict[str,float] | None = None, names: List[str] | None = None) -> float:
    if not values:
        return 0.0
    if method == "mean":
        return sum(values)/len(values)
    elif method == "harmonic":
        eps = 1e-9
        return len(values) / sum((1.0/(v+eps)) for v in values)
    elif method == "geometric":
        eps = 1e-9
        s = 1.0
        for v in values:
            s *= max(v, eps)
        return s ** (1.0/len(values))
    elif method == "weighted_mean" and weights and names:
        tot_w = 0.0
        acc = 0.0
        for v, n in zip(values, names):
            w = weights.get(n, 0.0)
            tot_w += w
            acc += w * v
        return acc / tot_w if tot_w > 0 else sum(values)/len(values)
    return sum(values)/len(values)

def w5h_aggregate(beacon: dict) -> Dict[str, float]:
    """Return per-dimension and final aggregate score according to policy."""
    w5h = beacon["w5h"]
    policy = beacon.get("policy", {})
    method = policy.get("aggregation", "mean")
    weights = policy.get("weights", {})
    priority = policy.get("priority_contexts", [])

    def dim_score(dim: str) -> float:
        ctxs = w5h[dim]["contexts"]
        vals = [float(c["score"]) for c in ctxs]
        names = [c["name"] for c in ctxs]
        return _agg(vals, method, weights, names)

    dims = ["who","what","where","when","why","how"]
    per_dim = {d: dim_score(d) for d in dims}

    # Final score: aggregate chosen priority contexts when present, else aggregate per-dim
    if priority:
        # Map priority names to find them inside contexts across dims
        collected = []
        for d in dims:
            for c in w5h[d]["contexts"]:
                if c["name"] in priority:
                    collected.append((c["name"], float(c["score"])))
        if collected:
            names = [n for n,_ in collected]
            vals = [v for _,v in collected]
            final = _agg(vals, method, weights, names)
        else:
            final = _agg(list(per_dim.values()), method)
    else:
        final = _agg(list(per_dim.values()), method)

    return {"final": final, **per_dim}
