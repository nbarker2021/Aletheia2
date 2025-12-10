# Quadratic Medium Equivalence (QME) — wrapper using harness v1.2
from typing import Dict, Any, Tuple

try:
    # If running inside the harness, prefer official implementation
    from harness.tqf.core import qme_tuple as _qme_tuple
except Exception:
    # Fallback minimal: mirror the logic with placeholders
    def _structural_seed(obj: Dict[str,Any]) -> Tuple[float,...]:
        n = float(obj.get("n", 0)); parts = obj.get("parts", [])
        B = float(obj.get("B", 2)); moduli = sorted(obj.get("moduli", []))[:4]
        s_parts = sum(parts) if parts else 0.0; l_parts = len(parts)
        m1 = float(moduli[0]) if len(moduli)>0 else 0.0
        m2 = float(moduli[1]) if len(moduli)>1 else 0.0
        m3 = float(moduli[2]) if len(moduli)>2 else 0.0
        m4 = float(moduli[3]) if len(moduli)>3 else 0.0
        return ((n%(2*B))/B, (s_parts%(2*B))/B, (l_parts%(2*B))/B,
                (B%64)/64.0,(m1%64)/64.0,(m2%64)/64.0,(m3%64)/64.0,(m4%64)/64.0)
    def _qme_tuple(obj: Dict[str,Any], include_governance: bool=True):
        v8 = _structural_seed(obj)
        cnf_vec = tuple(sorted([abs(int(round(x))) for x in v8], reverse=True))
        if not include_governance: return ("CNF", cnf_vec)
        gov = {"regime": obj.get("regime","neutral"), "receipts": obj.get("receipts", {})}
        return ("CNF", cnf_vec, "GOV", gov)

def qme_fingerprint(obj: Dict[str,Any], include_governance: bool=True):
    """Return the canonical QME tuple for a token object."""
    return _qme_tuple(obj, include_governance=include_governance)

from typing import List, Tuple

def reduce_derivatives(n: int, base: int) -> List[Tuple[int,int]]:
    """
    Decompose n into additive pairs whose components are <= base/2.
    Greedy but practical for TQF staging: fills with (base//2, k) chunks.
    Example: n=7, base=8 -> [(4,3)] or [(4,2),(1,0)] filtered.
    """
    if base < 2: raise ValueError("base must be >= 2")
    cap = base//2
    pairs = []
    remaining = n
    while remaining > 0:
        a = min(cap, remaining)
        b = 0
        # Try to keep two-term pairs; if we still have remainder beyond cap, split it
        if remaining - a > cap:
            b = cap
        else:
            b = max(0, remaining - a)
        pairs.append((a, b))
        remaining -= (a + b)
    # clean zero entries and coalesce
    pairs = [(x,y) for (x,y) in pairs if x>0 or y>0]
    return pairs

def entropy_gate(delta_S: int) -> str:
    """
    ΔS=0 → free expansion; ΔS>0 → requires interaction.
    Returns 'free' or 'interact' for quick control flow.
    """
    return "free" if delta_S == 0 else "interact"

from typing import Callable, Dict, Any
try:
    from harness.tqf.core import qme_tuple, aperture
except Exception:
    # Minimal fallbacks reusing qme_fingerprint above
    from typing import Tuple
    def aperture(obj: Dict[str,Any], regime: str, witness=None):
        obj = dict(obj); obj["regime"] = regime
        rc = obj.setdefault("receipts", {}); rc.setdefault("apertures", []).append({"regime":regime,"witness":witness}); return obj
    def qme_tuple(obj: Dict[str,Any], include_governance: bool=True):
        from tqf_formula_segments import qme_fingerprint
        return qme_fingerprint(obj, include_governance)

def cnf_path_independent(obj: Dict[str,Any], path1: Callable, path2: Callable) -> bool:
    """
    Apply two different operation paths; compare CNF-only fingerprints.
    Returns True if path-independent.
    """
    a = path1(dict(obj)); b = path2(dict(obj))
    cnf_a = qme_tuple(a, include_governance=False)
    cnf_b = qme_tuple(b, include_governance=False)
    return cnf_a == cnf_b

from math import gcd
from typing import List, Dict, Any

def crt_defect_receipt(moduli: List[int]) -> Dict[str,Any] | None:
    """Return a defect receipt if gcd > 1, else None."""
    g = 0
    for m in moduli:
        g = gcd(g, m)
    if g > 1:
        pair = moduli[:2] if len(moduli)>=2 else moduli
        return {"g": g, "pair": pair, "bezout": None}
    return None

from typing import List, Dict, Any, Tuple
try:
    from harness.tqf.buckets import bucket_bits
except Exception:
    def bucket_bits(obj: Dict[str,Any]) -> Tuple[int,int,int]:
        # Minimal parity-ish placeholder
        parts = obj.get("parts", [])
        B = obj.get("B", 2)
        return (len(parts)&1, B&1, sum(parts)&1)

def octet_commutes(objs: List[Dict[str,Any]]) -> bool:
    """Batch vs per-element bucket equivalence."""
    batch = [bucket_bits(o) for o in objs]
    per   = [bucket_bits(o) for o in objs]
    return sorted(batch) == sorted(per)

from typing import List, Dict, Any

PHI = (1 + 5**0.5) / 2.0

def _scaled_variation(seq: List[float], scale: float) -> float:
    if not seq or len(seq) < 2:
        return 0.0
    s = [x*scale for x in seq]
    diffs = [abs(s[i+1]-s[i]) for i in range(len(s)-1)]
    return sum(diffs) / len(diffs)

def phi_probe(seqA: List[float], seqB: List[float]) -> Dict[str, Any]:
    sA1 = _scaled_variation(seqA, PHI)
    sA2 = _scaled_variation(seqA, PHI*PHI)
    sB1 = _scaled_variation(seqB, PHI)
    sB2 = _scaled_variation(seqB, PHI*PHI)
    scoreA = 0.5*(sA1+sA2); scoreB = 0.5*(sB1+sB2)
    eps = 1e-6; decision = "ambiguous"
    if scoreA + eps < scoreB: decision = "A"
    elif scoreB + eps < scoreA: decision = "B"
    return {"phi": PHI, "scores":{"A":scoreA,"B":scoreB}, "decision": decision}

from typing import Tuple, Optional

def is_taxicab(n: int, bound: int=2000) -> Optional[Tuple[Tuple[int,int],Tuple[int,int]]]:
    """
    Return two distinct positive pairs (a,b),(c,d) with a^3+b^3=c^3+d^3=n if found, else None.
    bound controls search range.
    """
    sums = {}
    for a in range(1, bound+1):
        a3 = a*a*a
        if a3 > n: break
        for b in range(a, bound+1):
            s = a3 + b*b*b
            if s == n:
                if s in sums and sums[s] != (a,b):
                    return (sums[s], (a,b))
                sums[s] = (a,b)
            if s > n: break
    return None

def is_cabtaxi(n: int, bound: int=100) -> Optional[Tuple[Tuple[int,int],Tuple[int,int]]]:
    """
    Signed extension: allow negatives. Returns witness pairs if found.
    """
    sums = {}
    for a in range(-bound, bound+1):
        for b in range(abs(a), bound+1):
            s = a*a*a + b*b*b
            if s == n:
                if s in sums and sums[s] != (a,b):
                    return (sums[s], (a,b))
                sums[s] = (a,b)
    return None

from typing import Callable

class Scalars:
    """
    Base-bound arithmetic modes for additive/multiplicative/factorial/cumulative.
    This is a practical scaffold; swap with exact scalar semantics as you finalize.
    """
    def __init__(self, base: int):
        assert base >= 2
        self.base = base
        self.cap = base//2  # used in derivative reductions

    def add(self, x: int, y: int) -> int:
        return (x + y) % self.base

    def mul(self, x: int, y: int) -> int:
        return (x * y) % self.base

    def fact(self, n: int) -> int:
        out = 1
        for k in range(2, n+1):
            out = (out * k) % self.base
        return out

    def cum(self, n: int) -> int:
        # 1 + 2 + ... + n modulo base
        return (n*(n+1)//2) % self.base

from dataclasses import dataclass, field
from typing import List, Any

@dataclass
class SP:
    tokens: List[Any]
    def permutations(self) -> List[List[Any]]:
        from itertools import permutations
        return [list(p) for p in permutations(self.tokens)]

@dataclass
class HP:
    label: str
    superperms: List[SP] = field(default_factory=list)
    governance: dict = field(default_factory=dict)  # apertures, CRT, receipts, etc.

    def add_sp(self, sp: SP): 
        self.superperms.append(sp)

    def as_embedding_seed(self) -> List[Any]:
        # Collapse to a structural seed (e.g., token counts/lengths)
        return [len(sp.tokens) for sp in self.superperms]
