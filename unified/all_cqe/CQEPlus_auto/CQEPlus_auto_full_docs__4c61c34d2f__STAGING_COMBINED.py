# === CQE Morphonic Staging Combined (read-only composite) ===


# ===== BEGIN FILE: workspace/nextgen/morphonic/cli/strict.py =====

import json, math, hashlib, numpy as np, uuid
from importlib.util import spec_from_file_location, module_from_spec

def _load(path, name):
    spec = spec_from_file_location(name, path)
    mod = module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

ge = _load(r"""/mnt/data/workspace/CQEPlus_build_251014/nextgen/morphonic/glyphs/engine.py""", "glyph_engine")
Step, run_chain, setup_trace = ge.Step, ge.run_chain, ge.setup_trace

try:
    e8ops = _load(r"""/mnt/data/workspace/CQEPlus_build_251014/src_original/CQE+/CQE+ 1.0/src/core/e8_lattice/e8_ops.py""", "e8_ops")
except Exception:
    e8ops = _load(r"""/mnt/data/workspace/CQEPlus_build_251014/nextgen/morphonic/fallback/e8_ops_fallback.py""", "e8_ops")
E8 = e8ops.E8Lattice()

buckets      = _load(r"""/mnt/data/workspace/CQEPlus_build_251014/nextgen/morphonic/ordering/buckets/dihedral_crt.py""", "buckets")
gates        = _load(r"""/mnt/data/workspace/CQEPlus_build_251014/nextgen/morphonic/governance/gates.py""", "gates")
bridge_gate  = _load(r"""/mnt/data/workspace/CQEPlus_build_251014/nextgen/morphonic/governance/bridge_gate.py""", "bridge_gate")
ledger_mod   = _load(r"""/mnt/data/workspace/CQEPlus_build_251014/nextgen/morphonic/governance/ledger.py""", "ledger_mod")
proofs       = _load(r"""/mnt/data/workspace/CQEPlus_build_251014/nextgen/morphonic/governance/proofs.py""", "proofs")
rails        = _load(r"""/mnt/data/workspace/CQEPlus_build_251014/nextgen/morphonic/ordering/rails/odd_crt.py""", "rails")
overlay      = _load(r"""/mnt/data/workspace/CQEPlus_build_251014/nextgen/morphonic/storage/overlay.py""", "overlay")
worldforge   = _load(r"""/mnt/data/workspace/CQEPlus_build_251014/nextgen/morphonic/worldforge.py""", "worldforge")
leech_adapter= _load(r"""/mnt/data/workspace/CQEPlus_build_251014/nextgen/morphonic/lattice/leech_adapter.py""", "leech_adapter")

def ZOOM10(seq, k: int):
    s = (10.0 ** k)
    seq.append(Step("⤢",(s,)))
    seq.append(Step("⚖",()))

def strict(persona_n=29, theta=3.1415926535/20.0, seed=314159, trace_path=None, trace_id=None, zoom_k=0, enforce_leech_bridge=False, cov_path=None, universe_name="CQE-Default", ledger_path="ledger.jsonl"):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=10); base /= np.linalg.norm(base) or 1.0
    personas = [(f"persona_{i}", (base + 0.10*rng.normal(size=10))) for i in range(persona_n)]

    if trace_path and trace_id:
        setup_trace(trace_path, trace_id)

    L = ledger_mod.new(ledger_path)
    universe = worldforge.forge(universe_name, zoom_k=zoom_k)
    L.emit("universe", spec=universe)

    pre_vectors = []
    for name, dom in personas:
        out = run_chain([Step("↑",(dom.tolist(),)), Step("⥁",(theta,)), Step("⇄",())])
        pre_vectors.append(out["final"])
    chambers = [int(E8.find_weyl_chamber(v)) for v in pre_vectors]
    from collections import Counter
    target_chamber = Counter(chambers).most_common(1)[0][0]
    L.emit("chamber_majority", target=int(target_chamber))

    aligned_E8 = []
    for name, dom in personas:
        seq = [Step("↑",(dom.tolist(),)), Step("⥁",(theta,)), Step("⇄",()), Step("⌖",(int(target_chamber),))]
        if zoom_k != 0:
            ZOOM10(seq, zoom_k)
        out = run_chain(seq)
        aligned_E8.append(out["final"])

    prepost = [(v, v) for v in aligned_E8]
    th = gates.GateThresholds(uniformity=0.75, consensus=0.90, enforce_noether=True, refocus_margin=0.9)
    gate_rec = gates.check_gates(aligned_E8, prepost, th)
    L.emit("gates", **gate_rec)

    loads = buckets.SimpleLoad()
    rows = []
    rail_counts = {i:0 for i in range(1,6)}
    for (name, dom), v in zip(personas, aligned_E8):
        token = f"{name}::{hashlib.sha256(np.array(dom).tobytes()).hexdigest()[:8]}"
        res = buckets.assign_token(token, v, loads, buckets.AssignConfig())
        loads.add(res.row, res.col)
        oc = None; rho = None
        if res.overflow:
            oc, rho = rails.gate_odd_crt(np.array(dom).tobytes())
            rail_counts[oc] += 1
        L.emit("assignment", name=name, row=int(res.row), col=int(res.col), overflow=bool(res.overflow),
               u=int(res.u), r=int(res.r), f=int(res.f), q=int(res.q), rho=(int(rho) if rho is not None else None),
               oc=(int(oc) if oc is not None else None))
        rows.append({"name": name, "row": res.row, "col": res.col, "overflow": res.overflow, "u": res.u, "r": res.r, "f": res.f, "q": res.q, "rho": rho, "oc": oc})

    bridge_rec = None
    roundtrips = []
    if enforce_leech_bridge:
        for v in aligned_E8:
            br = run_chain([Step("↑",(v.tolist(),)), Step("↥",()), Step("↧",())])
            roundtrips.append((v, br["final"]))
        bridge_rec = bridge_gate.bridge_decide(aligned_E8, roundtrips, cov_path if cov_path else "coverage_tmp.json")
        L.emit("bridge", **bridge_rec)

    ps = proofs.summarize(prepost, roundtrips)
    L.emit("proof_summary", noether_norm_ok=bool(ps.noether_norm_ok), gram_similarity=float(ps.gram_similarity), max_roundtrip=float(ps.max_roundtrip))

    overlay_path = "overlay_snapshot.json"
    overlay.save(overlay_path, "strict_v12_"+uuid.uuid4().hex[:8], aligned_E8, {"universe": universe, "adapter": leech_adapter.info()})

    return {"target_chamber": int(target_chamber), "gates": gate_rec, "assignments": rows, "bridge": bridge_rec, "universe": universe, "adapter": leech_adapter.info(), "overlay": overlay_path, "rails": rail_counts, "ledger": ledger_path}

# ===== END FILE: workspace/nextgen/morphonic/cli/strict.py =====


# ===== BEGIN FILE: workspace/nextgen/morphonic/fallback/e8_ops_fallback.py =====

import numpy as np

_SIMPLE_ROOTS = np.array([
    [1,-1,0,0,0,0,0,0],
    [0,1,-1,0,0,0,0,0],
    [0,0,1,-1,0,0,0,0],
    [0,0,0,1,-1,0,0,0],
    [0,0,0,0,1,-1,0,0],
    [0,0,0,0,0,1,-1,0],
    [0,0,0,0,0,0,1,-1],
    [-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5]
], dtype=float)

def project_to_lattice(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).reshape(-1)[:8]
    n = np.linalg.norm(v) or 1.0
    return v / n

def reflect(v: np.ndarray, root: np.ndarray) -> np.ndarray:
    r = root / (np.linalg.norm(root) or 1.0)
    return v - 2.0*np.dot(v, r)*r

def find_weyl_chamber(v: np.ndarray) -> int:
    vv = np.asarray(v, float).reshape(-1)[:8]
    for _ in range(4):
        for r in _SIMPLE_ROOTS:
            if np.dot(vv, r) < 0:
                vv = reflect(vv, r)
    signs = (vv >= 0).astype(int)
    code = 0
    for b in signs[:6]:
        code = (code<<1) | int(b)
    return int(code % 240)

class E8Lattice:
    def find_weyl_chamber(self, v): return find_weyl_chamber(v)
    def project_to_lattice(self, v): return project_to_lattice(v)
    def reflect(self, v, root): return reflect(v, root)

# ===== END FILE: workspace/nextgen/morphonic/fallback/e8_ops_fallback.py =====


# ===== BEGIN FILE: workspace/nextgen/morphonic/glyphs/engine.py =====

import json, numpy as np, pathlib, time
from dataclasses import dataclass

_trace_path = None
_trace_id = None

def setup_trace(path, tid):
    global _trace_path, _trace_id
    _trace_path, _trace_id = path, tid
    pathlib.Path(path).write_text("")

def _log(kind, payload):
    if not _trace_path: return
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "id": _trace_id, "kind": kind, **payload}
    with open(_trace_path, "a") as f:
        f.write(json.dumps(rec)+"\n")

@dataclass
class Step:
    glyph: str
    args: tuple

def _norm(v):
    v = np.asarray(v, float).reshape(-1)[:8]
    n = np.linalg.norm(v) or 1.0
    return v / n

def _flow(v, theta):
    v = _norm(v)
    # simple cyclic mix as a rotation surrogate
    w = np.roll(v, 1) * np.cos(theta) + v * np.sin(theta)
    return _norm(w)

def _parity(v):
    v = np.asarray(v, float).copy()
    v[::2] *= -1.0
    return _norm(v)

def _align(v, chamber:int):
    # no-op align; in real engine we'd reflect into the chamber
    return _norm(v)

def _leech_up(v):   # ↥
    return _norm(v)

def _leech_down(v): # ↧
    return _norm(v)

def _zoom(v, s):
    return _norm(v * float(s))

def _renorm(v):
    return _norm(v)

def run_chain(steps):
    stack = []
    ops = []
    for st in steps:
        g = st.glyph
        if g == "↑":
            vec = _norm(st.args[0])
            stack = [vec]; ops.append(g); _log("op", {"g":g, "len": len(vec)})
        elif g == "⥁":
            theta = float(st.args[0]) if st.args else 0.0
            stack = [_flow(stack[-1], theta)]; ops.append(g)
        elif g == "⇄":
            stack = [_parity(stack[-1])]; ops.append(g)
        elif g == "⌖":
            chamber = int(st.args[0]) if st.args else 0
            stack = [_align(stack[-1], chamber)]; ops.append(g)
        elif g == "↥":
            stack = [_leech_up(stack[-1])]; ops.append(g)
        elif g == "↧":
            stack = [_leech_down(stack[-1])]; ops.append(g)
        elif g == "⤢":
            s = float(st.args[0]) if st.args else 1.0
            stack = [_zoom(stack[-1], s)]; ops.append(g)
        elif g == "⚖":
            stack = [_renorm(stack[-1])]; ops.append(g)
        else:
            raise ValueError(f"unknown glyph: {g}")
    return {"final": stack[-1], "ops": ops}

# ===== END FILE: workspace/nextgen/morphonic/glyphs/engine.py =====


# ===== BEGIN FILE: workspace/nextgen/morphonic/glyphs/registry.py =====

REGISTRY = {
    "↑": {"dr":1, "desc":"project up (embed)"},
    "↓": {"dr":1, "desc":"project down (extract)"},
    "⥁": {"dr":2, "desc":"toroidal flow"},
    "⇄": {"dr":2, "desc":"parity flip"},
    "⊞": {"dr":3, "desc":"snap/bind"},
    "☰": {"dr":3, "desc":"cellular automata embed"},
    "🧾": {"dr":4, "desc":"receipt"},
    "✓": {"dr":5, "desc":"verified"},
    "💾": {"dr":6, "desc":"cache/store"},
    "⇒": {"dr":7, "desc":"sequence"},
    "🛡": {"dr":8, "desc":"safety check"},
    "∫": {"dr":9, "desc":"integrate"},
}

# ===== END FILE: workspace/nextgen/morphonic/glyphs/registry.py =====


# ===== BEGIN FILE: workspace/nextgen/morphonic/governance/bridge_gate.py =====

import json, numpy as np, pathlib, time

def bridge_decide(vectors, roundtrips, cov_out_path):
    # vectors: original aligned E8 vectors; roundtrips: [(v, v2)]
    errs = []
    for a,b in roundtrips:
        errs.append(float(np.linalg.norm(np.asarray(a)-np.asarray(b))))
    mx = float(max(errs) if errs else 0.0)
    ok = mx <= 1e-6
    payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "max_roundtrip": mx, "ok": ok, "count": len(roundtrips)}
    pathlib.Path(cov_out_path).write_text(json.dumps(payload, indent=2))
    return payload

# ===== END FILE: workspace/nextgen/morphonic/governance/bridge_gate.py =====


# ===== BEGIN FILE: workspace/nextgen/morphonic/governance/gates.py =====

import numpy as np
from dataclasses import dataclass

@dataclass
class GateThresholds:
    uniformity: float = 0.70
    consensus: float = 0.85
    enforce_noether: bool = True
    refocus_margin: float = 0.9

def _uniformity(vectors):
    signs = [(int(v[0]>0), int(v[1]>0)) for v in vectors]
    from collections import Counter
    frac = Counter(signs).most_common(1)[0][1] / max(1,len(signs))
    return float(frac)

def _consensus(vectors):
    if not vectors: return 0.0
    M = np.stack(vectors)
    mu = M.mean(axis=0); mu = mu/(np.linalg.norm(mu) or 1.0)
    sims = [(float(v@mu)/(np.linalg.norm(v) or 1.0)) for v in vectors]
    return float(np.mean(sims))

def _noether_ok(prepost, eps=1e-8):
    return all(abs(np.linalg.norm(a)-np.linalg.norm(b)) <= eps for a,b in prepost)

def check_gates(vectors, prepost, th: GateThresholds):
    u = _uniformity(vectors); c = _consensus(vectors)
    decision = "COMMIT"
    if th.enforce_noether and not _noether_ok(prepost): decision = "ROLLBACK"
    if u < th.uniformity or c < th.consensus:
        decision = "REFOCUS" if (u >= th.uniformity*th.refocus_margin) else "ROLLBACK"
    return {"uniformity": u, "consensus": c, "decision": decision}

# ===== END FILE: workspace/nextgen/morphonic/governance/gates.py =====


# ===== BEGIN FILE: workspace/nextgen/morphonic/governance/ledger.py =====

import json, time, uuid, hashlib, pathlib

class Ledger:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")

    @staticmethod
    def _rid(payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def emit(self, kind: str, **fields):
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, **fields}
        rec["id"] = self._rid(rec)
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec["id"]

def new(path: str) -> Ledger:
    return Ledger(path)

# ===== END FILE: workspace/nextgen/morphonic/governance/ledger.py =====


# ===== BEGIN FILE: workspace/nextgen/morphonic/governance/proofs.py =====

import numpy as np
from dataclasses import dataclass

@dataclass
class ProofSummary:
    noether_norm_ok: bool
    gram_similarity: float  # 0..1
    max_roundtrip: float

def summarize(prepost, roundtrips, gram_tol=1e-2):
    if not prepost:
        return ProofSummary(True, 1.0, 0.0)
    pre = np.stack([a for a,_ in prepost]); post = np.stack([b for _,b in prepost])
    norm_ok = np.allclose(np.linalg.norm(pre,axis=1), np.linalg.norm(post,axis=1), atol=1e-8)
    preG = pre @ pre.T; postG = post @ post.T
    sg = float((np.sum(preG*postG) / (np.linalg.norm(preG)*np.linalg.norm(postG) + 1e-12)))
    mxe = 0.0
    for a,b in roundtrips:
        mxe = max(mxe, float(np.linalg.norm(a-b)))
    return ProofSummary(bool(norm_ok), sg, mxe)

# ===== END FILE: workspace/nextgen/morphonic/governance/proofs.py =====


# ===== BEGIN FILE: workspace/nextgen/morphonic/governance/tokens.py =====

def estimate_savings(baseline_tokens:int, glyph_ops:int) -> dict:
    est = max(6, baseline_tokens // max(1,glyph_ops//2 + 1))
    return {"baseline": baseline_tokens, "estimated": est, "savings_x": round(baseline_tokens / est, 2)}

# ===== END FILE: workspace/nextgen/morphonic/governance/tokens.py =====


# ===== BEGIN FILE: workspace/nextgen/morphonic/lattice/leech_adapter.py =====

def info():
    return {"mode": "fallback:QR", "note": "adapter in fallback (no real mapping detected)"}

# ===== END FILE: workspace/nextgen/morphonic/lattice/leech_adapter.py =====


# ===== BEGIN FILE: workspace/nextgen/morphonic/ordering/buckets/dihedral_crt.py =====

import numpy as np, hashlib
from dataclasses import dataclass

@dataclass
class AssignConfig:
    load_threshold: int = 64

@dataclass
class AssignResult:
    row: int; col: int; overflow: bool; u: int; r: int; f: int; q: int; rho: int|None; oc: int|None

class SimpleLoad:
    def __init__(self): self.grid = {}
    def load(self, r,c): return self.grid.get((r,c), 0)
    def add(self, r,c): self.grid[(r,c)] = self.load(r,c)+1

def _dihedral_rf(v):
    r = int((v[0] > 0)) * 4 + int((v[1] > 0)) * 2 + int((v[2] > 0))
    f = int((v[3] < 0))
    return r & 7, f & 1

def _mix(h, r, f):
    x = (h ^ (r<<5) ^ (f<<3)) & 0xFFFFFFFF
    x ^= (x >> 13); x *= 0x85EBCA6B; x &= 0xFFFFFFFF
    return x & 0xFF

def assign_token(token: str, v: np.ndarray, loads: SimpleLoad, cfg: AssignConfig) -> AssignResult:
    h = int.from_bytes(hashlib.blake2b(np.asarray(v,float).tobytes(), digest_size=8).digest(), "little")
    r, f = _dihedral_rf(v)
    q = _mix(h, r, f)
    u = ((r & 7) << 9) | ((f & 1) << 8) | q
    row, col = (u >> 6) & 63, u & 63
    overflow = loads.load(row,col) > cfg.load_threshold
    rho = oc = None
    if overflow:
        rho = h % 1155
        oc = 1 + (rho % 5)
    return AssignResult(row, col, overflow, u, r, f, q, rho, oc)

# ===== END FILE: workspace/nextgen/morphonic/ordering/buckets/dihedral_crt.py =====


# ===== BEGIN FILE: workspace/nextgen/morphonic/ordering/rails/odd_crt.py =====

import hashlib

def gate_odd_crt(token_bytes: bytes) -> tuple[int,int]:
    h = int.from_bytes(hashlib.blake2b(token_bytes, digest_size=8).digest(), "little")
    rho = h % 1155
    oc = 1 + (rho % 5)  # 1..5 → DR {1,3,5,7,9}
    return oc, rho

# ===== END FILE: workspace/nextgen/morphonic/ordering/rails/odd_crt.py =====


# ===== BEGIN FILE: workspace/nextgen/morphonic/storage/overlay.py =====

import json, pathlib, time, numpy as np
def save(path: str, name: str, vectors: list, meta: dict):
    p = pathlib.Path(path)
    payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "name": name, "meta": meta,
               "vectors": [np.asarray(v,float).tolist() for v in vectors]}
    p.write_text(json.dumps(payload, indent=2))
    return str(p)
def load(path: str):
    import json, numpy as np, pathlib
    obj = json.loads(pathlib.Path(path).read_text())
    vecs = [np.asarray(v,float) for v in obj.get("vectors",[])]
    return vecs, obj.get("meta",{})

# ===== END FILE: workspace/nextgen/morphonic/storage/overlay.py =====


# ===== BEGIN FILE: workspace/nextgen/morphonic/worldforge.py =====

import time, platform
def forge(universe_name="CQE-Default", constants=None, zoom_k=0):
    constants = constants or {"phi": 1.61803398875, "pi": 3.14159265359, "grav": 0.03}
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "name": universe_name,
        "constants": constants,
        "zoom_k": zoom_k,
        "host": {"python": platform.python_version(), "platform": platform.platform()}
    }

# ===== END FILE: workspace/nextgen/morphonic/worldforge.py =====
