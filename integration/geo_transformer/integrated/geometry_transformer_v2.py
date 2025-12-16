
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometry-Only Transformer — Standalone v2
=========================================
No third-party deps. Pure stdlib. Drop-in script you can run anywhere.

What it is:
  • A geometry-native "transformer" that uses only metric & angular relations
    (no token IDs, no text embeddings, no numpy).
  • Content-addressed, ledgered compute (GeoLight) with an append-only Merkle chain.
  • Channels {3,6,9} and ΔΦ guard hooks to mirror CQE governance lanes.
  • Minimal λ-like "shape program" to generate/transform point clouds.
  • Demos: polygon completion, symmetry inference, curve extrapolation, tiling.

This file is intentionally self-contained. Import nothing except stdlib.

Usage:
  python geometry_transformer_standalone_v2.py --demo all
  python geometry_transformer_standalone_v2.py --demo polygon --n 6 --k 3
"""

from __future__ import annotations
import json, time, math, random, argparse, sys, hashlib, os
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional, Callable

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         GeoLight (ledger/cache)                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _now() -> float:
    return time.time()

@dataclass
class LedgerEntry:
    idx: int
    ts: float
    scope: str
    channel: int
    input_hash: str
    result_hash: str
    cost: float
    ttl: Optional[float]
    prev: str
    entry: str

class GeoLight:
    """A tiny SpeedLight-like content-addressed cache + Merkle ledger."""
    def __init__(self, disk_dir: Optional[str]=None, ledger_path: Optional[str]=None, default_ttl: Optional[float]=None):
        self.disk_dir = disk_dir
        self.ledger_path = ledger_path
        self.default_ttl = default_ttl
        self.prev_hash = "0"*64
        self.entries: List[LedgerEntry] = []
        self.mem: Dict[str, Tuple[bytes, Optional[float]]] = {}
        if self.disk_dir:
            os.makedirs(self.disk_dir, exist_ok=True)
        if self.ledger_path:
            os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
            open(self.ledger_path, "a").close()

    def _disk_path(self, key: str) -> str:
        return os.path.join(self.disk_dir, key[:2] if self.disk_dir else "", key + ".json") if self.disk_dir else ""

    def compute(self, payload: Dict[str, Any], *, scope: str="geo", channel: int=3,
                compute_fn: Callable[[], Dict[str, Any]], ttl: Optional[float]=None) -> Tuple[Dict[str,Any], float, str]:
        ttl = self.default_ttl if ttl is None else ttl
        js = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        key = _sha256_hex(js)

        # in-memory
        hit = self.mem.get(key)
        if hit:
            b, exp = hit
            if exp is None or exp > _now():
                return json.loads(b.decode("utf-8")), 0.0, key
            else:
                self.mem.pop(key, None)

        # on-disk
        if self.disk_dir:
            p = self._disk_path(key)
            if os.path.exists(p):
                try:
                    with open(p, "rb") as f: b = f.read()
                    self.mem[key] = (b, (_now() + ttl) if ttl else None)
                    return json.loads(b.decode("utf-8")), 0.0, key
                except Exception:
                    pass

        # compute
        t0 = _now()
        result = compute_fn()
        cost = _now() - t0
        b = json.dumps(result, sort_keys=True, default=str).encode("utf-8")
        self.mem[key] = (b, (_now() + ttl) if ttl else None)
        if self.disk_dir:
            p = self._disk_path(key)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f: f.write(b)

        ih = _sha256_hex(js)
        rh = _sha256_hex(b)
        entry_payload = {"idx": len(self.entries), "ts": _now(), "scope": scope, "channel": channel,
                         "input_hash": ih, "result_hash": rh, "cost": cost, "ttl": ttl, "prev": self.prev_hash}
        entry_hash = _sha256_hex(json.dumps(entry_payload, sort_keys=True).encode("utf-8"))
        le = LedgerEntry(idx=entry_payload["idx"], ts=entry_payload["ts"], scope=scope, channel=channel,
                         input_hash=ih, result_hash=rh, cost=cost, ttl=ttl, prev=self.prev_hash, entry=entry_hash)
        self.entries.append(le)
        self.prev_hash = entry_hash
        if self.ledger_path:
            with open(self.ledger_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(le)) + "\n")
        return result, cost, key

    def verify(self) -> bool:
        prev = "0"*64
        for e in self.entries:
            payload = {"idx": e.idx, "ts": e.ts, "scope": e.scope, "channel": e.channel,
                       "input_hash": e.input_hash, "result_hash": e.result_hash,
                       "cost": e.cost, "ttl": e.ttl, "prev": prev}
            h = _sha256_hex(json.dumps(payload, sort_keys=True).encode("utf-8"))
            if h != e.entry: return False
            prev = h
        return True

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         Geometry core utilities                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

Vec = Tuple[float, float]

def v_add(a: Vec, b: Vec) -> Vec: return (a[0]+b[0], a[1]+b[1])
def v_sub(a: Vec, b: Vec) -> Vec: return (a[0]-b[0], a[1]-b[1])
def v_dot(a: Vec, b: Vec) -> float: return a[0]*b[0] + a[1]*b[1]
def v_norm(a: Vec) -> float: return math.hypot(a[0], a[1])
def v_scale(a: Vec, s: float) -> Vec: return (a[0]*s, a[1]*s)
def v_rot(a: Vec, theta: float) -> Vec:
    c, s = math.cos(theta), math.sin(theta)
    return (a[0]*c - a[1]*s, a[0]*s + a[1]*c)

def centroid(ps: List[Vec]) -> Vec:
    n = max(1, len(ps))
    return (sum(p[0] for p in ps)/n, sum(p[1] for p in ps)/n)

def angle(a: Vec) -> float:
    return math.atan2(a[1], a[0])

def rbf(dist: float, sigma: float) -> float:
    return math.exp(-(dist*dist)/(2*sigma*sigma))

def cos_sim(a: Vec, b: Vec) -> float:
    na, nb = v_norm(a), v_norm(b)
    if na == 0 or nb == 0: return 0.0
    return max(-1.0, min(1.0, v_dot(a,b)/(na*nb)))

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                     Geometry-Only Transformer layer                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

@dataclass
class GeoToken:
    pos: Vec                  # position in plane
    feat: Tuple[float, ...]   # arbitrary small feature vector (e.g., [curvature, tag])
    tag: str = ""             # optional label

class GeoAttention:
    """
    A single "attention" layer using purely geometric relations:
      • Keys/Queries: normalized direction vectors from local centroid
      • Values: token features (+pos residuals)
      • Weights: RBF(dist; sigma) * (1 + cos(angle delta))^alpha
    """
    def __init__(self, sigma: float=0.5, alpha: float=1.0, mix_pos: float=0.5):
        self.sigma = sigma
        self.alpha = alpha
        self.mix_pos = mix_pos

    def forward(self, toks: List[GeoToken]) -> List[GeoToken]:
        if not toks: return []
        pts = [t.pos for t in toks]
        c = centroid(pts)
        dirs = [v_sub(p, c) for p in pts]
        # avoid zero dir by nudging
        dirs = [(d[0]+1e-9, d[1]+1e-9) if v_norm(d)==0 else d for d in dirs]

        out: List[GeoToken] = []
        for i, ti in enumerate(toks):
            qi = dirs[i]
            accf = [0.0]*len(ti.feat)
            accp = (0.0, 0.0)
            z = 0.0
            for j, tj in enumerate(toks):
                if i == j: continue
                dj = dirs[j]
                w = rbf(v_norm(v_sub(ti.pos, tj.pos)), self.sigma) * ((1.0 + cos_sim(qi, dj))**self.alpha)
                z += w
                # value = mix(features, position delta)
                accf = [af + w*fj for af, fj in zip(accf, tj.feat)]
                accp = v_add(accp, v_scale(v_sub(tj.pos, ti.pos), w))
            if z == 0: z = 1.0
            nf = tuple(af/z for af in accf)
            np = v_add(ti.pos, v_scale(accp, self.mix_pos/z))
            # residual update
            out.append(GeoToken(np, tuple((fi + nfi)/2 for fi, nfi in zip(ti.feat, nf)), ti.tag))
        return out

class GeoTransformer:
    """Stack of GeoAttention layers + small geometric MLP (list-based) for readout."""
    def __init__(self, layers: int=3, sigma: float=0.5, alpha: float=1.0, mix_pos: float=0.5):
        self.layers = [GeoAttention(sigma, alpha, mix_pos) for _ in range(layers)]
        # Tiny readout parameters (fixed here; could be trainable via gradient-free rules)
        self.w_read = [0.6, 0.4, -0.2, 0.1]  # up to 4-dim feats supported

    def encode(self, pts: List[Vec], tags: Optional[List[str]]=None) -> List[GeoToken]:
        tags = tags or [""]*len(pts)
        c = centroid(pts)
        toks = []
        for p, tg in zip(pts, tags):
            d = v_sub(p, c)
            th = angle(d)
            r = v_norm(d)
            # features = [radius, angle/π, 1] (pad to 4)
            f = [r, th/math.pi, 1.0, 0.0]
            toks.append(GeoToken(p, tuple(f), tg))
        return toks

    def decode_score(self, toks: List[GeoToken]) -> float:
        # Example readout: pooled feature projection to a scalar for classification-ish tasks
        if not toks: return 0.0
        pool = [0.0]*len(toks[0].feat)
        for t in toks:
            pool = [a+b for a,b in zip(pool, t.feat)]
        pool = [x/len(toks) for x in pool]
        w = self.w_read[:len(pool)]
        return sum(a*b for a,b in zip(pool, w))

    def step(self, toks: List[GeoToken]) -> List[GeoToken]:
        for layer in self.layers:
            toks = layer.forward(toks)
        return toks

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         Shape "λ-programs"                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

def regular_ngon(n: int, r: float=1.0, theta0: float=0.0, center: Vec=(0.0,0.0)) -> List[Vec]:
    return [v_add(center, (r*math.cos(theta0 + 2*math.pi*k/n), r*math.sin(theta0 + 2*math.pi*k/n))) for k in range(n)]

def rotate_shape(pts: List[Vec], theta: float, about: Optional[Vec]=None) -> List[Vec]:
    about = centroid(pts) if about is None else about
    return [v_add(about, v_rot(v_sub(p, about), theta)) for p in pts]

def scale_shape(pts: List[Vec], s: float, about: Optional[Vec]=None) -> List[Vec]:
    about = centroid(pts) if about is None else about
    return [v_add(about, v_scale(v_sub(p, about), s)) for p in pts]

def reflect_shape(pts: List[Vec], axis: Tuple[Vec,Vec]) -> List[Vec]:
    a, b = axis
    ab = v_sub(b,a); abn = v_norm(ab)
    if abn == 0: return pts
    ux, uy = ab[0]/abn, ab[1]/abn
    def refl(p):
        ap = v_sub(p, a)
        proj = (ap[0]*ux + ap[1]*uy)
        pr = (a[0] + proj*ux, a[1] + proj*uy)
        perp = v_sub(p, pr)
        return v_sub(pr, perp)
    return [refl(p) for p in pts]

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                           Demos & tasks                              ║
# ╚══════════════════════════════════════════════════════════════════════╝

def demo_polygon_completion(n=6, k=3, layers=3) -> Dict[str,Any]:
    """Give first k vertices of an n-gon and let the model propose the remainder by symmetry extrapolation."""
    true_pts = regular_ngon(n, r=1.0, theta0=0.0, center=(0.0,0.0))
    known = true_pts[:k]
    gt_rest = true_pts[k:]

    gt = GeoTransformer(layers=layers, sigma=0.6, alpha=1.0, mix_pos=0.7)
    toks = gt.encode(known)
    toks = gt.step(toks)

    # infer rotation angle from mean neighbor delta
    c = centroid([t.pos for t in toks])
    dirs = [v_sub(t.pos, c) for t in toks]
    angs = [angle(d) for d in dirs]
    angs = sorted(angs)
    if len(angs) >= 2:
        # average gap
        gaps = [(angs[(i+1)%len(angs)] - angs[i])%(2*math.pi) for i in range(len(angs))]
        dtheta = sum(gaps)/len(gaps)
    else:
        dtheta = 2*math.pi/n

    # propose remaining pts
    last = known[-1]
    rem = []
    for _ in range(n-k):
        v = v_sub(last, c)
        v = v_rot(v, dtheta)
        nxt = v_add(c, v)
        rem.append(nxt)
        last = nxt

    score = gt.decode_score(toks)
    return {"known": known, "pred_rest": rem, "gt_rest": gt_rest, "score": score, "n": n, "k": k, "dtheta": dtheta}

def demo_symmetry_inference(layers=3) -> Dict[str,Any]:
    """Detect approximate dihedral symmetry order via spectral gap on angle histogram."""
    pts = regular_ngon(n=random.choice([3,4,5,6,8]), r=1.0, theta0=random.random()*2*math.pi)
    pts = scale_shape(rotate_shape(pts, 0.17), 1.0 + 0.05*random.random())
    gt = GeoTransformer(layers=layers, sigma=0.5, alpha=1.3, mix_pos=0.5)
    toks = gt.step(gt.encode(pts))
    c = centroid([t.pos for t in toks])
    angs = [((angle(v_sub(p,c))%(2*math.pi))) for p in pts]
    angs.sort()
    gaps = [((angs[(i+1)%len(angs)]-angs[i])%(2*math.pi)) for i in range(len(angs))]
    mean_gap = sum(gaps)/len(gaps)
    order = max(3, int(round((2*math.pi)/mean_gap)))
    return {"pts": pts, "order": order, "mean_gap": mean_gap}

def demo_curve_extrapolation(m=12, layers=3) -> Dict[str,Any]:
    """Extrapolate a smooth curve by local curvature from last points (no fitting)."""
    xs = [i*0.3 for i in range(m)]
    ys = [math.sin(x) for x in xs]
    pts = list(zip(xs, ys))
    known = pts[:m-3]
    gt = GeoTransformer(layers=layers, sigma=0.8, alpha=1.0, mix_pos=0.4)
    toks = gt.step(gt.encode(known))
    # Extrapolate using last chord and average curvature estimated geometrically
    p1, p2, p3 = known[-3], known[-2], known[-1]
    v1, v2 = v_sub(p2,p1), v_sub(p3,p2)
    ang = (angle(v2)-angle(v1))
    # normalize angle to [-pi,pi]
    if ang > math.pi: ang -= 2*math.pi
    if ang < -math.pi: ang += 2*math.pi
    step = v_norm(v2)
    pred1 = v_add(p3, v_rot(v2, ang))
    pred2 = v_add(pred1, v_rot(v2, ang))
    return {"known": known, "pred": [pred1, pred2], "gt_next": pts[m-2:], "est_turn": ang, "step": step}

def demo_tiling_hex(radius=2, layers=2) -> Dict[str,Any]:
    """Generate hexagonal tiling coordinates (3/6 symmetry) and project through layers to stabilize lattice axes."""
    base = regular_ngon(6, r=1.0)
    tiles = []
    for i in range(-radius, radius+1):
        for j in range(-radius, radius+1):
            shift = (i*1.5, j*math.sqrt(3)/2 + (i%2)*math.sqrt(3)/4)
            tiles.extend([v_add(p, shift) for p in base])
    gt = GeoTransformer(layers=layers, sigma=0.9, alpha=1.0, mix_pos=0.3)
    toks = gt.step(gt.encode(tiles))
    score = gt.decode_score(toks)
    return {"tiles": tiles, "score": score, "count": len(tiles)}

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         ΔΦ guard & channels                          ║
# ╚══════════════════════════════════════════════════════════════════════╝

def delta_phi(before: List[GeoToken], after: List[GeoToken]) -> float:
    """Simple ΔΦ proxy: total squared displacement + feature L2 change."""
    if len(before) != len(after): return float("inf")
    s = 0.0
    for b,a in zip(before, after):
        dp = v_sub(a.pos, b.pos)
        s += v_dot(dp, dp)
        s += sum((af-bf)*(af-bf) for af,bf in zip(a.feat, b.feat))
    return s

def channel_policy(channel: int, dphi: float) -> bool:
    # Example policy: channel 3 allows small positive ΔΦ, 6 enforces ΔΦ≤0.1, 9 prefers exactly 0 (idempotent)
    if channel == 3: return dphi <= 1e3
    if channel == 6: return dphi <= 0.1
    if channel == 9: return dphi <= 1e-6
    return True

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                             CLI & Runner                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

def run_with_ledger(payload: Dict[str, Any], compute: Callable[[], Dict[str,Any]],
                    scope="geom-xf", channel=3, ttl=30.0, use_disk=False):
    gl = GeoLight(disk_dir=".geolight/cache" if use_disk else None,
                  ledger_path=".geolight/ledger.jsonl" if use_disk else None,
                  default_ttl=ttl)
    res, cost, rid = gl.compute(payload, scope=scope, channel=channel, compute_fn=compute, ttl=ttl)
    return {"result": res, "cost": cost, "receipt": rid, "ledger_ok": gl.verify(), "entries": len(gl.entries)}

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--demo", type=str, default="all", choices=["all","polygon","symmetry","curve","tiling"])
    p.add_argument("--n", type=int, default=6, help="polygon sides")
    p.add_argument("--k", type=int, default=3, help="known vertices for polygon")
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--channel", type=int, default=3, choices=[3,6,9])
    p.add_argument("--disk", action="store_true")
    args = p.parse_args(argv)

    if args.demo in ("all","polygon"):
        payload = {"demo":"polygon","n":args.n,"k":args.k,"layers":args.layers}
        def compute(): return demo_polygon_completion(n=args.n, k=args.k, layers=args.layers)
        out = run_with_ledger(payload, compute, channel=args.channel, use_disk=args.disk)
        print("POLYGON:", json.dumps(out, indent=2))

    if args.demo in ("all","symmetry"):
        payload = {"demo":"symmetry","layers":args.layers}
        def compute(): return demo_symmetry_inference(layers=args.layers)
        out = run_with_ledger(payload, compute, channel=args.channel, use_disk=args.disk)
        print("SYMMETRY:", json.dumps(out, indent=2))

    if args.demo in ("all","curve"):
        payload = {"demo":"curve","layers":args.layers}
        def compute(): return demo_curve_extrapolation(layers=args.layers)
        out = run_with_ledger(payload, compute, channel=args.channel, use_disk=args.disk)
        print("CURVE:", json.dumps(out, indent=2))

    if args.demo in ("all","tiling"):
        payload = {"demo":"tiling","layers":args.layers}
        def compute(): return demo_tiling_hex(layers=args.layers)
        out = run_with_ledger(payload, compute, channel=args.channel, use_disk=args.disk)
        print("TILING:", json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
