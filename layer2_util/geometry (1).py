"""
GEOMETRY Module
---------------

Contains: index.html, test_speedlight_plus, geo_tokenizer_tiein_v1, cqe_sidecar_adapter, quickstart_personal_node, server, cqe_time, geometry_bridge, TestPerformanceBenchmarks, SacredBinaryPattern, coherence_metrics, api, inverse_residue, dihedral_ca, test_beta_eta_delta_mu, ToroidalRotationType, CQEOperationMode, RotationMode, viewer_api, CQEConfiguration, dihedral_ca_1, CQELanguageEngine, FractalDataProcessor, factorial_mu, ForceType, runtime, SacredFractalPattern, AtomCombinationType, ProcessingPriority
"""

import hashlib
import io
import json
import math
import os
import sys
import time

try:
    import Any
except ImportError:
    Any = None
try:
    import Callable
except ImportError:
    Callable = None
try:
    import Dict
except ImportError:
    Dict = None
try:
    import List
except ImportError:
    List = None
try:
    import Optional
except ImportError:
    Optional = None
try:
    import SpeedLightPlus
except ImportError:
    SpeedLightPlus = None
try:
    import Tuple
except ImportError:
    Tuple = None
try:
    import add_item
except ImportError:
    add_item = None
try:
    import annotations
except ImportError:
    annotations = None
try:
    import argparse
except ImportError:
    argparse = None
try:
    import asdict
except ImportError:
    asdict = None
try:
    import ast
except ImportError:
    ast = None
try:
    import eval
except ImportError:
    eval = None
try:
    import fuse
except ImportError:
    fuse = None
try:
    import get_item
except ImportError:
    get_item = None
try:
    import l2_norm
except ImportError:
    l2_norm = None
try:
    import list_items
except ImportError:
    list_items = None
try:
    import log
except ImportError:
    log = None
try:
    import main
except ImportError:
    main = None
try:
    import random
except ImportError:
    random = None
try:
    import rgb_to_hex
except ImportError:
    rgb_to_hex = None
try:
    import search
except ImportError:
    search = None
try:
    import stats
except ImportError:
    stats = None
try:
    import struct
except ImportError:
    struct = None
try:
    import threading
except ImportError:
    threading = None
try:
    import uuid
except ImportError:
    uuid = None
try:
    import wavelength_to_rgb
except ImportError:
    wavelength_to_rgb = None
try:
    import zlib
except ImportError:
    zlib = None

try:
    import numpy as np
except ImportError:
    np = None


# ============================================================================
# index.html
# ============================================================================



<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Viewer24 v2 — Dihedral CA Overlay</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <header>
    <h1>Viewer24 Controller v2 (CA Overlay)</h1>
    <div class="controls">
      <textarea id="points" rows="3" placeholder='[[x,y], ...]'></textarea>
      <button id="load">Load Points</button>
      <button id="caInit">Init CA</button>
      <button id="caPlay">Play</button>
      <button id="caPause">Pause</button>
      <label>Alpha <input id="alpha" type="number" value="160" min="0" max="255" step="5"></label>
      <span id="status"></span>
    </div>
  </header>
  <main id="grid"></main>
  <script src="/static/overlay_ca.js"></script>
</body>
</html>




# ============================================================================
# test_speedlight_plus
# ============================================================================



def test_basic_cache():
    sl = SpeedLightPlus(mem_bytes=5_000_000)
    payload = {"op":"square_sum","n":10000}
    def compute():
        return {"sum": sum(i*i for i in range(10000))}
    r1, c1, id1 = sl.compute(payload, scope="test", channel=3, compute_fn=compute)
    r2, c2, id2 = sl.compute(payload, scope="test", channel=3, compute_fn=compute)
    assert r1 == r2 and id1 == id2 and c2 == 0.0




# ============================================================================
# geo_tokenizer_tiein_v1
# ============================================================================



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
GeoTokenizer Tie-In v1 — Geometry-First Token Memory & Codec
============================================================
Pure stdlib. Companion to Geometry-Only Transformer v2, but runs standalone.

What you get:
  • Geometry-native token codec (encode/decode) with quantization + varint + zlib.
  • Token ops: break/extend/combine/refine + synthesis hooks via transformer when present.
  • Memory store of "equivalence tokens" (prototypes) using shape embeddings and cosine match.
  • Receipts-first: content-addressed compute + Merkle-chained ledger (TokLight).
  • CLI for encode/decode/learn/convert/synthesize/extend/refine/combine/break.

This is not a text tokenizer. It’s a geometry/memory manager that can mint/upgrade
tokens on demand and convert to known canonical tokens using past learned embeddings.
\"\"\"

# ───────────────────────────── Ledger: TokLight ─────────────────────────────

def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

@dataclass
class LedgerEntry:
    idx: int
    ts: float
    scope: str
    op: str
    input_hash: str
    result_hash: str
    cost: float
    prev: str
    entry: str

class TokLight:
    def __init__(self, ledger_path: Optional[str]=None):
        self.ledger_path = ledger_path
        self.prev = "0"*64
        self.entries: List[LedgerEntry] = []
        if self.ledger_path:
            os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
            open(self.ledger_path, "a").close()

    def log(self, scope: str, op: str, inp: bytes, out: bytes, cost: float):
        ih, oh = _sha256_hex(inp), _sha256_hex(out)
        payload = {"idx": len(self.entries), "ts": time.time(), "scope": scope, "op": op,
                   "input_hash": ih, "result_hash": oh, "cost": cost, "prev": self.prev}
        entry = _sha256_hex(json.dumps(payload, sort_keys=True).encode("utf-8"))
        le = LedgerEntry(idx=payload["idx"], ts=payload["ts"], scope=scope, op=op,
                         input_hash=ih, result_hash=oh, cost=cost, prev=self.prev, entry=entry)
        self.entries.append(le)
        self.prev = entry
        if self.ledger_path:
            with open(self.ledger_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(le)) + "\\n")

    def verify(self) -> bool:
        prev = "0"*64
        for e in self.entries:
            payload = {"idx": e.idx, "ts": e.ts, "scope": e.scope, "op": e.op,
                       "input_hash": e.input_hash, "result_hash": e.result_hash, "cost": e.cost, "prev": prev}
            h = _sha256_hex(json.dumps(payload, sort_keys=True).encode("utf-8"))
            if h != e.entry: return False
            prev = h
        return True

# ───────────────────────────── Geometry primitives ───────────────────────────

Vec = Tuple[float, float]

@dataclass
class GeoToken:
    pos: Vec
    feat: Tuple[float, ...]
    tag: str = ""

def centroid(ps: List[Vec]) -> Vec:
    n = max(1, len(ps))
    return (sum(p[0] for p in ps)/n, sum(p[1] for p in ps)/n)

def v_sub(a: Vec, b: Vec) -> Vec: return (a[0]-b[0], a[1]-b[1])
def v_add(a: Vec, b: Vec) -> Vec: return (a[0]+b[0], a[1]+b[1])
def v_norm(a: Vec) -> float: return math.hypot(a[0], a[1])
def angle(a: Vec) -> float: return math.atan2(a[1], a[0])

# ───────────────────────────── Codec: varint + zigzag ───────────────────────

def zigzag_encode(x: int) -> int:
    return (x << 1) ^ (x >> 63)

def zigzag_decode(u: int) -> int:
    return (u >> 1) ^ -(u & 1)

def write_varint(n: int, buf: bytearray):
    while True:
        to_write = n & 0x7F
        n >>= 7
        if n:
            buf.append(to_write | 0x80)
        else:
            buf.append(to_write)
            break

def read_varint(b: bytes, i: int) -> Tuple[int, int]:
    shift = 0; result = 0
    while True:
        if i >= len(b): raise ValueError("varint overflow")
        byte = b[i]; i += 1
        result |= ((byte & 0x7F) << shift)
        if not (byte & 0x80): break
        shift += 7
    return result, i

class GeoCodec:
    MAGIC = b"GEO2"
    VERSION = 1

    def __init__(self, scale: float=1e-3, compress: bool=True):
        self.scale = scale
        self.compress = compress

    def _quant(self, x: float) -> int:
        return int(round(x / self.scale))

    def _dequant(self, q: int) -> float:
        return q * self.scale

    def encode(self, toks: List[GeoToken]) -> bytes:
        # Build a tag dictionary
        tags = sorted({t.tag for t in toks if t.tag})
        tag2id = {t:i+1 for i,t in enumerate(tags)}  # 0 reserved for ""
        buf = bytearray()
        buf.extend(self.MAGIC)
        buf.append(self.VERSION)
        buf.extend(struct.pack(">d", self.scale))  # 8-byte float
        write_varint(len(toks), buf)
        write_varint(len(tags), buf)
        # tag table
        for t in tags:
            tb = t.encode("utf-8")
            write_varint(len(tb), buf); buf.extend(tb)
        # tokens: delta-code positions, varint feats, tag ids
        px, py = 0, 0
        for tok in toks:
            qx, qy = self._quant(tok.pos[0]), self._quant(tok.pos[1])
            dx, dy = qx - px, qy - py
            write_varint(zigzag_encode(dx), buf)
            write_varint(zigzag_encode(dy), buf)
            px, py = qx, qy
            # features: clamp to 8, quantize by same scale (ok for demo)
            f = list(tok.feat)[:8] + [0.0]*(max(0, 8-len(tok.feat)))
            write_varint(8, buf)
            for fv in f:
                qf = self._quant(fv)
                write_varint(zigzag_encode(qf), buf)
            # tag id
            tid = tag2id.get(tok.tag, 0)
            write_varint(tid, buf)
        raw = bytes(buf)
        if self.compress:
            return b"Z" + zlib.compress(raw)
        else:
            return b"N" + raw

    def decode(self, b: bytes) -> List[GeoToken]:
        if not b: return []
        if b[0:1] == b"Z":
            raw = zlib.decompress(b[1:])
        elif b[0:1] == b"N":
            raw = b[1:]
        else:
            raise ValueError("Bad header")
        i = 0
        if raw[i:i+4] != self.MAGIC: raise ValueError("Magic mismatch"); i += 4
        ver = raw[i]; i += 1
        if ver != self.VERSION: raise ValueError("Version mismatch")
        scale = struct.unpack(">d", raw[i:i+8])[0]; i += 8
        self.scale = scale
        n, i = read_varint(raw, i)
        m, i = read_varint(raw, i)
        tags = []
        for _ in range(m):
            L, i = read_varint(raw, i)
            s = raw[i:i+L].decode("utf-8"); i += L
            tags.append(s)
        toks: List[GeoToken] = []
        px, py = 0, 0
        for _ in range(n):
            dx, i = read_varint(raw, i); dy, i = read_varint(raw, i)
            qx, qy = px + zigzag_decode(dx), py + zigzag_decode(dy)
            x, y = self._dequant(qx), self._dequant(qy); px, py = qx, qy
            k, i = read_varint(raw, i)
            feats = []
            for _j in range(k):
                qf, i = read_varint(raw, i)
                feats.append(self._dequant(zigzag_decode(qf)))
            tid, i = read_varint(raw, i)
            tag = "" if tid == 0 else tags[tid-1]
            toks.append(GeoToken((x,y), tuple(feats), tag))
        return toks

# ───────────────────────────── Memory: embeddings ────────────────────────────

def radial_angle_embed(toks: List[GeoToken], rbins=16, abins=16) -> List[float]:
    if not toks: return [0.0]*(rbins+abins+4)
    c = centroid([t.pos for t in toks])
    rs, ths = [], []
    for t in toks:
        d = v_sub(t.pos, c)
        rs.append(v_norm(d))
        ths.append((angle(d)%(2*math.pi)))
    R = max(1e-9, max(rs))
    rh = [0]*rbins; ah = [0]*abins
    for r, th in zip(rs, ths):
        ri = min(rbins-1, int(rbins * (r / R)))
        ai = min(abins-1, int(abins * (th /(2*math.pi))))
        rh[ri] += 1; ah[ai] += 1
    # normalize
    rh = [x/len(toks) for x in rh]
    ah = [x/len(toks) for x in ah]
    return rh + ah + [float(len(toks)), R, sum(rs)/len(rs), sum(1 for t in toks if t.tag!="")/len(toks)]

def cos_sim(u: List[float], v: List[float]) -> float:
    if len(u)!=len(v): return 0.0
    du = sum(x*x for x in u); dv = sum(y*y for y in v)
    if du==0 or dv==0: return 0.0
    return sum(x*y for x,y in zip(u,v)) / math.sqrt(du*dv)

class TokenMemory:
    def __init__(self, path: str=".geo_tokenizer/memory.json"):
        self.path = path
        self.db: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(self.path):
            try:
                self.db = json.load(open(self.path, "r", encoding="utf-8"))
            except Exception:
                self.db = {}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.db, f, indent=2)

    def learn(self, name: str, toks: List[GeoToken], meta: Optional[Dict[str,Any]]=None):
        emb = radial_angle_embed(toks)
        self.db[name] = {"emb": emb, "meta": meta or {}, "ts": time.time()}
        self.save()

    def nearest(self, toks: List[GeoToken]) -> Tuple[Optional[str], float]:
        if not self.db: return (None, 0.0)
        emb = radial_angle_embed(toks)
        best, bests = None, -1.0
        for k, v in self.db.items():
            s = cos_sim(emb, v["emb"])
            if s > bests: best, bests = k, s
        return best, bests

# ───────────────────────────── Token Ops & Synthesis ─────────────────────────

def split_tokens(toks: List[GeoToken], idx: int) -> Tuple[List[GeoToken], List[GeoToken]]:
    return toks[:idx], toks[idx:]

def merge_tokens(a: List[GeoToken], b: List[GeoToken]) -> List[GeoToken]:
    return a + b

def refine_tokens(toks: List[GeoToken], iters: int=1) -> List[GeoToken]:
    # simple Laplacian-like smoothing on positions
    if len(toks) < 3: return toks
    pts = [t.pos for t in toks]
    for _ in range(iters):
        new_pts = [pts[0]]  # keep endpoints
        for i in range(1, len(pts)-1):
            x = (pts[i-1][0] + 2*pts[i][0] + pts[i+1][0]) / 4
            y = (pts[i-1][1] + 2*pts[i][1] + pts[i+1][1]) / 4
            new_pts.append((x,y))
        new_pts.append(pts[-1]); pts = new_pts
    out = []
    for t, p in zip(toks, pts):
        out.append(GeoToken(p, t.feat, t.tag))
    return out

def extend_tokens_polygon(toks: List[GeoToken], target_n: int) -> List[GeoToken]:
    # If the transformer is available, use it; otherwise geometric gap inference
    try:
        import geometry_transformer_standalone_v2 as G
        gt = G.GeoTransformer(layers=3, sigma=0.6, alpha=1.0, mix_pos=0.7)
        st = gt.encode([t.pos for t in toks]); st = gt.step(st)
        c = G.centroid([t.pos for t in st])
        angs = sorted([G.angle(G.v_sub(t.pos,c))%(2*math.pi) for t in st])
        gaps = [((angs[(i+1)%len(angs)]-angs[i])%(2*math.pi)) for i in range(len(angs))]
        dtheta = sum(gaps)/len(gaps) if gaps else 2*math.pi/target_n
        last = toks[-1].pos; rem = []
        for _ in range(max(0, target_n-len(toks))):
            v = (last[0]-c[0], last[1]-c[1])
            v = (v[0]*math.cos(dtheta)-v[1]*math.sin(dtheta), v[0]*math.sin(dtheta)+v[1]*math.cos(dtheta))
            nxt = (c[0]+v[0], c[1]+v[1]); rem.append(GeoToken(nxt, toks[-1].feat, toks[-1].tag)); last = nxt
        return toks + rem
    except Exception:
        # fallback
        if len(toks) < 2: return toks
        c = centroid([t.pos for t in toks])
        angs = sorted([(angle(v_sub(t.pos,c))%(2*math.pi)) for t in toks])
        gaps = [((angs[(i+1)%len(angs)]-angs[i])%(2*math.pi)) for i in range(len(angs))]
        dtheta = sum(gaps)/len(gaps) if gaps else 2*math.pi/target_n
        last = toks[-1].pos; rem = []
        for _ in range(max(0, target_n-len(toks))):
            v = (last[0]-c[0], last[1]-c[1])
            v = (v[0]*math.cos(dtheta)-v[1]*math.sin(dtheta), v[0]*math.sin(dtheta)+v[1]*math.cos(dtheta))
            nxt = (c[0]+v[0], c[1]+v[1]); rem.append(GeoToken(nxt, toks[-1].feat, toks[-1].tag)); last = nxt
        return toks + rem

# ───────────────────────────── High-level API ────────────────────────────────

class GeoTokenizer:
    def __init__(self, scale: float=1e-3, compressed: bool=True, memory_path: str=".geo_tokenizer/memory.json",
                 ledger_path: Optional[str]=".geo_tokenizer/ledger.jsonl"):
        self.codec = GeoCodec(scale=scale, compress=compressed)
        self.mem = TokenMemory(memory_path)
        self.ledger = TokLight(ledger_path)

    # Encode/decode
    def encode(self, toks: List[GeoToken]) -> bytes:
        t0 = time.time()
        raw_inp = json.dumps({"count": len(toks)}).encode("utf-8")
        b = self.codec.encode(toks)
        self.ledger.log("tokenizer", "encode", raw_inp, b, time.time()-t0)
        return b

    def decode(self, b: bytes) -> List[GeoToken]:
        t0 = time.time()
        toks = self.codec.decode(b)
        raw_out = json.dumps({"count": len(toks)}).encode("utf-8")
        self.ledger.log("tokenizer", "decode", b, raw_out, time.time()-t0)
        return toks

    # Memory
    def learn_equivalence(self, name: str, toks: List[GeoToken], meta: Optional[Dict[str,Any]]=None):
        t0 = time.time()
        self.mem.learn(name, toks, meta)
        raw_inp = json.dumps({"name": name, "count": len(toks)}).encode("utf-8")
        raw_out = json.dumps({"ok": True}).encode("utf-8")
        self.ledger.log("memory", "learn", raw_inp, raw_out, time.time()-t0)

    def convert_to_known(self, toks: List[GeoToken], threshold: float=0.92) -> Tuple[Optional[str], float]:
        t0 = time.time()
        name, score = self.mem.nearest(toks)
        if name is not None and score >= threshold:
            out = json.dumps({"name": name, "score": score}).encode("utf-8")
        else:
            out = json.dumps({"name": None, "score": score}).encode("utf-8")
        inp = json.dumps({"count": len(toks)}).encode("utf-8")
        self.ledger.log("memory", "convert", inp, out, time.time()-t0)
        return (name if score>=threshold else None, score)

    # Ops
    def break_apart(self, toks: List[GeoToken], idx: int) -> Tuple[List[GeoToken], List[GeoToken]]:
        a, b = split_tokens(toks, idx)
        return a, b

    def combine(self, a: List[GeoToken], b: List[GeoToken]) -> List[GeoToken]:
        return merge_tokens(a, b)

    def refine(self, toks: List[GeoToken], iters: int=1) -> List[GeoToken]:
        return refine_tokens(toks, iters=iters)

    def extend(self, toks: List[GeoToken], target_n: int) -> List[GeoToken]:
        return extend_tokens_polygon(toks, target_n)

# ───────────────────────────── Utilities & CLI ───────────────────────────────

def regular_ngon(n, r=1.0, theta0=0.0, center=(0.0,0.0)):
    return [(center[0]+r*math.cos(theta0+2*math.pi*k/n), center[1]+r*math.sin(theta0+2*math.pi*k/n)) for k in range(n)]

def toks_from_points(pts: List[Tuple[float,float]], tag="") -> List[GeoToken]:
    c = centroid(pts)
    out = []
    for p in pts:
        d = v_sub(p, c); th = angle(d); r = v_norm(d)
        feat = (r, th/math.pi, 1.0, 0.0)
        out.append(GeoToken(p, feat, tag))
    return out

def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    enc = sub.add_parser("encode")
    enc.add_argument("--in-json", type=str, help="JSON file of points [[x,y],...]")
    enc.add_argument("--out", type=str, required=True)

    dec = sub.add_parser("decode")
    dec.add_argument("--in", dest="inp", type=str, required=True)
    dec.add_argument("--out-json", type=str, required=True)

    learn = sub.add_parser("learn")
    learn.add_argument("--name", required=True)
    learn.add_argument("--from-json", type=str, required=True)

    conv = sub.add_parser("convert")
    conv.add_argument("--from-json", type=str, required=True)

    syn = sub.add_parser("synthesize")
    syn.add_argument("--n", type=int, default=6)
    syn.add_argument("--k", type=int, default=3)

    ext = sub.add_parser("extend")
    ext.add_argument("--from-json", type=str, required=True)
    ext.add_argument("--target-n", type=int, required=True)

    refn = sub.add_parser("refine")
    refn.add_argument("--from-json", type=str, required=True)
    refn.add_argument("--iters", type=int, default=1)

    brk = sub.add_parser("break")
    brk.add_argument("--from-json", type=str, required=True)
    brk.add_argument("--idx", type=int, required=True)

    args = p.parse_args(argv)
    gtok = GeoTokenizer()

    if args.cmd == "encode":
        pts = json.load(open(args.in_json))  # list of [x,y]
        toks = toks_from_points([tuple(p) for p in pts])
        b = gtok.encode(toks)
        with open(args.out, "wb") as f: f.write(b)
        print(json.dumps({"bytes": len(b)}))
        return

    if args.cmd == "decode":
        b = open(args.inp, "rb").read()
        toks = gtok.decode(b)
        pts = [list(t.pos) for t in toks]
        json.dump(pts, open(args.out_json, "w"), indent=2)
        print(json.dumps({"count": len(pts)}))
        return

    if args.cmd == "learn":
        pts = json.load(open(args.from_json))
        toks = toks_from_points([tuple(p) for p in pts])
        gtok.learn_equivalence(args.name, toks, meta={"src":"learn-cli"})
        print(json.dumps({"ok": True, "name": args.name}))
        return

    if args.cmd == "convert":
        pts = json.load(open(args.from_json))
        toks = toks_from_points([tuple(p) for p in pts])
        name, score = gtok.convert_to_known(toks)
        print(json.dumps({"name": name, "score": score}))
        return

    if args.cmd == "synthesize":
        # create k known vertices of n-gon then extend to full
        pts = regular_ngon(args.n)
        known = pts[:args.k]
        toks = toks_from_points(known, tag="seed")
        ext = gtok.extend(toks, target_n=args.n)
        out = {"known": known, "extended": [list(t.pos) for t in ext]}
        print(json.dumps(out, indent=2))
        return

    if args.cmd == "extend":
        pts = json.load(open(args.from_json))
        toks = toks_from_points([tuple(p) for p in pts])
        ext = gtok.extend(toks, target_n=args.target_n)
        out = {"extended": [list(t.pos) for t in ext]}
        print(json.dumps(out, indent=2))
        return

    if args.cmd == "refine":
        pts = json.load(open(args.from_json))
        toks = toks_from_points([tuple(p) for p in pts])
        out = gtok.refine(toks, iters=args.iters)
        print(json.dumps({"refined": [list(t.pos) for t in out]}, indent=2))
        return

    if args.cmd == "break":
        pts = json.load(open(args.from_json))
        toks = toks_from_points([tuple(p) for p in pts])
        a, b = gtok.break_apart(toks, args.idx)
        print(json.dumps({"a":[list(t.pos) for t in a], "b":[list(t.pos) for t in b]}, indent=2))
        return

    p.print_help()

if __name__ == "__main__":
    main()




# ============================================================================
# cqe_sidecar_adapter
# ============================================================================



try:
    from morphonic_cqe_unified.sidecar.speedlight_sidecar_plus import SpeedLightPlus as SpeedLight
except Exception:
    from morphonic_cqe_unified.sidecar.speedlight_sidecar import SpeedLight  # type: ignore

class CQESidecar:
    def __init__(self):
        self._sl = SpeedLight()
        self._meta: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def _hash_payload(self, payload: Any) -> str:
        js = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(js.encode("utf-8")).hexdigest()

    def compute(self, payload: Any, scope: str, channel: int, compute_fn=None, *args, **kwargs) -> Tuple[Any, float, str]:
        with self._lock:
            rid = self._hash_payload({"payload": payload, "scope": scope})
            result, cost, receipt_id = self._sl.compute(payload, scope=scope, channel=channel, compute_fn=compute_fn, **kwargs)
            if cost > 0.0 and receipt_id not in self._meta:
                self._meta[receipt_id] = {"scope": scope, "channel": channel, "note": kwargs.get("note","")}
            return result, cost, receipt_id

    def get_meta(self, receipt_id: str) -> Dict[str, Any]:
        with self._lock:
            meta = dict(self._meta.get(receipt_id, {}))
            if not meta and hasattr(self._sl, "get_meta"):
                meta = self._sl.get_meta(receipt_id) or meta
            return meta

    def report(self) -> str:
        return self._sl.report()




# ============================================================================
# quickstart_personal_node
# ============================================================================



if __name__ == "__main__":
    main()




# ============================================================================
# server
# ============================================================================



SESSION = {"points": [], "meta": {}}
TILES_X, TILES_Y = 6, 4
N = 64
CA = DihedralCA(tiles_x=TILES_X, tiles_y=TILES_Y, n=N, seed=1337)
CA.seed_from_specs(NIEMEIER_SPECS + ["LEECH"])
INV = ResidueAnalyzer(CA)

def read_json(environ):
    try: length = int(environ.get('CONTENT_LENGTH','0'))
    except (ValueError): length = 0
    body = environ['wsgi.input'].read(length) if length>0 else b'{}'
    return json.loads(body.decode('utf-8') or "{}")

def respond(start_response, status, obj, ctype="application/json"):
    data = json.dumps(obj).encode("utf-8")
    start_response(status, [('Content-Type', ctype), ('Content-Length', str(len(data)))])
    return [data]

def app(environ, start_response):
    path = environ.get('PATH_INFO','/'); method = environ.get('REQUEST_METHOD','GET')

    if path == "/api/load" and method == "POST":
        payload = read_json(environ); SESSION["points"]=payload.get("points") or []; SESSION["meta"]=payload.get("meta") or {}
        return respond(start_response,'200 OK',{"ok":True,"count":len(SESSION["points"])})
    if path == "/api/screens":
        labs = NIEMEIER_SPECS + ["LEECH"]
        return respond(start_response,'200 OK',{"screens":[{"index":i,"label":lab} for i,lab in enumerate(labs)]})
    if path == "/api/frame":
        q = parse_qs(environ.get('QUERY_STRING','')); w=int(q.get('w',['320'])[0]); h=int(q.get('h',['180'])[0])
        s,tx,ty = world_to_screen(SESSION.get("points") or [], w, h, padding=0.08)
        return respond(start_response,'200 OK',{"s":s,"tx":tx,"ty":ty})
    # CA controls
    if path == "/api/ca/init":
        q = parse_qs(environ.get('QUERY_STRING','')); n=int(q.get('n',['64'])[0])
        global CA,N,INV; N=n; CA=DihedralCA(tiles_x=TILES_X, tiles_y=TILES_Y, n=N, seed=1337); CA.seed_from_specs(NIEMEIER_SPECS+["LEECH"]); INV = ResidueAnalyzer(CA)
        return respond(start_response,'200 OK',{"ok":True,"n":N})
    if path == "/api/ca/step":
        q = parse_qs(environ.get('QUERY_STRING','')); steps=int(q.get('steps',['1'])[0]); kappa=float(q.get('kappa',['0.08'])[0])
        for _ in range(steps): CA.step(kappa=kappa, dual=True)
        return respond(start_response,'200 OK',{"ok":True,"step":CA.step_id})
    if path == "/api/ca/tile":
        q = parse_qs(environ.get('QUERY_STRING','')); idx=int(q.get('index',['0'])[0]); alpha=int(q.get('alpha',['160'])[0])
        tile = CA.tile_pixels_em(idx, alpha=alpha); return respond(start_response,'200 OK',tile)
    # Inverse/residue endpoints
    if path == "/api/inverse/baseline":
        INV.capture_baseline(); return respond(start_response,'200 OK',{"ok":True})
    if path == "/api/inverse/tile":
        q = parse_qs(environ.get('QUERY_STRING','')); idx=int(q.get('index',['0'])[0])
        tile = INV.residue_tile(idx)
        return respond(start_response,'200 OK',tile)
    # Static
    if path == "/":
        with open("./static/index.html","rb") as f: data=f.read()
        start_response('200 OK',[('Content-Type','text/html')]); return [data]
    if path == "/inverse":
        with open("./static/inverse.html","rb") as f: data=f.read()
        start_response('200 OK',[('Content-Type','text/html')]); return [data]
    if path.startswith("/static/"):
        p = "."+path
        if not os.path.exists(p): start_response('404 NOT FOUND',[('Content-Type','text/plain')]); return [b'not found']
        ctype="text/plain"
        if p.endswith(".html"): ctype="text/html"
        if p.endswith(".js"): ctype="text/javascript"
        if p.endswith(".css"): ctype="text/css"
        with open(p,"rb") as f: data=f.read()
        start_response('200 OK',[('Content-Type',ctype)]); return [data]
    start_response('404 NOT FOUND',[('Content-Type','application/json')]); return [b'{}']

def serve(host="127.0.0.1", port=9091):
    httpd = make_server(host, port, app)
    print(f"Viewer24 v2 + CA + Inverse on http://{host}:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    serve()




# ============================================================================
# cqe_time
# ============================================================================



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




# ============================================================================
# geometry_bridge
# ============================================================================



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




# ============================================================================
# TestPerformanceBenchmarks
# ============================================================================

class TestPerformanceBenchmarks(unittest.TestCase):
    """Test system performance benchmarks"""
    
    def setUp(self):
        self.cqe = UltimateCQESystem()
    
    def test_atom_creation_speed(self):
        """Test atom creation performance"""
        test_data = ["test"] * 100  # 100 identical items
        
        start_time = time.time()
        
        atom_ids = []
        for data in test_data:
            atom_id = self.cqe.create_universal_atom(data)
            atom_ids.append(atom_id)
        
        end_time = time.time()
        
        # Calculate performance metrics
        total_time = end_time - start_time
        atoms_per_second = len(test_data) / total_time
        
        # Performance should be reasonable (>100 atoms/second)
        self.assertGreater(atoms_per_second, 100)
        
        # All atoms should be created successfully
        self.assertEqual(len(atom_ids), len(test_data))
    
    def test_processing_throughput(self):
        """Test processing throughput"""
        test_data = [f"test_{i}" for i in range(50)]
        
        start_time = time.time()
        
        results = []
        for data in test_data:
            result = self.cqe.process_data_geometry_first(data)
            results.append(result)
        
        end_time = time.time()
        
        # Calculate throughput
        total_time = end_time - start_time
        operations_per_second = len(test_data) / total_time
        
        # Throughput should be reasonable (>50 operations/second)
        self.assertGreater(operations_per_second, 50)
        
        # All operations should complete successfully
        self.assertEqual(len(results), len(test_data))
    
    def test_memory_efficiency(self):
        """Test memory usage efficiency"""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create many atoms
        test_data = [f"memory_test_{i}" for i in range(1000)]
        atom_ids = []
        
        for data in test_data:
            atom_id = self.cqe.create_universal_atom(data)
            atom_ids.append(atom_id)
        
        # Get final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory per atom should be reasonable (<10KB per atom)
        memory_per_atom = memory_increase / len(test_data)
        self.assertLess(memory_per_atom, 10000)  # 10KB per atom
    
    def test_compression_efficiency(self):
        """Test compression efficiency"""
        # Test with various data types
        test_data = [
            "short",
            "this is a longer string with more content to compress",
            [1, 2, 3, 4, 5] * 10,  # Repetitive data
            {"key": "value", "nested": {"deep": "data"}},
            list(range(100)),  # Sequential data
        ]
        
        compression_ratios = []
        
        for data in test_data:
            result = self.cqe.process_data_geometry_first(data)
            ratio = result['storage_efficiency']['compression_ratio']
            compression_ratios.append(ratio)
        
        # Average compression should be reasonable (0.3 to 0.9)
        avg_compression = sum(compression_ratios) / len(compression_ratios)
        self.assertGreater(avg_compression, 0.3)
        self.assertLess(avg_compression, 0.9)




# ============================================================================
# SacredBinaryPattern
# ============================================================================

class SacredBinaryPattern(Enum):
    """Sacred geometry patterns for binary guidance"""
    INWARD_COMPRESSION = "111"      # 9-pattern: 1+1+1=3, 3*3=9
    OUTWARD_EXPANSION = "110"       # 6-pattern: 1+1+0=2, 2*3=6  
    CREATIVE_SEED = "011"           # 3-pattern: 0+1+1=2, but creative
    TRANSFORMATIVE_CYCLE = "101"    # Variable pattern: alternating
    UNITY_FOUNDATION = "001"        # 1-pattern: foundation
    DUALITY_BALANCE = "010"         # 2-pattern: balance
    STABILITY_ANCHOR = "100"        # 4-pattern: stability




# ============================================================================
# coherence_metrics
# ============================================================================



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
Coherence/Decoherence Metrics (pure stdlib)
-------------------------------------------
Geometry-first measures + embedding-alignment + dPhi guard.
This defines how we measure coherence, decoherence, and collapse events.
\"\"\"

Vec = Tuple[float, float]

def centroid(ps: List[Vec]) -> Vec:
    n = max(1, len(ps))
    return (sum(p[0] for p in ps)/n, sum(p[1] for p in ps)/n)

def v_sub(a: Vec, b: Vec) -> Vec: return (a[0]-b[0], a[1]-b[1])
def v_norm(a: Vec) -> float: return math.hypot(a[0], a[1])
def angle(a: Vec) -> float: return math.atan2(a[1], a[0])

def angular_coherence(points: List[Vec]) -> float:
    \"\"\"Circular statistic Rbar in [0,1]: 1 means perfect phase alignment.\"\"\"
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
    \"\"\"1 - Coefficient of variation of radii, clamped to [0,1].\"\"\"
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
    \"\"\"Normalized spectral entropy of a real series via naive DFT. Returns 0..1 (higher = more decoherence).\"\"\"
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
    \"\"\"Cosine similarity in [-1,1] mapped to [0,1].\"\"\"
    if not a or not b or len(a)!=len(b): return 0.0
    da = sum(x*x for x in a); db = sum(y*y for y in b)
    if da==0 or db==0: return 0.0
    dot = sum(x*y for x,y in zip(a,b))
    cos = dot / math.sqrt(da*db)
    return 0.5*(cos+1.0)

def delta_phi(before_points: List[Vec], after_points: List[Vec]) -> float:
    \"\"\"Average squared displacement between two point sets (index-aligned).\"\"\"
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




# ============================================================================
# api
# ============================================================================



DB_PATH = "./data/monster_moonshine.db"

os.makedirs("./data", exist_ok=True)
con = connect(DB_PATH)

def read_json(environ):
    try:
        length = int(environ.get('CONTENT_LENGTH', '0'))
    except (ValueError):
        length = 0
    body = environ['wsgi.input'].read(length) if length > 0 else b'{}'
    return json.loads(body.decode('utf-8') or "{}")

def respond(start_response, status: str, obj: dict, ctype="application/json"):
    data = json.dumps(obj).encode("utf-8")
    headers = [('Content-Type', ctype), ('Content-Length', str(len(data)))]
    start_response(status, headers)
    return [data]

def app(environ, start_response):
    path = environ.get('PATH_INFO', '/')
    method = environ.get('REQUEST_METHOD', 'GET')

    if path == "/api/stats":
        return respond(start_response, '200 OK', stats(con))

    if path == "/api/list":
        q = parse_qs(environ.get('QUERY_STRING', ''))
        limit = int(q.get('limit',['100'])[0]); offset = int(q.get('offset',['0'])[0])
        return respond(start_response, '200 OK', {"items": list_items(con, limit, offset)})

    if path == "/api/get":
        q = parse_qs(environ.get('QUERY_STRING', ''))
        iid = q.get('id', [''])[0]
        item = get_item(con, iid)
        if not item: return respond(start_response, '404 NOT FOUND', {"error":"not found"})
        return respond(start_response, '200 OK', item)

    if path == "/api/add" and method == "POST":
        payload = read_json(environ)
        kind = payload.get("kind","geom")
        meta = payload.get("meta",{})
        chart_names = payload.get("charts",["moonshine","geom","cqe"])
        parts = {
            "moonshine": moonshine_feature(dim=32),
            "geom": radial_angle_hist(payload.get("points", []), rbins=16, abins=16),
            "cqe": summarize_lane(meta),
        }
        vec = fuse(parts)
        item_id = payload.get("id") or str(uuid.uuid4())
        add_item(con, item_id=item_id, kind=kind, vec=vec, meta=meta, chart_names=chart_names)
        log(con, "add", {"id": item_id, "kind": kind})
        return respond(start_response, '200 OK', {"id": item_id, "dim": len(vec)})

    if path == "/api/search" and method == "POST":
        payload = read_json(environ)
        parts = {
            "moonshine": moonshine_feature(dim=32),
            "geom": radial_angle_hist(payload.get("points", []), rbins=16, abins=16),
            "cqe": summarize_lane(payload.get("meta", {})),
        }
        vec = fuse(parts)
        res = search(con, vec, topk=int(payload.get("topk",10)), chart_name=payload.get("chart"))
        return respond(start_response, '200 OK', {"results": res})

    if path == "/" or path.startswith("/static/"):
        if path == "/": path = "/static/index.html"
        try:
            with open("."+path, "rb") as f:
                data = f.read()
            ctype = "text/html"
            if path.endswith(".js"): ctype = "text/javascript"
            if path.endswith(".css"): ctype = "text/css"
            start_response('200 OK', [('Content-Type', ctype)])
            return [data]
        except Exception:
            start_response('404 NOT FOUND', [('Content-Type','text/plain')])
            return [b'Not found']

    start_response('404 NOT FOUND', [('Content-Type','text/plain')])
    return [b'Unknown route']

def serve(host="127.0.0.1", port=8765):
    httpd = make_server(host, port, app)
    print(f"Serving Monster/Moonshine DB on http://{host}:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    serve()




# ============================================================================
# inverse_residue
# ============================================================================



# Inverse/residue analysis on EM-hex gradient shifts.
# Baseline capture + delta-hex histograms + residue vs wrap heuristic.

class ResidueAnalyzer:
    def __init__(self, ca: DihedralCA):
        self.ca = ca
        self.baseline_hex = None  # list of hex strings (per pixel of global grid)

    def capture_baseline(self):
        # render entire global grid as hex map
        W,H = self.ca.W, self.ca.H
        out = ["#000000"]*(W*H)
        for y in range(H):
            for x in range(W):
                k = self.ca.idx(x,y)
                wl = self.ca.wavelength(k)
                R,G,B = wavelength_to_rgb(wl)
                out[k] = rgb_to_hex(R,G,B)
        self.baseline_hex = out

    def _hex_to_rgb(self, h: str) -> Tuple[int,int,int]:
        return int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)

    def _nibble_hist(self, hexes: List[str]) -> Dict[str, List[int]]:
        # 16-bin hist for each channel nibble high (R_hi,G_hi,B_hi)
        R=[0]*16; G=[0]*16; B=[0]*16
        for h in hexes:
            r,g,b = self._hex_to_rgb(h)
            R[r>>4]+=1; G[g>>4]+=1; B[b>>4]+=1
        return {"R":R,"G":G,"B":B}

    def residue_tile(self, tile_index: int, thresh_wrap=12) -> Dict:
        # Return per-pixel residue likelihood based on hex delta from baseline and seam-consistency test.
        if self.baseline_hex is None:
            self.capture_baseline()
        tx=tile_index%self.ca.tiles_x; ty=tile_index//self.ca.tiles_x
        w=self.ca.n; h=self.ca.n
        res_data=[]; wrap_data=[]
        # compute current hex map for tile
        curr_hex = []
        for j in range(h):
            for i in range(w):
                x=tx*w+i; y=ty*h+j; k=self.ca.idx(x,y)
                wl=self.ca.wavelength(k); R,G,B = wavelength_to_rgb(wl)
                curr_hex.append(rgb_to_hex(R,G,B))
        # residue vs wrap: measure delta from baseline and compare to neighbor across the nearest seam
        # simple heuristic: if delta to baseline is big but local difference across seam is small => wrap (continuing wave)
        # else large delta with local stationary gradient => residue.
        def l1_rgb(a,b):
            ra,ga,ba = self._hex_to_rgb(a); rb,gb,bb = self._hex_to_rgb(b)
            return abs(ra-rb)+abs(ga-gb)+abs(ba-bb)
        for j in range(h):
            for i in range(w):
                x=tx*w+i; y=ty*h+j; k=self.ca.idx(x,y)
                base = self.baseline_hex[k]; cur = curr_hex[j*w+i]
                d_hex = l1_rgb(base, cur)
                # neighbor across right seam (wrapping)
                k_right = self.ca.idx(x+1,y); base_r = self.baseline_hex[k_right]
                d_seam = l1_rgb(base_r, rgb_to_hex(*wavelength_to_rgb(self.ca.wavelength(k_right))))
                wrap_like = 1 if d_seam < thresh_wrap else 0
                # residue score: high when big change not explained by seam continuation
                score = max(0, d_hex - d_seam)
                score = 255 if score>255 else score
                res_data.extend([score,score,score,160])  # grayscale alpha
                wrap_data.extend([wrap_like*255,0,0,120]) # red marks likely wrap awaiting closure
        # nibble hist "fingerprint"
        hist = self._nibble_hist(curr_hex)
        return {"w":w,"h":h,"residue_rgba":res_data,"wrap_rgba":wrap_data,"hist":hist}




# ============================================================================
# dihedral_ca
# ============================================================================



class DihedralCA:
    def __init__(self, tiles_x=6, tiles_y=4, n=64, seed=1337):
        self.tiles_x = tiles_x; self.tiles_y = tiles_y; self.n = n
        self.W = tiles_x*n; self.H = tiles_y*n
        self.zr = [0.0]*(self.W*self.H); self.zi = [0.0]*(self.W*self.H)
        self.cr = [0.0]*(self.W*self.H); self.ci = [0.0]*(self.W*self.H)
        self.wr = [0.0]*(self.W*self.H); self.wi = [0.0]*(self.W*self.H)
        self.step_id = 0; self.rnd = random.Random(seed)
    def idx(self, x,y): x%=self.W; y%=self.H; return y*self.W + x
    def seed_from_specs(self, specs: List[str]):
        def ph(spec):
            h=0
            for ch in spec: h=(h*131+ord(ch))&0xffffffff
            return (h%360)*math.pi/180.0
        amp=0.7885
        for ty in range(self.tiles_y):
            for tx in range(self.tiles_x):
                tile=ty*self.tiles_x+tx
                phi=ph(specs[tile] if tile<len(specs) else "LEECH")
                cr=amp*math.cos(phi); ci=amp*math.sin(phi)
                for j in range(self.n):
                    for i in range(self.n):
                        x=tx*self.n+i; y=ty*self.n+j; k=self.idx(x,y)
                        self.cr[k]=cr; self.ci[k]=ci
                        self.zr[k]=0.001*math.cos((i+j)*0.1)
                        self.zi[k]=0.001*math.sin((i-j)*0.1)
                        self.wr[k]=self.zr[k]; self.wi[k]=self.zi[k]
    def neighbor_sum(self,x,y):
        s1=s2=0.0
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            k=self.idx(x+dx,y+dy); s1+=self.zr[k]; s2+=self.zi[k]
        return s1,s2
    def step(self,kappa=0.08,dual=True):
        out_zr=[0.0]*len(self.zr); out_zi=[0.0]*len(self.zi)
        out_wr=[0.0]*len(self.wr); out_wi=[0.0]*len(self.wi)
        for y in range(self.H):
            for x in range(self.W):
                k=self.idx(x,y); zr=self.zr[k]; zi=self.zi[k]; cr=self.cr[k]; ci=self.ci[k]
                nsr,nsi=self.neighbor_sum(x,y); lr=nsr-4.0*zr; li=nsi-4.0*zi
                zr2=zr*zr-zi*zi+cr+kappa*lr; zi2=2*zr*zi+ci+kappa*li
                out_zr[k]=zr2; out_zi[k]=zi2
                if dual:
                    ar=zr-cr; ai=zi-ci; r=max(0.0, (ar*ar+ai*ai))**0.5; th=math.atan2(ai,ar)
                    sr=math.sqrt(r); th2=0.5*th
                    out_wr[k]=sr*math.cos(th2); out_wi[k]=sr*math.sin(th2)
                else:
                    out_wr[k]=self.wr[k]; out_wi[k]=self.wi[k]
        self.zr,self.zi=out_zr,out_zi; self.wr,self.wi=out_wr,out_wi; self.step_id+=1
    def tile_pixels_em(self,tile_index:int,alpha:int=160)->Dict:
        tx=tile_index%self.tiles_x; ty=tile_index//self.tiles_x
        w=self.n; h=self.n; data=[]
        for j in range(h):
            for i in range(w):
                x=tx*self.n+i; y=ty*self.n+j; k=self.idx(x,y)
                r1=(self.zr[k]*self.zr[k]+self.zi[k]*self.zi[k])**0.5
                r2=(self.wr[k]*self.wr[k]+self.wi[k]*self.wi[k])**0.5
                r=0.6*r1+0.4*r2; th=math.atan2(self.zi[k],self.zr[k])
                wl=380.0+400.0*(math.tanh(0.5*r))
                R,G,B=wavelength_to_rgb(wl); band=0.5*(1.0+math.cos(6.0*th))
                R=int(R*band); G=int(G*band); B=int(B*band)
                data.extend([R,G,B,alpha])
        return {"w":w,"h":h,"rgba":data}
def wavelength_to_rgb(wl: float):
    if wl<380: wl=380
    if wl>780: wl=780
    def clamp(x): return 0 if x<0 else (1 if x>1 else x)
    if wl<440: t=(wl-380)/(440-380); R,G,B=(clamp(1.0-t),0.0,1.0)
    elif wl<490: t=(wl-440)/(490-440); R,G,B=(0.0,clamp(t),1.0)
    elif wl<510: t=(wl-490)/(510-490); R,G,B=(0.0,1.0,clamp(1.0-t))
    elif wl<580: t=(wl-510)/(580-510); R,G,B=(clamp(t),1.0,0.0)
    elif wl<645: t=(wl-580)/(645-580); R,G,B=(1.0,clamp(1.0-t),0.0)
    else: t=(wl-645)/(780-645); R,G,B=(1.0,0.0,clamp(0.3*(1.0-t)))
    if wl<420: f=0.3+0.7*(wl-380)/(420-380)
    elif wl>700: f=0.3+0.7*(780-wl)/(780-700)
    else: f=1.0
    return (int(255*R*f), int(255*G*f), int(255*B*f))




# ============================================================================
# test_beta_eta_delta_mu
# ============================================================================



def test_beta():
    e = A.App(A.Lam("x", A.Var("x")), A.Const("nat", 7))
    v, steps = E.eval_normal(e)
    assert isinstance(v, A.Const) and v.value == 7

def test_delta_succ():
    e = A.App(A.Const("succ", None), A.Const("nat", 2))
    v, steps = E.eval_normal(e)
    assert isinstance(v, A.Const) and v.value == 3

def test_pair_proj():
    e = A.Fst(A.Pair(A.Const("nat", 1), A.Const("nat", 2)))
    v, steps = E.eval_normal(e)
    assert isinstance(v, A.Const) and v.value == 1

def test_mu_unrolls():
    # μx.x -> diverges by unrolling until fuel ends; we test single step occurs.
    v, steps = E.eval_normal(A.Mu("x", A.Var("x")), fuel=1)
    assert steps == 1




# ============================================================================
# ToroidalRotationType
# ============================================================================

class ToroidalRotationType(Enum):
    """Types of rotations around toroidal shell"""
    POLOIDAL = "POLOIDAL"          # Around minor circumference (inward/9-pattern)
    TOROIDAL = "TOROIDAL"          # Around major circumference (outward/6-pattern)
    MERIDIONAL = "MERIDIONAL"      # Through torus center (creative/3-pattern)
    HELICAL = "HELICAL"            # Spiral combination (transformative)




# ============================================================================
# CQEOperationMode
# ============================================================================

class CQEOperationMode(Enum):
    """CQE system operation modes"""
    BASIC = "BASIC"
    ENHANCED = "ENHANCED"
    ULTIMATE_UNIFIED = "ULTIMATE_UNIFIED"
    SACRED_GEOMETRY = "SACRED_GEOMETRY"
    MANDELBROT_FRACTAL = "MANDELBROT_FRACTAL"
    TOROIDAL_ANALYSIS = "TOROIDAL_ANALYSIS"




# ============================================================================
# RotationMode
# ============================================================================

class RotationMode(Enum):
    """Four fundamental rotation modes corresponding to four forces"""
    POLOIDAL = "electromagnetic"     # Rotation around minor axis
    TOROIDAL = "weak_nuclear"        # Rotation around major axis
    MERIDIONAL = "strong_nuclear"    # Rotation in meridional plane
    HELICAL = "gravitational"        # Combined rotation (all modes)




# ============================================================================
# viewer_api
# ============================================================================



SESSION = {"points": [], "meta": {}}
TILES_X, TILES_Y = 6, 4
N = 64
CA = DihedralCA(tiles_x=TILES_X, tiles_y=TILES_Y, n=N, seed=1337)
CA.seed_from_specs(NIEMEIER_SPECS + ["LEECH"])

def read_json(environ):
    try: length = int(environ.get('CONTENT_LENGTH','0'))
    except (ValueError): length = 0
    body = environ['wsgi.input'].read(length) if length>0 else b'{}'
    return json.loads(body.decode('utf-8') or "{}")

def respond(start_response, status, obj, ctype="application/json"):
    data = json.dumps(obj).encode("utf-8")
    start_response(status, [('Content-Type', ctype), ('Content-Length', str(len(data)))])
    return [data]

def app(environ, start_response):
    path = environ.get('PATH_INFO','/'); method = environ.get('REQUEST_METHOD','GET')

    if path == "/api/load" and method == "POST":
        payload = read_json(environ); SESSION["points"]=payload.get("points") or []; SESSION["meta"]=payload.get("meta") or {}
        return respond(start_response,'200 OK',{"ok":True,"count":len(SESSION["points"])})
    if path == "/api/screens":
        labs = NIEMEIER_SPECS + ["LEECH"]
        return respond(start_response,'200 OK',{"screens":[{"index":i,"label":lab} for i,lab in enumerate(labs)]})
    if path == "/api/frame":
        q = parse_qs(environ.get('QUERY_STRING','')); w=int(q.get('w',['320'])[0]); h=int(q.get('h',['180'])[0])
        s,tx,ty = world_to_screen(SESSION.get("points") or [], w, h, padding=0.08)
        return respond(start_response,'200 OK',{"s":s,"tx":tx,"ty":ty})
    if path == "/api/ca/init":
        q = parse_qs(environ.get('QUERY_STRING','')); n=int(q.get('n',['64'])[0])
        global CA,N; N=n; CA=DihedralCA(tiles_x=TILES_X, tiles_y=TILES_Y, n=N, seed=1337); CA.seed_from_specs(NIEMEIER_SPECS+["LEECH"])
        return respond(start_response,'200 OK',{"ok":True,"n":N})
    if path == "/api/ca/step":
        q = parse_qs(environ.get('QUERY_STRING','')); steps=int(q.get('steps',['1'])[0]); kappa=float(q.get('kappa',['0.08'])[0])
        for _ in range(steps): CA.step(kappa=kappa, dual=True)
        return respond(start_response,'200 OK',{"ok":True,"step":CA.step_id})
    if path == "/api/ca/tile":
        q = parse_qs(environ.get('QUERY_STRING','')); idx=int(q.get('index',['0'])[0]); alpha=int(q.get('alpha',['160'])[0])
        tile = CA.tile_pixels_em(idx, alpha=alpha); return respond(start_response,'200 OK',tile)
    if path == "/":
        with open("./static/index.html","rb") as f: data=f.read()
        start_response('200 OK',[('Content-Type','text/html')]); return [data]
    if path.startswith("/static/"):
        p = "."+path
        if not os.path.exists(p): start_response('404 NOT FOUND',[('Content-Type','text/plain')]); return [b'not found']
        ctype="text/plain"
        if p.endswith(".html"): ctype="text/html"
        if p.endswith(".js"): ctype="text/javascript"
        if p.endswith(".css"): ctype="text/css"
        with open(p,"rb") as f: data=f.read()
        start_response('200 OK',[('Content-Type',ctype)]); return [data]
    start_response('404 NOT FOUND',[('Content-Type','application/json')]); return [b'{}']

def serve(host="127.0.0.1", port=8989):
    httpd = make_server(host, port, app)
    print(f"Viewer24 Controller v2 + CA on http://{host}:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    serve()




# ============================================================================
# CQEConfiguration
# ============================================================================

class CQEConfiguration:
    """Configuration for CQE system"""
    operation_mode: CQEOperationMode = CQEOperationMode.ULTIMATE_UNIFIED
    processing_priority: ProcessingPriority = ProcessingPriority.GEOMETRY_FIRST
    enable_sacred_geometry: bool = True
    enable_mandelbrot_storage: bool = True
    enable_toroidal_geometry: bool = True
    enable_validation: bool = True
    max_iterations: int = 1000
    precision_threshold: float = 1e-10
    memory_optimization: bool = True
    parallel_processing: bool = True
    log_level: str = "INFO"

@dataclass



# ============================================================================
# dihedral_ca_1
# ============================================================================



class DihedralCA:
    def __init__(self, tiles_x=6, tiles_y=4, n=64, seed=1337):
        self.tiles_x = tiles_x; self.tiles_y = tiles_y; self.n = n
        self.W = tiles_x*n; self.H = tiles_y*n
        self.zr = [0.0]*(self.W*self.H); self.zi = [0.0]*(self.W*self.H)
        self.cr = [0.0]*(self.W*self.H); self.ci = [0.0]*(self.W*self.H)
        self.wr = [0.0]*(self.W*self.H); self.wi = [0.0]*(self.W*self.H)
        self.step_id = 0; self.rnd = random.Random(seed)
    def idx(self, x,y): x%=self.W; y%=self.H; return y*self.W + x
    def seed_from_specs(self, specs: List[str]):
        def ph(spec):
            h=0
            for ch in spec: h=(h*131+ord(ch))&0xffffffff
            return (h%360)*math.pi/180.0
        amp=0.7885
        for ty in range(self.tiles_y):
            for tx in range(self.tiles_x):
                tile=ty*self.tiles_x+tx
                phi=ph(specs[tile] if tile<len(specs) else "LEECH")
                cr=amp*math.cos(phi); ci=amp*math.sin(phi)
                for j in range(self.n):
                    for i in range(self.n):
                        x=tx*self.n+i; y=ty*self.n+j; k=self.idx(x,y)
                        self.cr[k]=cr; self.ci[k]=ci
                        self.zr[k]=0.001*math.cos((i+j)*0.1)
                        self.zi[k]=0.001*math.sin((i-j)*0.1)
                        self.wr[k]=self.zr[k]; self.wi[k]=self.zi[k]
    def neighbor_sum(self,x,y):
        s1=s2=0.0
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            k=self.idx(x+dx,y+dy); s1+=self.zr[k]; s2+=self.zi[k]
        return s1,s2
    def step(self,kappa=0.08,dual=True):
        out_zr=[0.0]*len(self.zr); out_zi=[0.0]*len(self.zi)
        out_wr=[0.0]*len(self.wr); out_wi=[0.0]*len(self.wi)
        for y in range(self.H):
            for x in range(self.W):
                k=self.idx(x,y); zr=self.zr[k]; zi=self.zi[k]; cr=self.cr[k]; ci=self.ci[k]
                nsr,nsi=self.neighbor_sum(x,y); lr=nsr-4.0*zr; li=nsi-4.0*zi
                zr2=zr*zr-zi*zi+cr+kappa*lr; zi2=2*zr*zi+ci+kappa*li
                out_zr[k]=zr2; out_zi[k]=zi2
                if dual:
                    ar=zr-cr; ai=zi-ci; r=max(0.0, (ar*ar+ai*ai))**0.5; th=math.atan2(ai,ar)
                    sr=math.sqrt(r); th2=0.5*th
                    out_wr[k]=sr*math.cos(th2); out_wi[k]=sr*math.sin(th2)
                else:
                    out_wr[k]=self.wr[k]; out_wi[k]=self.wi[k]
        self.zr,self.zi=out_zr,out_zi; self.wr,self.wi=out_wr,out_wi; self.step_id+=1
    def wavelength(self,k):
        r1=(self.zr[k]*self.zr[k]+self.zi[k]*self.zi[k])**0.5
        return 380.0+400.0*(math.tanh(0.5*r1))
    def tile_pixels_em(self,tile_index:int,alpha:int=160)->Dict:
        tx=tile_index%self.tiles_x; ty=tile_index//self.tiles_x
        w=self.n; h=self.n; data=[]; hexes=[]
        for j in range(h):
            for i in range(w):
                x=tx*self.n+i; y=ty*self.n+j; k=self.idx(x,y)
                wl=self.wavelength(k)
                R,G,B=wavelength_to_rgb(wl)
                data.extend([R,G,B,alpha])
                hexes.append(rgb_to_hex(R,G,B))
        return {"w":w,"h":h,"rgba":data,"hex":hexes}
def wavelength_to_rgb(wl: float):
    if wl<380: wl=380
    if wl>780: wl=780
    def clamp(x): return 0 if x<0 else (1 if x>1 else x)
    if wl<440: t=(wl-380)/(440-380); R,G,B=(clamp(1.0-t),0.0,1.0)
    elif wl<490: t=(wl-440)/(490-440); R,G,B=(0.0,clamp(t),1.0)
    elif wl<510: t=(wl-490)/(510-490); R,G,B=(0.0,1.0,clamp(1.0-t))
    elif wl<580: t=(wl-510)/(580-510); R,G,B=(clamp(t),1.0,0.0)
    elif wl<645: t=(wl-580)/(645-580); R,G,B=(1.0,clamp(1.0-t),0.0)
    else: t=(wl-645)/(780-645); R,G,B=(1.0,0.0,clamp(0.3*(1.0-t)))
    if wl<420: f=0.3+0.7*(wl-380)/(420-380)
    elif wl>700: f=0.3+0.7*(780-wl)/(780-700)
    else: f=1.0
    return (int(255*R*f), int(255*G*f), int(255*B*f))
def rgb_to_hex(R,G,B):
    return "#{:02X}{:02X}{:02X}".format(max(0,min(255,R)), max(0,min(255,G)), max(0,min(255,B)))




# ============================================================================
# CQELanguageEngine
# ============================================================================

class CQELanguageEngine:
    """Universal language processing engine using CQE principles"""
    
    def __init__(self, kernel: CQEKernel):
        self.kernel = kernel
        self.language_patterns: Dict[str, LanguagePattern] = {}
        self.language_rules: Dict[str, LanguageRule] = {}
        self.language_models: Dict[str, Dict[str, Any]] = {}
        
        # Language detection and classification
        self.language_detectors: Dict[LanguageType, Callable] = {}
        self.syntax_analyzers: Dict[SyntaxLevel, Callable] = {}
        self.semantic_processors: Dict[str, Callable] = {}
        
        # Universal language features
        self.universal_patterns = {}
        self.cross_language_mappings = defaultdict(dict)
        
        # Initialize language processing components
        self._initialize_language_detectors()
        self._initialize_syntax_analyzers()
        self._initialize_semantic_processors()
        self._initialize_universal_patterns()
    
    def _initialize_language_detectors(self):
        """Initialize language detection functions"""
        self.language_detectors = {
            LanguageType.NATURAL: self._detect_natural_language,
            LanguageType.PROGRAMMING: self._detect_programming_language,
            LanguageType.MARKUP: self._detect_markup_language,
            LanguageType.FORMAL: self._detect_formal_language,
            LanguageType.SYMBOLIC: self._detect_symbolic_language,
            LanguageType.CONSTRUCTED: self._detect_constructed_language
        }
    
    def _initialize_syntax_analyzers(self):
        """Initialize syntax analysis functions"""
        self.syntax_analyzers = {
            SyntaxLevel.PHONETIC: self._analyze_phonetic,
            SyntaxLevel.MORPHEMIC: self._analyze_morphemic,
            SyntaxLevel.SYNTACTIC: self._analyze_syntactic,
            SyntaxLevel.SEMANTIC: self._analyze_semantic,
            SyntaxLevel.PRAGMATIC: self._analyze_pragmatic,
            SyntaxLevel.DISCOURSE: self._analyze_discourse
        }
    
    def _initialize_semantic_processors(self):
        """Initialize semantic processing functions"""
        self.semantic_processors = {
            'entity_extraction': self._extract_entities,
            'relation_extraction': self._extract_relations,
            'sentiment_analysis': self._analyze_sentiment,
            'intent_detection': self._detect_intent,
            'concept_mapping': self._map_concepts,
            'meaning_representation': self._represent_meaning
        }
    
    def _initialize_universal_patterns(self):
        """Initialize universal language patterns"""
        # Universal syntactic patterns that appear across languages
        self.universal_patterns = {
            'subject_verb_object': {
                'quad_signature': (1, 2, 3, 1),
                'description': 'Basic SVO sentence structure',
                'languages': ['english', 'chinese', 'spanish', 'french']
            },
            'question_formation': {
                'quad_signature': (4, 1, 2, 3),
                'description': 'Question formation patterns',
                'languages': ['english', 'german', 'russian']
            },
            'negation': {
                'quad_signature': (2, 4, 2, 4),
                'description': 'Negation patterns',
                'languages': ['universal']
            },
            'conditional': {
                'quad_signature': (3, 1, 4, 2),
                'description': 'Conditional/if-then structures',
                'languages': ['universal']
            },
            'recursion': {
                'quad_signature': (1, 3, 1, 3),
                'description': 'Recursive/nested structures',
                'languages': ['universal']
            }
        }
    
    def process_text(self, text: str, language_hint: Optional[str] = None,
                    analysis_levels: List[SyntaxLevel] = None) -> List[str]:
        """Process text through CQE language analysis"""
        if analysis_levels is None:
            analysis_levels = list(SyntaxLevel)
        
        # Detect language type
        language_type = self._detect_language_type(text, language_hint)
        
        # Create text atom
        text_atom = CQEAtom(
            data={
                'text': text,
                'language_type': language_type.value,
                'language_hint': language_hint,
                'processing_timestamp': time.time()
            },
            metadata={'language_engine': True, 'text_input': True}
        )
        
        text_atom_id = self.kernel.memory_manager.store_atom(text_atom)
        result_atom_ids = [text_atom_id]
        
        # Process through each analysis level
        for level in analysis_levels:
            if level in self.syntax_analyzers:
                analyzer = self.syntax_analyzers[level]
                analysis_result = analyzer(text, language_type)
                
                # Create analysis atom
                analysis_atom = CQEAtom(
                    data={
                        'analysis_level': level.value,
                        'language_type': language_type.value,
                        'result': analysis_result,
                        'source_text': text[:100]  # Truncated for reference
                    },
                    parent_id=text_atom_id,
                    metadata={'analysis_level': level.value, 'language_type': language_type.value}
                )
                
                analysis_atom_id = self.kernel.memory_manager.store_atom(analysis_atom)
                result_atom_ids.append(analysis_atom_id)
        
        # Extract and store language patterns
        patterns = self._extract_patterns(text, language_type)
        for pattern in patterns:
            pattern_atom = CQEAtom(
                data=pattern,
                parent_id=text_atom_id,
                metadata={'pattern': True, 'language_type': language_type.value}
            )
            
            pattern_atom_id = self.kernel.memory_manager.store_atom(pattern_atom)
            result_atom_ids.append(pattern_atom_id)
        
        return result_atom_ids
    
    def translate_between_languages(self, source_text: str, source_lang: str,
                                  target_lang: str) -> str:
        """Translate between languages using CQE universal patterns"""
        # Process source text
        source_atoms = self.process_text(source_text, source_lang)
        
        # Extract universal patterns
        universal_representation = self._extract_universal_representation(source_atoms)
        
        # Generate target language text
        target_text = self._generate_from_universal(universal_representation, target_lang)
        
        return target_text
    
    def analyze_syntax_diversity(self, texts: List[str], languages: List[str] = None) -> Dict[str, Any]:
        """Analyze syntax diversity across multiple texts/languages"""
        if languages is None:
            languages = [None] * len(texts)
        
        diversity_analysis = {
            'total_texts': len(texts),
            'pattern_distribution': defaultdict(int),
            'universal_patterns': defaultdict(int),
            'language_specific_patterns': defaultdict(lambda: defaultdict(int)),
            'cross_language_similarities': {},
            'syntax_complexity': []
        }
        
        all_patterns = []
        
        for text, lang_hint in zip(texts, languages):
            # Process text
            atom_ids = self.process_text(text, lang_hint)
            
            # Extract patterns from atoms
            for atom_id in atom_ids:
                atom = self.kernel.memory_manager.retrieve_atom(atom_id)
                if atom and atom.metadata.get('pattern'):
                    pattern_data = atom.data
                    all_patterns.append(pattern_data)
                    
                    # Update distribution
                    pattern_type = pattern_data.get('type', 'unknown')
                    diversity_analysis['pattern_distribution'][pattern_type] += 1
                    
                    # Check for universal patterns
                    if pattern_data.get('universal', False):
                        diversity_analysis['universal_patterns'][pattern_type] += 1
                    
                    # Language-specific patterns
                    lang_type = pattern_data.get('language_type', 'unknown')
                    diversity_analysis['language_specific_patterns'][lang_type][pattern_type] += 1
        
        # Calculate complexity metrics
        for text in texts:
            complexity = self._calculate_syntax_complexity(text)
            diversity_analysis['syntax_complexity'].append(complexity)
        
        # Calculate cross-language similarities
        diversity_analysis['cross_language_similarities'] = self._calculate_cross_language_similarities(all_patterns)
        
        return diversity_analysis
    
    def create_universal_grammar(self, training_texts: List[str], 
                               languages: List[str]) -> Dict[str, Any]:
        """Create universal grammar from multiple languages"""
        universal_grammar = {
            'universal_rules': [],
            'pattern_mappings': {},
            'transformation_rules': {},
            'semantic_universals': {},
            'syntactic_universals': {}
        }
        
        # Process all training texts
        all_patterns = []
        language_patterns = defaultdict(list)
        
        for text, lang in zip(training_texts, languages):
            atom_ids = self.process_text(text, lang)
            
            for atom_id in atom_ids:
                atom = self.kernel.memory_manager.retrieve_atom(atom_id)
                if atom and atom.metadata.get('pattern'):
                    pattern = atom.data
                    all_patterns.append(pattern)
                    language_patterns[lang].append(pattern)
        
        # Extract universal patterns
        universal_grammar['universal_rules'] = self._extract_universal_rules(all_patterns)
        
        # Create pattern mappings between languages
        universal_grammar['pattern_mappings'] = self._create_pattern_mappings(language_patterns)
        
        # Extract transformation rules
        universal_grammar['transformation_rules'] = self._extract_transformation_rules(language_patterns)
        
        # Identify semantic and syntactic universals
        universal_grammar['semantic_universals'] = self._identify_semantic_universals(all_patterns)
        universal_grammar['syntactic_universals'] = self._identify_syntactic_universals(all_patterns)
        
        return universal_grammar
    
    def generate_text(self, intent: str, target_language: str, 
                     style: str = "neutral", constraints: Dict[str, Any] = None) -> str:
        """Generate text in target language using CQE principles"""
        if constraints is None:
            constraints = {}
        
        # Create intent representation
        intent_atom = CQEAtom(
            data={
                'intent': intent,
                'target_language': target_language,
                'style': style,
                'constraints': constraints
            },
            metadata={'generation_request': True}
        )
        
        # Process intent through semantic analysis
        semantic_representation = self._analyze_semantic(intent, LanguageType.NATURAL)
        
        # Map to universal patterns
        universal_patterns = self._map_to_universal_patterns(semantic_representation)
        
        # Generate in target language
        generated_text = self._generate_from_patterns(universal_patterns, target_language, style)
        
        # Apply constraints
        if constraints:
            generated_text = self._apply_generation_constraints(generated_text, constraints)
        
        return generated_text
    
    # Language Detection Functions
    def _detect_language_type(self, text: str, hint: Optional[str] = None) -> LanguageType:
        """Detect the type of language"""
        if hint:
            # Use hint to guide detection
            hint_lower = hint.lower()
            if hint_lower in ['python', 'javascript', 'java', 'c++', 'c', 'go', 'rust']:
                return LanguageType.PROGRAMMING
            elif hint_lower in ['html', 'xml', 'markdown', 'latex']:
                return LanguageType.MARKUP
            elif hint_lower in ['logic', 'math', 'formal']:
                return LanguageType.FORMAL
        
        # Automatic detection
        for lang_type, detector in self.language_detectors.items():
            if detector(text):
                return lang_type
        
        return LanguageType.NATURAL  # Default
    
    def _detect_natural_language(self, text: str) -> bool:
        """Detect natural language"""
        # Check for natural language characteristics
        word_count = len(text.split())
        alpha_ratio = sum(1 for c in text if c.isalpha()) / max(1, len(text))
        
        return word_count > 3 and alpha_ratio > 0.6
    
    def _detect_programming_language(self, text: str) -> bool:
        """Detect programming language"""
        # Check for programming language patterns
        programming_indicators = [
            r'\bdef\b', r'\bclass\b', r'\bfunction\b', r'\bvar\b', r'\blet\b', r'\bconst\b',
            r'\bif\b', r'\belse\b', r'\bfor\b', r'\bwhile\b', r'\breturn\b',
            r'[{}();]', r'==', r'!=', r'<=', r'>='
        ]
        
        matches = sum(1 for pattern in programming_indicators 
                     if re.search(pattern, text, re.IGNORECASE))
        
        return matches >= 3
    
    def _detect_markup_language(self, text: str) -> bool:
        """Detect markup language"""
        # Check for markup patterns
        markup_patterns = [r'<[^>]+>', r'\[([^\]]+)\]\([^)]+\)', r'#+\s', r'\*\*[^*]+\*\*']
        
        matches = sum(1 for pattern in markup_patterns if re.search(pattern, text))
        
        return matches >= 2
    
    def _detect_formal_language(self, text: str) -> bool:
        """Detect formal language"""
        # Check for formal language symbols
        formal_symbols = ['∀', '∃', '∧', '∨', '¬', '→', '↔', '∈', '∉', '⊂', '⊃', '∪', '∩']
        math_symbols = ['∑', '∏', '∫', '∂', '∇', '∞', '±', '≈', '≡', '≤', '≥']
        
        symbol_count = sum(1 for symbol in formal_symbols + math_symbols if symbol in text)
        
        return symbol_count >= 3
    
    def _detect_symbolic_language(self, text: str) -> bool:
        """Detect symbolic language"""
        # Check for symbolic notation
        symbolic_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(1, len(text))
        
        return symbolic_ratio > 0.3
    
    def _detect_constructed_language(self, text: str) -> bool:
        """Detect constructed language"""
        # This would require more sophisticated analysis
        # For now, return False
        return False
    
    # Syntax Analysis Functions
    def _analyze_phonetic(self, text: str, language_type: LanguageType) -> Dict[str, Any]:
        """Analyze phonetic/character level"""
        analysis = {
            'character_count': len(text),
            'character_distribution': dict(Counter(text.lower())),
            'unicode_categories': {},
            'phonetic_patterns': []
        }
        
        # Unicode category analysis
        for char in text:
            category = unicodedata.category(char)
            analysis['unicode_categories'][category] = analysis['unicode_categories'].get(category, 0) + 1
        
        # Extract phonetic patterns (simplified)
        if language_type == LanguageType.NATURAL:
            # Consonant-vowel patterns
            vowels = 'aeiouAEIOU'
            cv_pattern = ''.join('V' if c in vowels else 'C' if c.isalpha() else c for c in text)
            analysis['cv_pattern'] = cv_pattern[:100]  # Truncate for storage
        
        return analysis
    
    def _analyze_morphemic(self, text: str, language_type: LanguageType) -> Dict[str, Any]:
        """Analyze morphemic/word level"""
        words = text.split()
        
        analysis = {
            'word_count': len(words),
            'unique_words': len(set(words)),
            'word_length_distribution': dict(Counter(len(word) for word in words)),
            'morphological_patterns': [],
            'token_types': {}
        }
        
        # Analyze word patterns
        for word in words:
            # Simple morphological analysis
            if word.endswith('ing'):
                analysis['morphological_patterns'].append('present_participle')
            elif word.endswith('ed'):
                analysis['morphological_patterns'].append('past_tense')
            elif word.endswith('ly'):
                analysis['morphological_patterns'].append('adverb')
            elif word.endswith('tion'):
                analysis['morphological_patterns'].append('nominalization')
        
        # Token type analysis
        for word in words:
            if word.isdigit():
                analysis['token_types']['number'] = analysis['token_types'].get('number', 0) + 1
            elif word.isalpha():
                analysis['token_types']['word'] = analysis['token_types'].get('word', 0) + 1
            elif not word.isalnum():
                analysis['token_types']['punctuation'] = analysis['token_types'].get('punctuation', 0) + 1
        
        return analysis
    
    def _analyze_syntactic(self, text: str, language_type: LanguageType) -> Dict[str, Any]:
        """Analyze syntactic/sentence level"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        analysis = {
            'sentence_count': len(sentences),
            'sentence_length_distribution': dict(Counter(len(s.split()) for s in sentences)),
            'syntactic_patterns': [],
            'clause_types': {},
            'dependency_patterns': []
        }
        
        # Analyze sentence patterns
        for sentence in sentences:
            words = sentence.split()
            if not words:
                continue
            
            # Simple syntactic pattern detection
            if words[0].lower() in ['what', 'who', 'where', 'when', 'why', 'how']:
                analysis['syntactic_patterns'].append('wh_question')
            elif words[0].lower() in ['is', 'are', 'was', 'were', 'do', 'does', 'did']:
                analysis['syntactic_patterns'].append('yes_no_question')
            elif words[-1] == '?':
                analysis['syntactic_patterns'].append('question')
            elif words[-1] == '!':
                analysis['syntactic_patterns'].append('exclamation')
            else:
                analysis['syntactic_patterns'].append('declarative')
        
        return analysis
    
    def _analyze_semantic(self, text: str, language_type: LanguageType) -> Dict[str, Any]:
        """Analyze semantic/meaning level"""
        analysis = {
            'semantic_fields': [],
            'entities': [],
            'relations': [],
            'concepts': [],
            'semantic_roles': {}
        }
        
        # Simple semantic analysis
        words = text.lower().split()
        
        # Semantic field detection (simplified)
        semantic_fields = {
            'technology': ['computer', 'software', 'algorithm', 'data', 'system'],
            'science': ['research', 'study', 'analysis', 'experiment', 'theory'],
            'business': ['company', 'market', 'customer', 'product', 'service'],
            'emotion': ['happy', 'sad', 'angry', 'excited', 'worried']
        }
        
        for field, keywords in semantic_fields.items():
            if any(keyword in words for keyword in keywords):
                analysis['semantic_fields'].append(field)
        
        # Entity extraction (simplified)
        # This would use more sophisticated NER in practice
        capitalized_words = [word for word in text.split() if word[0].isupper() and len(word) > 1]
        analysis['entities'] = capitalized_words[:10]  # Limit for storage
        
        return analysis
    
    def _analyze_pragmatic(self, text: str, language_type: LanguageType) -> Dict[str, Any]:
        """Analyze pragmatic/context level"""
        analysis = {
            'speech_acts': [],
            'politeness_markers': [],
            'discourse_markers': [],
            'register': 'neutral',
            'formality': 'medium'
        }
        
        text_lower = text.lower()
        
        # Speech act detection
        if any(word in text_lower for word in ['please', 'could you', 'would you']):
            analysis['speech_acts'].append('request')
        if any(word in text_lower for word in ['thank', 'thanks', 'grateful']):
            analysis['speech_acts'].append('gratitude')
        if any(word in text_lower for word in ['sorry', 'apologize', 'excuse']):
            analysis['speech_acts'].append('apology')
        
        # Politeness markers
        politeness_markers = ['please', 'thank you', 'excuse me', 'sorry', 'pardon']
        for marker in politeness_markers:
            if marker in text_lower:
                analysis['politeness_markers'].append(marker)
        
        # Discourse markers
        discourse_markers = ['however', 'therefore', 'moreover', 'furthermore', 'nevertheless']
        for marker in discourse_markers:
            if marker in text_lower:
                analysis['discourse_markers'].append(marker)
        
        return analysis
    
    def _analyze_discourse(self, text: str, language_type: LanguageType) -> Dict[str, Any]:
        """Analyze discourse/document level"""
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        analysis = {
            'paragraph_count': len(paragraphs),
            'discourse_structure': [],
            'coherence_markers': [],
            'topic_progression': [],
            'rhetorical_structure': {}
        }
        
        # Analyze discourse structure
        for i, paragraph in enumerate(paragraphs):
            if i == 0:
                analysis['discourse_structure'].append('introduction')
            elif i == len(paragraphs) - 1:
                analysis['discourse_structure'].append('conclusion')
            else:
                analysis['discourse_structure'].append('body')
        
        # Coherence markers
        coherence_markers = ['first', 'second', 'finally', 'in conclusion', 'to summarize']
        for marker in coherence_markers:
            if marker in text.lower():
                analysis['coherence_markers'].append(marker)
        
        return analysis
    
    # Pattern Extraction and Processing
    def _extract_patterns(self, text: str, language_type: LanguageType) -> List[Dict[str, Any]]:
        """Extract language patterns from text"""
        patterns = []
        
        # Extract universal patterns
        for pattern_name, pattern_info in self.universal_patterns.items():
            if self._matches_universal_pattern(text, pattern_name, pattern_info):
                patterns.append({
                    'type': pattern_name,
                    'universal': True,
                    'quad_signature': pattern_info['quad_signature'],
                    'description': pattern_info['description'],
                    'language_type': language_type.value,
                    'confidence': 0.8
                })
        
        # Extract language-specific patterns
        specific_patterns = self._extract_language_specific_patterns(text, language_type)
        patterns.extend(specific_patterns)
        
        return patterns
    
    def _matches_universal_pattern(self, text: str, pattern_name: str, pattern_info: Dict[str, Any]) -> bool:
        """Check if text matches a universal pattern"""
        # Simplified pattern matching
        if pattern_name == 'subject_verb_object':
            # Look for SVO structure
            words = text.split()
            return len(words) >= 3 and any(word.lower() in ['is', 'are', 'was', 'were', 'has', 'have'] for word in words)
        
        elif pattern_name == 'question_formation':
            return text.strip().endswith('?') or text.lower().startswith(('what', 'who', 'where', 'when', 'why', 'how'))
        
        elif pattern_name == 'negation':
            return any(neg in text.lower() for neg in ['not', 'no', 'never', 'nothing', 'nobody'])
        
        elif pattern_name == 'conditional':
            return any(cond in text.lower() for cond in ['if', 'when', 'unless', 'provided'])
        
        elif pattern_name == 'recursion':
            # Look for nested structures
            return '(' in text and ')' in text or '[' in text and ']' in text
        
        return False
    
    def _extract_language_specific_patterns(self, text: str, language_type: LanguageType) -> List[Dict[str, Any]]:
        """Extract language-specific patterns"""
        patterns = []
        
        if language_type == LanguageType.PROGRAMMING:
            # Programming language patterns
            if re.search(r'\bdef\s+\w+\s*\(', text):
                patterns.append({
                    'type': 'function_definition',
                    'universal': False,
                    'quad_signature': (1, 4, 2, 3),
                    'language_type': language_type.value,
                    'confidence': 0.9
                })
            
            if re.search(r'\bclass\s+\w+', text):
                patterns.append({
                    'type': 'class_definition',
                    'universal': False,
                    'quad_signature': (2, 1, 4, 3),
                    'language_type': language_type.value,
                    'confidence': 0.9
                })
        
        elif language_type == LanguageType.MARKUP:
            # Markup language patterns
            if re.search(r'<\w+[^>]*>', text):
                patterns.append({
                    'type': 'tag_structure',
                    'universal': False,
                    'quad_signature': (3, 2, 1, 4),
                    'language_type': language_type.value,
                    'confidence': 0.8
                })
        
        return patterns
    
    # Universal Language Processing
    def _extract_universal_representation(self, atom_ids: List[str]) -> Dict[str, Any]:
        """Extract universal representation from processed atoms"""
        universal_rep = {
            'semantic_structure': {},
            'syntactic_patterns': [],
            'universal_patterns': [],
            'meaning_components': []
        }
        
        for atom_id in atom_ids:
            atom = self.kernel.memory_manager.retrieve_atom(atom_id)
            if not atom:
                continue
            
            if atom.metadata.get('analysis_level') == 'semantic':
                universal_rep['semantic_structure'].update(atom.data.get('result', {}))
            
            elif atom.metadata.get('pattern'):
                pattern_data = atom.data
                if pattern_data.get('universal'):
                    universal_rep['universal_patterns'].append(pattern_data)
                else:
                    universal_rep['syntactic_patterns'].append(pattern_data)
        
        return universal_rep
    
    def _generate_from_universal(self, universal_rep: Dict[str, Any], target_lang: str) -> str:
        """Generate text from universal representation"""
        # Simplified generation - in practice would use sophisticated generation models
        
        # Start with universal patterns
        generated_parts = []
        
        for pattern in universal_rep.get('universal_patterns', []):
            pattern_type = pattern.get('type')
            
            if pattern_type == 'subject_verb_object':
                if target_lang.lower() == 'spanish':
                    generated_parts.append("El sujeto verbo objeto")
                elif target_lang.lower() == 'french':
                    generated_parts.append("Le sujet verbe objet")
                else:
                    generated_parts.append("The subject verb object")
            
            elif pattern_type == 'question_formation':
                if target_lang.lower() == 'spanish':
                    generated_parts.append("¿Qué?")
                elif target_lang.lower() == 'french':
                    generated_parts.append("Qu'est-ce que?")
                else:
                    generated_parts.append("What?")
        
        # Combine parts
        if generated_parts:
            return ' '.join(generated_parts)
        else:
            return f"Generated text in {target_lang}"
    
    # Utility Functions
    def _calculate_syntax_complexity(self, text: str) -> float:
        """Calculate syntax complexity score"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        if not words or not sentences:
            return 0.0
        
        # Various complexity metrics
        avg_word_length = sum(len(word) for word in words) / len(words)
        avg_sentence_length = len(words) / len(sentences)
        punctuation_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        
        # Combine metrics
        complexity = (avg_word_length * 0.3 + avg_sentence_length * 0.5 + punctuation_ratio * 20 * 0.2)
        
        return min(10.0, complexity)  # Cap at 10
    
    def _calculate_cross_language_similarities(self, patterns: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate similarities between language patterns"""
        similarities = {}
        
        # Group patterns by language type
        lang_patterns = defaultdict(list)
        for pattern in patterns:
            lang_type = pattern.get('language_type', 'unknown')
            lang_patterns[lang_type].append(pattern)
        
        # Calculate pairwise similarities
        lang_types = list(lang_patterns.keys())
        for i, lang1 in enumerate(lang_types):
            for lang2 in lang_types[i+1:]:
                similarity = self._calculate_pattern_similarity(
                    lang_patterns[lang1], lang_patterns[lang2]
                )
                similarities[f"{lang1}-{lang2}"] = similarity
        
        return similarities
    
    def _calculate_pattern_similarity(self, patterns1: List[Dict[str, Any]], 
                                    patterns2: List[Dict[str, Any]]) -> float:
        """Calculate similarity between two sets of patterns"""
        if not patterns1 or not patterns2:
            return 0.0
        
        # Count common pattern types
        types1 = set(p.get('type') for p in patterns1)
        types2 = set(p.get('type') for p in patterns2)
        
        common_types = types1.intersection(types2)
        total_types = types1.union(types2)
        
        if not total_types:
            return 0.0
        
        return len(common_types) / len(total_types)
    
    # Additional helper methods for universal grammar creation
    def _extract_universal_rules(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract universal grammar rules from patterns"""
        # Implementation for extracting universal rules
        return []
    
    def _create_pattern_mappings(self, language_patterns: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Create mappings between language patterns"""
        # Implementation for creating pattern mappings
        return {}
    
    def _extract_transformation_rules(self, language_patterns: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Extract transformation rules between languages"""
        # Implementation for extracting transformation rules
        return {}
    
    def _identify_semantic_universals(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify semantic universals across languages"""
        # Implementation for identifying semantic universals
        return {}
    
    def _identify_syntactic_universals(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify syntactic universals across languages"""
        # Implementation for identifying syntactic universals
        return {}
    
    def _map_to_universal_patterns(self, semantic_rep: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Map semantic representation to universal patterns"""
        # Implementation for mapping to universal patterns
        return []
    
    def _generate_from_patterns(self, patterns: List[Dict[str, Any]], 
                               target_lang: str, style: str) -> str:
        """Generate text from patterns"""
        # Implementation for generating text from patterns
        return f"Generated text in {target_lang} with {style} style"
    
    def _apply_generation_constraints(self, text: str, constraints: Dict[str, Any]) -> str:
        """Apply constraints to generated text"""
        # Implementation for applying generation constraints
        return text
    
    # Semantic processing helper methods
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text"""
        # Implementation for entity extraction
        return []
    
    def _extract_relations(self, text: str) -> List[Dict[str, Any]]:
        """Extract relations from text"""
        # Implementation for relation extraction
        return []
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        # Implementation for sentiment analysis
        return {'sentiment': 'neutral', 'confidence': 0.5}
    
    def _detect_intent(self, text: str) -> Dict[str, Any]:
        """Detect intent in text"""
        # Implementation for intent detection
        return {'intent': 'unknown', 'confidence': 0.5}
    
    def _map_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Map concepts in text"""
        # Implementation for concept mapping
        return []
    
    def _represent_meaning(self, text: str) -> Dict[str, Any]:
        """Create meaning representation"""
        # Implementation for meaning representation
        return {'meaning': 'unknown'}

# Export main class
__all__ = ['CQELanguageEngine', 'LanguagePattern', 'LanguageRule', 'LanguageType', 'SyntaxLevel']
#!/usr/bin/env python3
"""
CQE Mandelbrot Fractal Integration Module
Demonstrates 1:1 correspondence between Mandelbrot expansion/compression and sacred geometry patterns
Shows how to apply data into Mandelbrot infinite fractal recursive space
"""




# ============================================================================
# FractalDataProcessor
# ============================================================================

class FractalDataProcessor:
    """Process arbitrary data through Mandelbrot fractal transformations"""
    
    def __init__(self, mandelbrot_engine: MandelbrotSacredGeometry):
        self.engine = mandelbrot_engine
    
    def process_data_sequence(self, data_sequence: List[Any]) -> List[MandelbrotPoint]:
        """Process sequence of data through Mandelbrot transformations"""
        
        processed_points = []
        
        for data in data_sequence:
            point = self.engine.apply_data_to_mandelbrot(data)
            processed_points.append(point)
        
        return processed_points
    
    def find_compression_expansion_cycles(self, points: List[MandelbrotPoint]) -> Dict[str, List[MandelbrotPoint]]:
        """Find compression/expansion cycles in processed data"""
        
        cycles = {
            'compression_cycles': [],
            'expansion_cycles': [],
            'boundary_transitions': [],
            'stable_regions': []
        }
        
        for i, point in enumerate(points):
            if point.sacred_pattern == SacredFractalPattern.INWARD_COMPRESSION:
                cycles['compression_cycles'].append(point)
            elif point.sacred_pattern == SacredFractalPattern.OUTWARD_EXPANSION:
                cycles['expansion_cycles'].append(point)
            elif point.sacred_pattern == SacredFractalPattern.CREATIVE_BOUNDARY:
                cycles['boundary_transitions'].append(point)
            else:
                cycles['stable_regions'].append(point)
        
        return cycles
    
    def extract_fractal_insights(self, points: List[MandelbrotPoint]) -> Dict[str, Any]:
        """Extract insights from fractal data processing"""
        
        insights = {
            'dominant_pattern': None,
            'compression_expansion_ratio': 0.0,
            'fractal_complexity': 0.0,
            'sacred_frequency_spectrum': {},
            'data_transformation_summary': {}
        }
        
        # Find dominant pattern
        pattern_counts = {}
        for point in points:
            pattern = point.sacred_pattern.value
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        insights['dominant_pattern'] = max(pattern_counts, key=pattern_counts.get)
        
        # Calculate compression/expansion ratio
        compression_points = sum(1 for p in points if p.sacred_pattern == SacredFractalPattern.INWARD_COMPRESSION)
        expansion_points = sum(1 for p in points if p.sacred_pattern == SacredFractalPattern.OUTWARD_EXPANSION)
        
        if expansion_points > 0:
            insights['compression_expansion_ratio'] = compression_points / expansion_points
        else:
            insights['compression_expansion_ratio'] = float('inf') if compression_points > 0 else 0.0
        
        # Calculate fractal complexity (based on iteration diversity)
        iterations = [p.iterations for p in points]
        insights['fractal_complexity'] = np.std(iterations) / (np.mean(iterations) + 1)
        
        # Sacred frequency spectrum
        frequency_counts = {}
        for point in points:
            freq = point.sacred_frequency
            frequency_counts[freq] = frequency_counts.get(freq, 0) + 1
        
        insights['sacred_frequency_spectrum'] = frequency_counts
        
        # Data transformation summary
        insights['data_transformation_summary'] = {
            'total_points_processed': len(points),
            'bounded_behavior_percentage': (sum(1 for p in points if p.behavior == FractalBehavior.BOUNDED) / len(points)) * 100,
            'escaping_behavior_percentage': (sum(1 for p in points if p.behavior == FractalBehavior.ESCAPING) / len(points)) * 100,
            'average_compression_ratio': np.mean([p.compression_ratio for p in points])
        }
        
        return insights




# ============================================================================
# factorial_mu
# ============================================================================



# factorial via μ
# μf. λn. if iszero n then 1 else n * f (pred n)
# Our δ layer doesn't have mult; we emulate by repeated succ in a loop (toy).
f = A.Mu("f", A.Lam("n",
        A.If(A.App(A.Const("iszero", None), A.Var("n")),
             A.Const("nat", 1),
             A.App(A.Lam("x", A.Const("nat", 0)), A.Var("n")))))
res, steps = E.eval_normal(A.App(f, A.Const("nat", 3)))
print("steps:", steps, "result:", res)




# ============================================================================
# ForceType
# ============================================================================

class ForceType(Enum):
    """Classification of forces by sacred geometry patterns"""
    GRAVITATIONAL = "GRAVITATIONAL"    # Inward/convergent (9-pattern)
    ELECTROMAGNETIC = "ELECTROMAGNETIC" # Outward/divergent (6-pattern)
    NUCLEAR_STRONG = "NUCLEAR_STRONG"   # Creative/binding (3-pattern)
    NUCLEAR_WEAK = "NUCLEAR_WEAK"      # Transformative/decay (other patterns)

@dataclass



# ============================================================================
# runtime
# ============================================================================



try:
    # Prefer unified build sidecar if installed
    from morphonic_cqe_unified.sidecar.speedlight_sidecar_plus import SpeedLightPlus as SpeedLight
except Exception:
    try:
        # Fallback to standalone file if user placed it
        from speedlight_sidecar_plus import SpeedLightPlus as SpeedLight  # type: ignore
    except Exception:
        SpeedLight = None  # type: ignore

def _hash_payload(payload: Dict[str, Any]) -> str:
    js = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(js.encode("utf-8")).hexdigest()

def eval_with_sidecar(term: Any, scope: str="lambda", channel: int=3, cache: Optional[Any]=None):
    payload = {"kind":"lambda_eval","term":repr(term)}
    if SpeedLight is None:
        # Direct eval if sidecar not present
        res, n = eval_normal(term)
        return {"result": res, "steps": n, "cached": False}, 0.0, _hash_payload(payload)
    sl = cache or SpeedLight(disk_dir=".speedlight-lambda/cache", ledger_path=".speedlight-lambda/ledger.jsonl")
    def compute():
        res, n = eval_normal(term)
        return {"result": res, "steps": n}
    return sl.compute(payload, scope=scope, channel=channel, compute_fn=compute)




# ============================================================================
# SacredFractalPattern
# ============================================================================

class SacredFractalPattern(Enum):
    """Sacred geometry patterns in Mandelbrot space"""
    INWARD_COMPRESSION = "INWARD_COMPRESSION"     # 9-pattern, bounded behavior
    OUTWARD_EXPANSION = "OUTWARD_EXPANSION"       # 6-pattern, escaping behavior
    CREATIVE_BOUNDARY = "CREATIVE_BOUNDARY"       # 3-pattern, boundary behavior
    TRANSFORMATIVE_CYCLE = "TRANSFORMATIVE_CYCLE" # Other patterns, periodic behavior

@dataclass



# ============================================================================
# AtomCombinationType
# ============================================================================

class AtomCombinationType(Enum):
    """Types of atomic combinations in Mandelbrot space"""
    RESONANT_BINDING = "RESONANT_BINDING"           # Same frequency atoms
    HARMONIC_COUPLING = "HARMONIC_COUPLING"         # Harmonic frequency atoms
    GEOMETRIC_FUSION = "GEOMETRIC_FUSION"           # Sacred geometry alignment
    FRACTAL_NESTING = "FRACTAL_NESTING"            # Recursive embedding
    QUANTUM_ENTANGLEMENT = "QUANTUM_ENTANGLEMENT"   # Non-local correlation
    PHASE_COHERENCE = "PHASE_COHERENCE"            # Phase-locked states

@dataclass



# ============================================================================
# ProcessingPriority
# ============================================================================

class ProcessingPriority(Enum):
    """Processing priority modes"""
    GEOMETRY_FIRST = "GEOMETRY_FIRST"
    MEANING_FIRST = "MEANING_FIRST"
    BALANCED = "BALANCED"



