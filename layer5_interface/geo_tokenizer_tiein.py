
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
"""

from __future__ import annotations
import os, io, sys, json, math, time, zlib, struct, hashlib, argparse, random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional, Callable

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
                f.write(json.dumps(asdict(le)) + "\n")

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
