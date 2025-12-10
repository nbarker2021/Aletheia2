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

This is not a text tokenizer. It's a geometry/memory manager that can mint/upgrade
tokens on demand and convert to known canonical tokens using past learned embeddings.
"""

import hashlib
import json
import math
import os
import struct
import time
import zlib
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

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
    """Receipt-first ledger for GeoTokenizer operations."""
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

# ───────────────────────────── GeoTokenizer ─────────────────────────────

class GeoTokenizer:
    def __init__(self, scale: float=1e-3, compressed: bool=True, memory_path: str=".geo_tokenizer/memory.json",
                 ledger_path: Optional[str]=".geo_tokenizer/ledger.jsonl"):
        self.codec = GeoCodec(scale, compressed)
        self.memory = TokenMemory(memory_path)
        self.ledger = TokLight(ledger_path)

    def encode(self, toks: List[GeoToken]) -> bytes:
        raw_inp = json.dumps([{"pos": t.pos, "feat": t.feat, "tag": t.tag} for t in toks]).encode("utf-8")
        t0 = time.time()
        b = self.codec.encode(toks)
        self.ledger.log("tokenizer", "encode", raw_inp, b, time.time()-t0)
        return b

    def decode(self, b: bytes) -> List[GeoToken]:
        t0 = time.time()
        toks = self.codec.decode(b)
        raw_out = json.dumps([{"pos": t.pos, "feat": t.feat, "tag": t.tag} for t in toks]).encode("utf-8")
        self.ledger.log("tokenizer", "decode", b, raw_out, time.time()-t0)
        return toks

    def learn(self, name: str, toks: List[GeoToken], meta: Optional[Dict[str,Any]]=None):
        self.memory.learn(name, toks, meta)

    def nearest(self, toks: List[GeoToken]) -> Tuple[Optional[str], float]:
        return self.memory.nearest(toks)
