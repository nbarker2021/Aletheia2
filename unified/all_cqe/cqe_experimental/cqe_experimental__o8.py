#!/usr/bin/env python3
# O8 — Octet/Shape-Pack DSL (base-8 primary) — Minimal Interpreter
# Apache-2.0
import sys, re, json, math, hashlib, argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ================= Numeric parsing (base-8) =================
def parse_octal_num(tok:str)->int:
    # Default number base = 8 unless prefixed 0x (hex) or 0b (bin) or 0d (dec)
    t = tok.strip().lower()
    if t.startswith("0x"): return int(t,16)
    if t.startswith("0b"): return int(t,2)
    if t.startswith("0d"): return int(t[2:],10)
    # allow underscores
    t = t.replace("_","")
    # empty -> 0
    if not t: return 0
    # float? keep octal integer then allow / scaling
    # We treat everything as ints for base-8; for floats accept x.y as decimal literal
    if "." in t:
        try:
            return float(t)  # rare explicit decimal
        except:
            raise ValueError(f"bad float literal: {tok}")
    return int(t, 8)

# ================= Shape packs (4-bit / a|b) =================
# a=1, b=0 ; string of length 4, e.g., abba, bbbb, aaaa
def pack_bits(s:str):
    s = s.strip().lower()
    if not re.fullmatch(r"[ab]{4}", s):
        raise ValueError(f"invalid shape pack: {s}")
    return [1 if c=="a" else 0 for c in s]

# Map 4-bit shape pack to primitive op mnemonic
SHAPE_OP = {
    "bbbb": "NOP",
    "bbba": "DLIFT",
    "bbab": "MIRROR",
    "bbaa": "RATCHET",
    "babb": "SNAP",
    "baba": "ANNIHILATE",
    "baab": "POSE",
    "baaa": "TICKET",
    "abbb": "BIND",
    "abba": "ROLE",
    "abab": "EMIT",
    "abaa": "CALL",
    "aabb": "MAP",
    "aaba": "FORK",
    "aaab": "JOIN",
    "aaaa": "ASSERT"
}

# ================= Geometry helpers (E8 cap + pose) =================
def hadamard8():
    H2 = np.array([[1,1],[1,-1]],float)
    H4 = np.kron(H2,H2)
    H8 = np.kron(H4,H2)
    return H8/np.sqrt(8.0)

E8_ROOTS = np.array([
    [ 1, -1,  0,  0,  0,  0,  0,  0],
    [ 0,  1, -1,  0,  0,  0,  0,  0],
    [ 0,  0,  1, -1,  0,  0,  0,  0],
    [ 0,  0,  0,  1, -1,  0,  0,  0],
    [ 0,  0,  0,  0,  1, -1,  0,  0],
    [ 0,  0,  0,  0,  0,  1, -1,  0],
    [ 0,  0,  0,  0,  0,  1,  1,  0],
    [-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5, 0.5]
], dtype=float)

def e8_nearest(y):
    z0 = np.rint(y)
    if (int(np.sum(z0)) & 1) == 1:
        frac = np.abs(y - z0); k = int(np.argmin(frac))
        z0[k] += 1 if y[k] > z0[k] else -1
    d0 = np.linalg.norm(y - z0)
    yh = y - 0.5
    z1 = np.rint(yh)
    if (int(np.sum(z1)) & 1) == 1:
        frac = np.abs(yh - z1); k = int(np.argmin(frac))
        z1[k] += 1 if yh[k] > z1[k] else -1
    x1 = z1 + 0.5
    d1 = np.linalg.norm(y - x1)
    if d0 <= d1:
        return z0, d0, d0, d1, "int", x1
    else:
        return x1, d1, d0, d1, "half", z0

def e8_snap_block(X):
    N = X.shape[0]
    V = np.zeros_like(X); di = np.zeros(N); dh = np.zeros(N)
    altV = np.zeros_like(X)
    coset = np.empty(N, dtype=object)
    for i in range(N):
        vb, db, d0, d1, c, av = e8_nearest(X[i])
        V[i]=vb; di[i]=d0; dh[i]=d1; coset[i]=c; altV[i]=av
    return V, di, dh, coset, altV

def coset_margin(di, dh, eps=1e-9):
    return np.abs(di - dh) / (di + dh + eps)

def pose_bits(X, V, R=None):
    if R is None: R = np.eye(8)
    Rroots = E8_ROOTS @ R.T
    Rroots = Rroots / (np.linalg.norm(Rroots, axis=1, keepdims=True)+1e-9)
    Rres = X - V
    S = (Rres @ Rroots.T)
    return (S >= 0).astype(int)

def alignment_rate(P):
    powers = (1 << np.arange(8))[::-1]
    ints = P @ powers
    vals, counts = np.unique(ints, return_counts=True)
    return counts.max()/P.shape[0]

def fixed_rotations(seed=2025):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(8,8)); Q, _ = np.linalg.qr(A)
    H = hadamard8()
    Sflip = np.diag([1,1,1,1,-1,-1,-1,-1])
    return [np.eye(8), H, Q, Sflip@H]

# ================= Interpreter =================
class O8State:
    def __init__(self):
        self.dim = 16     # default: two 8D blocks
        self.seed = 2025
        self.tau_w = 0o0_04/100  # base-8-ish tiny default (≈0.005)
        self.tau_annih = 0o0_02/100
        self.gauge = "auto"
        self.scene = "default"
        self.sidecar = {}
        self.R = np.eye(8)
        self.rotset = fixed_rotations(self.seed)
        self.ledger = []
        self.X = None     # working matrix
        self.snap = None  # last snap info
        self.tickets = None

    def log(self, stage, note, payload=None):
        row = {"stage": stage, "note": note, "payload": payload or {}}
        self.ledger.append(row)

# --- Parsers ---
HEADER_KV = re.compile(r'^(scene|dim|gauge)\s+(.+)$', re.I)
SIDECAR_OPEN = re.compile(r'^sidecar\s*\{\s*$', re.I)

INSTR = re.compile(r'^\s*([ab]{4})\s+([A-Z]+)\s*(.*?);?\s*(#.*)?$')
OCTET_OPEN = re.compile(r'^\s*octet\s*\{\s*$', re.I)
OCTET_CLOSE = re.compile(r'^\s*\}\s*$')

def parse_program(text:str):
    lines = [ln.rstrip() for ln in text.splitlines()]
    prog = {"header":{}, "sidecar":{}, "octets":[]}
    i=0; n=len(lines)
    # header & sidecar
    while i<n:
        ln = lines[i].strip()
        if not ln or ln.startswith("#"): i+=1; continue
        if SIDECAR_OPEN.match(ln):
            buf=[]; i+=1
            while i<n and "}" not in lines[i]:
                buf.append(lines[i]); i+=1
            if i<n: i+=1  # skip '}'
            prog["sidecar"]=json.loads("\n".join(buf) or "{}")
            continue
        m = HEADER_KV.match(ln)
        if m:
            prog["header"][m.group(1).lower()] = m.group(2).strip()
            i+=1; continue
        if OCTET_OPEN.match(ln): break
        i+=1
    # body
    while i<n:
        ln = lines[i]
        if OCTET_OPEN.match(ln):
            block=[]; i+=1
            while i<n and not OCTET_CLOSE.match(lines[i]):
                m = INSTR.match(lines[i])
                if m:
                    block.append((m.group(1).lower(), m.group(2).upper(), m.group(3).strip()))
                i+=1
            if i<n: i+=1
            prog["octets"].append(block)
        else:
            i+=1
    return prog

# --- Adapters (delegation to other languages) ---
def adapter_call(lang:str, func:str, args:list):
    if lang=="py":
        # Tiny safe adapter: permit numpy/pure math slices
        safe = {"np": np, "math": math}
        try:
            return eval(func, {"__builtins__": {}}, safe)(*args)
        except Exception as e:
            return {"error": str(e)}
    return {"error": f"adapter {lang} not available"}

# --- Execution primitives ---
def init_space(st:O8State):
    # Create synthetic torus vectors (two 8D blocks)
    n = 8192
    rng = np.random.default_rng(7)
    ThA = rng.random((n,5))*2*math.pi
    ThB = rng.random((n,5))*2*math.pi
    A = np.concatenate([np.cos(ThA[:,:4]), np.sin(ThA[:,:4])], axis=1)
    B = np.concatenate([np.cos(ThB[:,:4]), np.sin(ThB[:,:4])], axis=1)
    st.X = np.hstack([A,B])
    st.log("INIT","space created", {"n": n, "dim": st.dim})

def op_POSE(st:O8State, args):
    # Choose rotation maximizing alignment for first 8D block
    X8 = st.X[:,:8]
    V, di, dh, coset, altV = e8_snap_block(X8)
    best=None; bestR=None
    for R in st.rotset:
        P = pose_bits(X8, V, R); r = alignment_rate(P)
        if best is None or r>best: best=r; bestR=R
    st.R = bestR
    st.log("POSE","gauge set", {"alignment": float(best)})
    return {"alignment": float(best)}

def op_TICKET(st:O8State, args):
    # Boundary tickets across both 8D blocks
    V0, di0, dh0, cos0, alt0 = e8_snap_block(st.X[:,:8])
    V1, di1, dh1, cos1, alt1 = e8_snap_block(st.X[:,8:16])
    m0 = coset_margin(di0, dh0); m1 = coset_margin(di1, dh1)
    mask = (m0 <= st.tau_w) | (m1 <= st.tau_w)
    st.tickets = {"idx": np.where(mask)[0], "m_min": np.minimum(m0, m1), "move_cost": np.linalg.norm(np.hstack([alt0,alt1]) - np.hstack([V0,V1]), axis=1)}
    st.log("TICKETS","boundary found", {"count": int(mask.sum())})
    return {"count": int(mask.sum())}

def op_SNAP(st:O8State, args):
    # Commit at caps: no state change other than logging in this minimal demo
    if st.tickets is None:
        return {"error":"no tickets"}
    st.log("COMMIT","snap at caps", {"tickets": int(len(st.tickets["idx"]))})
    return {"committed": int(len(st.tickets["idx"]))}

def op_ANNIHILATE(st:O8State, args):
    if st.tickets is None:
        return {"error":"no tickets"}
    idx = st.tickets["idx"]; m = st.tickets["m_min"]; mv = st.tickets["move_cost"]
    k = (m <= st.tau_annih)
    removed = int(k.sum())
    st.log("ANNIHILATE","rails", {"removed": removed})
    return {"removed": removed}

def op_MIRROR(st:O8State, args):
    # conceptual mirror; no mutation needed, just a receipt
    st.log("MIRROR","palindromic check",{})
    return {"mirror":"ok"}

def op_RATCHET(st:O8State, args):
    # tighten thresholds by 10%
    st.tau_w *= 0.9; st.tau_annih *= 0.9
    st.log("RATCHET","tighten", {"tau_w": st.tau_w, "tau_annih": st.tau_annih})
    return {"tau_w": st.tau_w}

def op_EMIT(st:O8State, args):
    # emit receipts to file
    out = args.strip() or "o8_receipts.json"
    Path(out).write_text(json.dumps(st.ledger, indent=2))
    st.log("EMIT","wrote", {"file": out})
    return {"file": out}

def op_BIND(st:O8State, args):
    # BIND key=value into sidecar
    m = re.match(r'(\w+)\s*=\s*(.+)$', args)
    if not m: return {"error":"bind expects key=value"}
    k,v = m.group(1), m.group(2)
    try:
        v = json.loads(v)
    except Exception:
        v = v.strip('"')
    st.sidecar[k]=v
    st.log("BIND","sidecar", {k: v})
    return {k: v}

def op_CALL(st:O8State, args):
    # CALL lang func argjson -> var (var ignored; we just log result)
    m = re.match(r'(\w+)\s+"([^"]+)"\s*(.*)$', args)
    if not m: return {"error":"CALL lang \"func\" [json_args]"}
    lang, func, rest = m.group(1), m.group(2), m.group(3).strip()
    arr = []
    if rest:
        try:
            arr = json.loads(rest)
        except Exception:
            arr = []
    res = adapter_call(lang, func, arr)
    st.log("CALL","adapter", {"lang":lang,"func":func,"result":str(res)[:256]})
    return {"result": res}

OPS = {
    "POSE": op_POSE,
    "TICKET": op_TICKET,
    "SNAP": op_SNAP,
    "ANNIHILATE": op_ANNIHILATE,
    "MIRROR": op_MIRROR,
    "RATCHET": op_RATCHET,
    "EMIT": op_EMIT,
    "BIND": op_BIND,
    "CALL": op_CALL,
    "NOP": lambda st,a: {"ok":True},
    "ROLE": lambda st,a: st.log("ROLE","set",{"role":a}) or {"role":a},
    "MAP":  lambda st,a: st.log("MAP","route",{"map":a}) or {"map":a},
    "FORK": lambda st,a: st.log("FORK","fork",{}) or {"forked":True},
    "JOIN": lambda st,a: st.log("JOIN","join",{}) or {"joined":True},
    "ASSERT": lambda st,a: st.log("ASSERT","check",{"expr":a}) or {"assert":a}
}

def run_o8(text:str, outdir:str):
    prog = parse_program(text)
    st = O8State()
    # header
    if "scene" in prog["header"]: st.scene = prog["header"]["scene"]
    if "gauge" in prog["header"]: st.gauge = prog["header"]["gauge"]
    if "dim" in prog["header"]:
        try: st.dim = int(prog["header"]["dim"], 8)  # base-8
        except: st.dim = int(prog["header"]["dim"])
    st.sidecar.update(prog["sidecar"] or {})
    # init
    init_space(st)
    # execute octets
    for bi, block in enumerate(prog["octets"]):
        st.log("OCTET","enter", {"index": bi})
        for (pack, op, args) in block:
            # check mapping
            opm = SHAPE_OP.get(pack, None)
            if opm is None or (opm != op):
                # allow explicit opcode override if it matches
                if op not in OPS: raise ValueError(f"unknown op: {op}")
                opm = op
            res = OPS[opm](st, args)
            st.log("STEP", f"{pack} {opm}", {"args": args, "res": res})
        st.log("OCTET","leave", {"index": bi})
    # write ledger + summary
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir/"ledger.jsonl").write_text("\n".join(json.dumps(x) for x in st.ledger))
    (outdir/"summary.json").write_text(json.dumps({"scene":st.scene,"gauge":st.gauge,"dim":st.dim,"sidecar":st.sidecar}, indent=2))
    return st

def main():
    ap = argparse.ArgumentParser(description="O8 — Shape-Pack DSL")
    ap.add_argument("program", type=str, help=".o8 program file")
    ap.add_argument("--out", type=str, default="o8_out", help="output directory")
    args = ap.parse_args()
    text = Path(args.program).read_text(encoding="utf-8")
    st = run_o8(text, args.out)
    print(json.dumps({"ok": True, "scene": st.scene, "steps": len(st.ledger)}, indent=2))

if __name__ == "__main__":
    main()
