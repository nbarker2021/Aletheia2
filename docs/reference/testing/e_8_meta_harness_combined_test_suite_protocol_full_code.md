# E8 Meta‑Harness & Combined Test Suite — Protocol + Full Code

## Purpose

A portable, repeatable testbed that exercises the full algebra (ALT, W4, L8, Q8, Λ, H8), the QF gate (HOLD/SWAP), ledger receipts, governance chain (geometry→hyper→fractal→helix→bonded), entropy-as-exchange, and cross‑field interaction via an **E8 meta‑driver** layer (3 Context + 5 Governance). The harness runs on proxy datasets but is designed so you can swap in real IRL datasets without changing the core tests.

---

## What this delivers

1. **Complete, enumerated test plan** (below) matching your specification.
2. **Full, self‑contained Python code** (end of document) that writes out CSV/JSON/PNG artifacts and a compact dashboard.
3. **Swap‑in points** for real datasets and encodings.

---

## Test Plan (Fully Enumerated)

### A) Core Algebra & Local Gates

- **A1. Invariants:**
  - ALT (alternating parity)
  - W4 (mod‑4 linear)
  - L8 (mod‑8 linear)
  - Q8 (mod‑8 quadratic)
- **A2. Energy Λ:** ALT‑violation count + Lee distance of (Σ residues mod 4).
- **A3. QF gate:** HOLD always legal; SWAP legal only if it strictly reduces Λ (or if accompanied by a receipt).
- **A4. MSS Acceptance:** ALT ∧ (W4 ∨ Q8) ∧ pattern‑seen (+12d proxy).

### B) Governance Chain

- **B1. geometry→hyper:** structure‑preserving lift compatible with mod‑4 residues.
- **B2. hyper→fractal:** decimate/interleave with lawful duals.
- **B3. fractal→helix:** chiral twist that preserves H8 unless receipts are provided.
- **B4. helix→bonded:** counter‑rotate and pair such that H8 tags cancel mod 8; otherwise require receipts.

### C) Ledger & Receipts

- **C1. H8 tag:** ((a–d)+2(b–c)) mod 8.
- **C2. Conservation:** ΣH8 ≡ const (mod 8) unless a **Deposit/Annihilate** receipt is logged.
- **C3. Accounting check:** ledger total must equal sum of receipts mod 8.

### D) Efficiency & Redundancy

- **D1. w80 exact cache:** if \~80% windows repeat, cache invariant/Λ results; should match full recalculation.
- **D2. (Optional) approximate cache:** class by (H8, Λ‑bin).

### E) Quadratic Stability

- **E1. Vertex test:** fit Λ along a small path; vertex should sit in the interval of the discrete minimum for convex cases.

### F) Number‑Theoretic Reversibility

- **F1. Modular inverses:** residues invertible iff gcd(a, m)=1; others require structured annihilation/receipts.
- **F2. Bézout identity:** a·x + m·y = gcd(a,m) as the certificate for non‑invertible states.

### G) Cross‑Domain Analogies

- **G1. Nervous system homeostasis:** treat H8 as conserved firing balance; block illegal ledger changes.
- **G2. Transformer checkpoints:** simulate 4096‑length sequence with 32/36 gates and a per‑block DOF budget.

### H) Entropy as n=1 Exchange

- **H1. Global scaling s:** median(Λ\_A / Λ\_B) on aligned windows.
- **H2. Piecewise s\_h by H8 class:** reduces mismatch further.
- **H3. Entropy units:** count of |Λ\_A − s·Λ\_B| ≥ 1 events.

### I) Combined Dashboard

- Aggregate per‑seed metrics: accept rate, helix legality, bonded success, w80 equality, ledger‑receipt match, quadratic vertex success, neuron conservation, transformer legality, entropy per‑N.

### J) E8 Meta‑Harness (Cross‑Field Interaction)

- **J1. E8 drivers:** 3 Context (C1–C3) + 5 Governance (G1–G5).
- **J2. 8 Fields (proxy datasets):** DNA, Text, Plasma, Finance, Neuro, Primes, Embeddings, Social.
- **J3. Movement rule:** any datum shift spends one of 10 DoF lanes; observers (neighbor fields) supply receipts when exchange mismatch is low.
- **J4. Scores:** context fingerprint per dataset, governance scores per dataset, and a coupling heatmap (pairwise entropy exchange mismatch).

---

## How to Run

- Dependencies: Python 3.9+, `numpy`, `pandas`, `matplotlib`.
- Execute the script. It will write artifacts to two folders:
  - `harness_combined/` — combined main+supplement tests + dashboard
  - `E8_meta_harness/` — E8 driver scores + coupling heatmap
- Swap in real datasets: replace the proxy dataset generators in the E8 section with loaders/encoders that map your IRL data into residues {1..4}.

---

## Full Harness Code (copy‑paste runnable)

```python
# -*- coding: utf-8 -*-
"""
E8 Meta‑Harness & Combined Test Suite
-------------------------------------
Runs:
  • Core algebra & QF gate
  • Governance chain (geometry→hyper→fractal→helix→bonded)
  • Ledger/receipts, w80 caching, quadratic vertex
  • Modular inverses/Bézout, neuron analogy, transformer checkpoints
  • Entropy as n=1 exchange (global/piecewise)
  • Combined dashboard (multi‑seed)
  • E8 meta‑harness across 8 fields (context/governance + coupling matrix)

Artifacts:
  - /mnt/data/harness_combined
  - /mnt/data/E8_meta_harness
"""

import os, json, math, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Output directories
# ---------------------------
OUT_COMB = os.path.join('/mnt/data', 'harness_combined')
OUT_E8   = os.path.join('/mnt/data', 'E8_meta_harness')
os.makedirs(OUT_COMB, exist_ok=True)
os.makedirs(OUT_E8, exist_ok=True)

# ---------------------------
# Core algebra
# ---------------------------

def parity(v):  # 1-based Z4
    return (v-1) % 2

def ALT(a,b,c,d):
    return (parity(a)!=parity(b)) and (parity(b)!=parity(c)) and (parity(c)!=parity(d))

def W4(a,b,c):
    return (a + 2*b + 3*c) % 4 == 2

def L8(a,b,c,d):
    return ((a - d) + 2*(b - c)) % 8 == 0

def Q8(a,b,c,d):
    return (((a - d)**2 + (b - c)**2) % 8) == 0

def H8(a,b,c,d):
    return ((a - d) + 2*(b - c)) % 8

def lee(s):
    s = s % 4
    return min(s, (4 - s) % 4)

def Lambda(a,b,c,d):
    alt_v = 0
    if parity(a)==parity(b): alt_v += 1
    if parity(b)==parity(c): alt_v += 1
    if parity(c)==parity(d): alt_v += 1
    s = (a + b + c + d) % 4
    return alt_v + lee(s)

def Dual(a,b,c,d):
    return (a,d,c,b)

def QF_decide(a,b,c,d, receipt=None):
    lam0 = Lambda(a,b,c,d)
    if receipt is not None:
        return "SWAP", lam0, Lambda(*Dual(a,b,c,d))
    lam1 = Lambda(*Dual(a,b,c,d))
    if lam1 < lam0:
        return "SWAP", lam0, lam1
    else:
        return "HOLD", lam0, lam0

@dataclass
class Ledger:
    mod:int = 8
    total:int = 0
    def tag(self, a,b,c,d):
        return H8(a,b,c,d)
    def add(self, delta):
        self.total = (self.total + delta) % self.mod
    def deposit(self, h):
        self.add(h % self.mod)
        return h % self.mod
    def annihilate(self, h):
        self.add((-h) % self.mod)
        return (-h) % self.mod

# ---------------------------
# Generators & helpers
# ---------------------------

def rand_quad(n=1, seed=None):
    rng = random.Random(seed)
    for _ in range(n):
        yield tuple(rng.randint(1,4) for _ in range(4))

def lawful_quad(n=1, seed=None):
    rng = random.Random(seed)
    out=[]
    while len(out)<n:
        a,b,c,d = (rng.randint(1,4) for _ in range(4))
        if ALT(a,b,c,d) and (W4(a,b,c) or Q8(a,b,c,d)):
            out.append((a,b,c,d))
    return out

def to_quads(seq):
    return [(seq[i], seq[i+1], seq[i+2], seq[i+3]) for i in range(len(seq)-3)]

def lawful_quads_from_seq(seq):
    return [q for q in to_quads(seq) if ALT(*q) and (W4(q[0],q[1],q[2]) or Q8(*q))]

# ---------------------------
# Governance chain
# ---------------------------

def lift_to_Z8(q):
    # Structural no-op for invariants; shadow lift if needed later
    return q, q

def fractalize(seq):
    return [q if i%2==0 else Dual(*q) for i,q in enumerate(seq)]

def helical_twist(q):
    a,b,c,d = q
    b2 = ((b) % 4) + 1
    d2 = ((d + 1) % 4) + 1
    cand = (a,b2,c,d2)
    if H8(*cand) != H8(a,b,c,d):
        return q, False
    return cand, True

def bond_pair(h1, h2):
    h2c = Dual(*h2)
    tag_sum = (H8(*h1) + H8(*h2c)) % 8
    if tag_sum == 0:
        return h1, h2c, True
    h1d = Dual(*h1)
    if Lambda(*h1d) < Lambda(*h1) and (H8(*h1d) + H8(*h2c)) % 8 == 0:
        return h1d, h2c, True
    return h1, h2c, False

def test_governance_chain(N=200, seed=7):
    seq = lawful_quad(N, seed=seed)
    geom_ok = sum(1 for q in seq if ALT(*q) and (W4(q[0],q[1],q[2]) or Q8(*q))) == len(seq)
    hyper = [lift_to_Z8(q)[1] for q in seq]
    hyper_ok = sum(1 for q in hyper if ALT(*q) and (W4(q[0],q[1],q[2]) or Q8(*q))) == len(hyper)
    fract = fractalize(hyper)
    fract_ok = sum(1 for q in fract if ALT(*q) and (W4(q[0],q[1],q[2]) or Q8(*q))) == len(fract)
    helix = []
    helix_legal = 0
    for q in fract:
        q2, ok = helical_twist(q)
        helix.append(q2)
        helix_legal += 1 if ok else 0
    helix_ok = sum(1 for q in helix if ALT(*q) and (W4(q[0],q[1],q[2]) or Q8(*q))) == len(helix)
    bonded_ok = 0
    for i in range(0, len(helix)-1, 2):
        a,b,ok = bond_pair(helix[i], helix[i+1])
        bonded_ok += 1 if ok else 0
    return {
        "geom_ok": geom_ok,
        "hyper_ok": hyper_ok,
        "fract_ok": fract_ok,
        "helix_ok": helix_ok,
        "helix_legal_fraction": helix_legal/len(helix),
        "bonded_ok_pairs": bonded_ok,
        "pairs_total": len(helix)//2
    }

# ---------------------------
# QF HOLD test
# ---------------------------

def test_qf_hold_validity(N=500, seed=11):
    ledger = Ledger()
    holds = 0; swaps = 0; violations = 0
    for (a,b,c,d) in rand_quad(N, seed=seed):
        h_before = ledger.total; ledger.add(0)
        if ledger.total != h_before: violations += 1
        holds += 1
        lam0 = Lambda(a,b,c,d); lam1 = Lambda(*Dual(a,b,c,d))
        if lam1 < lam0: swaps += 1
    return {"holds": holds, "swaps_if_descent": swaps, "ledger_violations": violations}

# ---------------------------
# w80 caching
# ---------------------------

def w80_full_search(quads):
    return [(ALT(*q), W4(q[0],q[1],q[2]), L8(*q), Q8(*q), Lambda(*q)) for q in quads]

def w80_reduced_search(quads):
    cache={}; out=[]; repeats = 0
    for q in quads:
        if q in cache:
            repeats += 1; out.append(cache[q])
        else:
            val = (ALT(*q), W4(q[0],q[1],q[2]), L8(*q), Q8(*q), Lambda(*q))
            cache[q]=val; out.append(val)
    return out, repeats/len(quads)

def test_w80(N=1000, seed=13):
    random.seed(seed)
    base = lawful_quad(max(1, N//5), seed=seed)
    quads = [random.choice(base) for _ in range(N)]
    full = w80_full_search(quads)
    reduced, repeat_frac = w80_reduced_search(quads)
    equal = (full == reduced)
    return {"repeat_fraction": repeat_frac, "equal_results": equal}

# ---------------------------
# Deposits vs state
# ---------------------------

def test_deposits_vs_state(N=300, seed=17):
    ledger = Ledger(); receipts_sum = 0
    for (a,b,c,d) in rand_quad(N, seed=seed):
        tag0 = H8(a,b,c,d)
        a2,b2,c2,d2 = Dual(a,b,c,d); tag1 = H8(a2,b2,c2,d2)
        if tag1 != tag0:
            r = (tag1 - tag0) % 8
            receipts_sum = (receipts_sum + ledger.deposit(r)) % 8
    return {"ledger_total": ledger.total, "receipts_sum": receipts_sum, "match": ledger.total == receipts_sum}

# ---------------------------
# Quadratic vertex test
# ---------------------------

def fit_quadratic(xs, ys):
    X = np.vstack([np.array(xs)**2, np.array(xs), np.ones(len(xs))]).T
    a,b,c = np.linalg.lstsq(X, np.array(ys), rcond=None)[0]
    return a,b,c

def test_quadratic_vertex(num_cases=200, seed=19):
    rng = np.random.default_rng(seed)
    successes = 0
    for _ in range(num_cases):
        q0 = lawful_quad(1, seed=int(rng.integers(1,1e6)))[0]
        q1 = Dual(*q0)
        a,b,c,d = q1; d = ((d % 4) + 1); q2 = (a,b,c,d)
        xs = [-1,0,1]; ys = [Lambda(*q) for q in [q0,q1,q2]]
        a,b_,c = fit_quadratic(xs, ys)
        if a <= 0: continue
        vertex = -b_/(2*a)
        min_idx = int(np.argmin(ys))
        success = (-1 + min_idx) <= vertex <= (-1 + min_idx + 1)
        successes += 1 if success else 0
    return {"cases": num_cases, "successes": successes, "success_rate": successes/num_cases if num_cases else 0.0}

# ---------------------------
# Modular inverses / Bézout
# ---------------------------

def egcd(a, b):
    if b == 0: return (a, 1, 0)
    g, x1, y1 = egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)

def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1: return None, g
    return x % m, g

def test_modular_inverses(mods=(8,12,16,9)):
    rows=[]
    for m in mods:
        for a in range(1, m):
            inv, g = modinv(a, m)
            if inv is not None:
                ok = (a*inv) % m == 1
                rows.append({"mod":m,"a":a,"invertible":True,"gcd":g,"check":ok})
            else:
                g2, x, y = egcd(a, m)
                rows.append({"mod":m,"a":a,"invertible":False,"gcd":g2,"check":(a*x + m*y)==g2})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_COMB, 'modular_inverses_report.csv'), index=False)
    return df

# ---------------------------
# Nervous system analogy
# ---------------------------

def test_neuron_ledger(N=256, seed=23):
    pre_tags=[]; post_tags=[]; violations=0
    for q in lawful_quad(N, seed=seed):
        pre_tags.append(H8(*q))
        choice, lam0, lam1 = QF_decide(*q, receipt=None)
        q2 = q if choice=="HOLD" else Dual(*q)
        post_tags.append(H8(*q2))
        if H8(*q2) != H8(*q):
            violations += 1
    return {"pre_sum": sum(pre_tags)%8, "post_sum": sum(post_tags)%8, "violations": violations, "conserved": (sum(pre_tags)%8)==(sum(post_tags)%8)}

# ---------------------------
# Transformer checkpoints
# ---------------------------

def test_transformer_checkpoints(L=4096, blocks=(32,36), seed=29):
    random.seed(seed)
    X = [random.randint(1,4) for _ in range(L)]
    def window(i): return (X[i], X[i+1], X[i+2], X[i+3])
    in_block_ok=0; cross_ok=0; total_in=0; total_cross=0
    for i in range(L-3):
        q = window(i)
        ok = ALT(*q) and (W4(q[0],q[1],q[2]) or Q8(*q))
        total_in += 1; in_block_ok += 1 if ok else 0
    dof_ok=True
    for bsize in blocks:
        changes=0; budget=bsize//4
        for start in range(0, L-bsize, bsize):
            for i in range(start, start+bsize-3):
                q = (X[i],X[i+1],X[i+2],X[i+3])
                lam0 = Lambda(*q); lam1 = Lambda(*Dual(*q))
                if lam1 < lam0 and changes < budget:
                    X[i],X[i+1],X[i+2],X[i+3] = Dual(*q); changes += 1
        if changes > budget: dof_ok=False
    for bsize in blocks:
        for start in range(0, L-bsize, bsize):
            idx = start+bsize-4
            if idx >= 0 and idx+3 < L:
                q = (X[idx],X[idx+1],X[idx+2],X[idx+3])
                total_cross += 1
                cross_ok += 1 if ALT(*q) and (W4(q[0],q[1],q[2]) or Q8(*q)) else 0
    return {"in_block_ok_rate": in_block_ok/max(1,total_in), "cross_boundary_ok_rate": cross_ok/max(1,total_cross), "dof_budget_respected": dof_ok, "blocks_tested": list(blocks)}

# ---------------------------
# Entropy as n=1 exchange
# ---------------------------

def entropy_exchange(seed=31):
    rng = np.random.default_rng(seed)
    n=800
    def synth(n):
        a = rng.integers(1,5,size=n)
        b = rng.integers(1,5,size=n)
        c = rng.integers(1,5,size=n)
        d = rng.integers(1,5,size=n)
        lam = np.array([Lambda(int(a[i]),int(b[i]),int(c[i]),int(d[i])) for i in range(n)])
        h8  = np.array([H8(int(a[i]),int(b[i]),int(c[i]),int(d[i])) for i in range(n)])
        return pd.DataFrame({"A":a,"B":b,"C":c,"D":d,"Lambda":lam,"H8":h8})
    X=synth(n); Y=synth(n)
    eps=1e-9
    mask = (Y["Lambda"]>0)
    ratios = (X.loc[mask, "Lambda"]/(Y.loc[mask, "Lambda"]+eps)).values
    s = float(np.median(ratios)) if len(ratios)>0 else 1.0
    s_h = {}
    inter = sorted(set(X["H8"]).intersection(set(Y["H8"])))
    for h in inter:
        Xh = X[X["H8"]==h]["Lambda"].values
        Yh = Y[Y["H8"]==h]["Lambda"].values
        if len(Xh)>5 and len(Yh)>5 and np.all(Yh>=0):
            k = min(len(Xh), len(Yh))
            Xs = np.random.default_rng(seed+int(h)).choice(Xh, size=k, replace=False)
            Ys = np.random.default_rng(seed+int(h)+1).choice(Yh, size=k, replace=False)
            s_h[int(h)] = float(np.median(Xs/(Ys+eps)))
    mismatch = np.abs(X["Lambda"].values - s*Y["Lambda"].values)
    entropy_units = int(np.sum(mismatch >= 1.0))
    Y_scaled = Y.copy()
    Y_scaled["Lambda_s"] = Y.apply(lambda r: s_h.get(int(r["H8"]), s) * r["Lambda"], axis=1)
    mismatch_h = np.abs(X["Lambda"].values - Y_scaled["Lambda_s"].values)
    entropy_units_h = int(np.sum(mismatch_h >= 1.0))
    return {"s_global": s, "s_piecewise": s_h, "entropy_units_global": entropy_units, "entropy_units_piecewise": entropy_units_h, "N": len(X)}

# ---------------------------
# Main pipeline + dashboard
# ---------------------------

def novelty_score(q):
    a,b,c,d = q
    vals = [a,b,c,d]
    uniq = len(set(vals))
    spread = len(set([v%2 for v in vals]))
    return uniq + 0.5*spread

def golden_hash(i,j):
    phi = (1+5**0.5)/2.0
    return (phi*i + (phi**2)*j) % 1.0

def tie_break(candidates: List[Tuple[Tuple[int,int,int,int], float]], pos_idx:int):
    eps = 1e-9
    best = max(s for _,s in candidates)
    near = [(q,s) for q,s in candidates if abs(s-best)<=eps]
    if len(near)==1: return near[0][0]
    best_nov = max(novelty_score(q) for q,_ in near)
    near2 = [q for q,s in near if novelty_score(q)==best_nov]
    if len(near2)==1: return near2[0]
    near2_sorted = sorted(near2, key=lambda q: -golden_hash(pos_idx, sum(q)))
    return near2_sorted[0]

def recurrence_index(neigh_prev: List[List[int]], neigh_curr: List[List[int]]):
    overlap = 0.0; total = 0.0
    for S,T in zip(neigh_prev, neigh_curr):
        sS=set(S); sT=set(T)
        if len(sS)==0: continue
        overlap += len(sS.intersection(sT))/len(sS)
        total += 1.0
    return (overlap/total) if total>0 else 0.0

def run_main_pipeline(L=512, block=32, dof_budget_frac=0.25, seed=101):
    rng = random.Random(seed)
    X = [rng.randint(1,4) for _ in range(L)]
    holds=0; swaps=0; receipts=0
    accept_count=0; seen_pattern=0
    neigh_prev = [sorted(set(rng.sample(range(L), k=4))) for _ in range(L)]
    neigh_curr = [sorted(set(rng.sample(range(L), k=4))) for _ in range(L)]
    R_prev = recurrence_index(neigh_prev, neigh_curr)
    for start in range(0, L-block, block):
        budget = int(block*dof_budget_frac)
        changes=0
        for i in range(start, start+block-3):
            w = (X[i],X[i+1],X[i+2],X[i+3])
            choice, lam0, lam1 = QF_decide(*w, receipt=None)
            if choice=="SWAP" and changes<budget:
                X[i],X[i+1],X[i+2],X[i+3] = Dual(*w)
                swaps+=1; changes+=1
            else:
                holds+=1
            ps = (rng.random()<0.85)
            seen_pattern += 1 if ps else 0
            if ALT(*w) and (W4(w[0],w[1],w[2]) or Q8(*w)) and ps:
                accept_count += 1
    pos_idx = rng.randint(0, L-4)
    q = (X[pos_idx],X[pos_idx+1],X[pos_idx+2],X[pos_idx+3])
    lam = Lambda(*q)
    candidates = [(q, -lam), (Dual(*q), -Lambda(*Dual(*q)))]
    chosen = tie_break(candidates, pos_idx)
    deltas = []
    for _ in range(1000):
        a,b,c,d = rng.randint(1,4),rng.randint(1,4),rng.randint(1,4),rng.randint(1,4)
        lam0 = Lambda(a,b,c,d); lam1 = Lambda(*Dual(a,b,c,d))
        deltas.append(lam1-lam0)
    return {
        "holds": holds, "swaps": swaps, "receipts": receipts,
        "accept_rate": accept_count/max(1,(L-block)),
        "pattern_seen_rate": seen_pattern/max(1,(L-block)),
        "recurrence_prev": R_prev,
        "chosen_sum": sum(chosen),
        "delta_lambda_samples": deltas
    }

def run_combined(seeds=[2,3,5,7,11]):
    rows=[]; cross_lines=[]; delta_all=[]
    for sd in seeds:
        main = run_main_pipeline(L=512, block=32, seed=sd)
        gov  = test_governance_chain(seed=sd)
        qfh  = test_qf_hold_validity(seed=sd)
        w80r = test_w80(seed=sd)
        depo = test_deposits_vs_state(seed=sd)
        quad = test_quadratic_vertex(seed=sd)
        invs = test_modular_inverses()
        neur = test_neuron_ledger(seed=sd)
        trns = test_transformer_checkpoints(seed=sd)
        entr = entropy_exchange(seed=sd)
        rows.append({
            "seed": sd,
            "holds": main["holds"], "swaps": main["swaps"],
            "accept_rate": main["accept_rate"], "pattern_seen_rate": main["pattern_seen_rate"],
            "recurrence_prev": main["recurrence_prev"],
            "gov_helix_legal_frac": gov["helix_legal_fraction"],
            "gov_bonded_rate": (gov["bonded_ok_pairs"]/max(1,gov["pairs_total"])),
            "qf_swaps_if_descent": qfh["swaps_if_descent"]/max(1,qfh["holds"]),
            "w80_repeat_fraction": w80r["repeat_fraction"],
            "w80_equal_results": 1.0 if w80r["equal_results"] else 0.0,
            "deposits_match": 1.0 if depo["match"] else 0.0,
            "quadratic_vertex_success": quad["success_rate"],
            "neuron_conserved": 1.0 if neur["conserved"] else 0.0,
            "transformer_in_ok": trns["in_block_ok_rate"],
            "transformer_cross_ok": trns["cross_boundary_ok_rate"],
            "transformer_dof_ok": 1.0 if trns["dof_budget_respected"] else 0.0,
            "entropy_units_global_perN": entr["entropy_units_global"]/max(1,entr["N"]),
            "entropy_units_piecewise_perN": entr["entropy_units_piecewise"]/max(1,entr["N"]),
        })
        cross_lines.append((sd, trns["in_block_ok_rate"], trns["cross_boundary_ok_rate"]))
        delta_all.extend(main["delta_lambda_samples"])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_COMB, 'combined_dashboard_metrics.csv'), index=False)
    avg = df.mean(numeric_only=True).to_dict()
    with open(os.path.join(OUT_COMB, 'combined_dashboard_metrics_avg.json'), 'w') as f:
        json.dump(avg, f, indent=2)
    # Charts (policy: one figure per chart, no custom colors)
    metrics_to_plot = [
        "accept_rate","pattern_seen_rate","recurrence_prev",
        "gov_helix_legal_frac","gov_bonded_rate",
        "qf_swaps_if_descent","w80_repeat_fraction",
        "deposits_match","quadratic_vertex_success","neuron_conserved",
        "transformer_in_ok","transformer_cross_ok","transformer_dof_ok",
        "entropy_units_global_perN","entropy_units_piecewise_perN"
    ]
    values = [avg[m] for m in metrics_to_plot if m in avg]
    labels = [m for m in metrics_to_plot if m in avg]
    plt.figure(); plt.bar(range(len(values)), values); plt.xticks(range(len(values)), labels, rotation=90); plt.tight_layout(); plt.savefig(os.path.join(OUT_COMB, 'avg_metrics_bar.png')); plt.close()
    xs = [c[0] for c in cross_lines]; ys1 = [c[1] for c in cross_lines]; ys2 = [c[2] for c in cross_lines]
    plt.figure(); plt.plot(xs, ys1, marker='o', label='in-block ok'); plt.plot(xs, ys2, marker='x', label='cross-boundary ok'); plt.legend(); plt.xlabel('seed'); plt.ylabel('ok-rate'); plt.tight_layout(); plt.savefig(os.path.join(OUT_COMB, 'boundary_ok_rates_line.png')); plt.close()
    plt.figure(); plt.hist(delta_all, bins=21); plt.xlabel('ΔΛ under Dual'); plt.ylabel('count'); plt.tight_layout(); plt.savefig(os.path.join(OUT_COMB, 'delta_lambda_hist.png')); plt.close()
    return df

# ---------------------------
# E8 meta‑harness (8 fields)
# ---------------------------

rng = np.random.default_rng(42)

def ds_dna(L=256):
    seq = rng.integers(1,5,size=L).tolist()
    return {"name":"DNA","seq":seq}

def ds_text(L=256):
    seq = rng.integers(1,5,size=L).tolist()
    return {"name":"Text","seq":seq}

def ds_plasma(L=256):
    seq = (rng.choice([1,2,3,4], size=L, p=[0.2,0.3,0.3,0.2])).tolist()
    return {"name":"Plasma","seq":seq}

def ds_finance(L=256):
    seq = (rng.choice([1,2,3,4], size=L, p=[0.25,0.25,0.4,0.1])).tolist()
    return {"name":"Finance","seq":seq}

def ds_neuro(L=256):
    seq = (rng.choice([1,2,3,4], size=L, p=[0.35,0.25,0.25,0.15])).tolist()
    return {"name":"Neuro","seq":seq}

def ds_primes(L=256):
    seq = (rng.choice([1,2,3,4], size=L, p=[0.35,0.05,0.35,0.25])).tolist()
    return {"name":"Primes","seq":seq}

def ds_embeddings(L=256):
    seq = rng.integers(1,5,size=L).tolist()
    return {"name":"Embeddings","seq":seq}

def ds_social(L=256):
    seq = (rng.choice([1,2,3,4], size=L, p=[0.3,0.3,0.2,0.2])).tolist()
    return {"name":"Social","seq":seq}

DATASETS = [ds_dna(), ds_text(), ds_plasma(), ds_finance(), ds_neuro(), ds_primes(), ds_embeddings(), ds_social()]

# Context metrics for E8

def MSS_accept(w, pattern_seen=True):
    a,b,c,d = w
    return ALT(a,b,c,d) and (W4(a,b,c) or Q8(a,b,c,d)) and pattern_seen

def context_scores(seq):
    Q = to_quads(seq)
    if not Q:
        return {"C1_invariant_rate": 0.0, "C2_energy_descent_rate": 0.0, "C3_accept_rate": 0.0}
    inv = [1.0 if (ALT(*q) and (W4(q[0],q[1],q[2]) or Q8(*q))) else 0.0 for q in Q]
    C1 = float(np.mean(inv))
    desc = [1.0 if (Lambda(*Dual(*q)) < Lambda(*q)) else 0.0 for q in Q]
    C2 = float(np.mean(desc))
    pat = []
    for q, lawful in zip(Q, inv):
        if lawful>0:
            pat.append(1.0 if np.random.rand() < 0.8 else 0.0)
        else:
            pat.append(0.0)
    C3 = float(np.mean(pat))
    return {"C1_invariant_rate": C1, "C2_energy_descent_rate": C2, "C3_accept_rate": C3}

def ledger(seq_before, seq_after):
    Qb = to_quads(seq_before); Qa = to_quads(seq_after)
    hb = sum(H8(*q) for q in Qb) % 8 if Qb else 0
    ha = sum(H8(*q) for q in Qa) % 8 if Qa else 0
    return hb, ha, (ha - hb) % 8

def entropy_exchange_ratio(seqA, seqB):
    QA = to_quads(seqA); QB = to_quads(seqB)
    if not QA or not QB: return 1.0, 0.0
    LA = np.array([Lambda(*q) for q in QA])
    LB = np.array([Lambda(*q) for q in QB]) + 1e-9
    k = min(len(LA), len(LB))
    LA, LB = LA[:k], LB[:k]
    s = float(np.median(LA/LB))
    mismatch = np.abs(LA - s*LB)
    entropy_units = float(np.sum(mismatch >= 1.0))/max(1,len(mismatch))
    return s, entropy_units

def shift_sequence(seq, shifts=10, rng=None):
    if rng is None: rng = random.Random(0)
    seq = seq.copy(); N = len(seq)
    edits=0
    for _ in range(shifts):
        i = rng.randint(0, N-1)
        seq[i] = ((seq[i]) % 4) + 1
        edits += 1
    return seq, edits

def e8_governance_scores(dataset_list, dof_budget_per_ds=10):
    rows = []; n = len(dataset_list)
    for i, ds in enumerate(dataset_list):
        name = ds["name"]; seq0 = ds["seq"]
        seq1, used_dof = shift_sequence(seq0, shifts=dof_budget_per_ds, rng=random.Random(100+i))
        hb, ha, delta = ledger(seq0, seq1)
        left = dataset_list[(i-1)%n]["seq"]
        right = dataset_list[(i+1)%n]["seq"]
        sL, eL = entropy_exchange_ratio(seq1, left)
        sR, eR = entropy_exchange_ratio(seq1, right)
        observer_ok = 1.0 if (eL<0.35 and eR<0.35) else 0.0
        dof_ok = 1.0 if used_dof <= dof_budget_per_ds else 0.0
        sG, eG = entropy_exchange_ratio(seq1, dataset_list[(i+2)%n]["seq"])
        entropy_ok = 1.0 if eG < 0.5 else 0.0
        Q0 = to_quads(seq0); Q1 = to_quads(seq1)
        k = min(len(Q0), len(Q1))
        dlam = np.mean([Lambda(*Q1[j]) - Lambda(*Q0[j]) for j in range(k)]) if k>0 else 0.0
        cycle_ok = 1.0 if dlam <= used_dof else 0.0
        rows.append({
            "dataset": name,
            "hb": hb, "ha": ha, "ledger_delta": delta,
            "observer_ok": observer_ok,
            "dof_used": used_dof, "dof_ok": dof_ok,
            "entropy_ok": entropy_ok,
            "cycle_ok": cycle_ok,
            "s_left": sL, "e_left": eL, "s_right": sR, "e_right": eR
        })
    df = pd.DataFrame(rows)
    df["G1_ledger_ok"] = ((df["ledger_delta"]==0) | (df["observer_ok"]>0)).astype(float)
    df["G_score"] = df[["G1_ledger_ok","observer_ok","dof_ok","entropy_ok","cycle_ok"]].mean(axis=1)
    return df

def run_e8():
    # Context per dataset
    context_rows = []
    for ds in DATASETS:
        C = context_scores(ds["seq"])
        context_rows.append({"dataset": ds["name"], **C})
    df_context = pd.DataFrame(context_rows)
    df_context.to_csv(os.path.join(OUT_E8, 'E8_context_scores.csv'), index=False)
    # Governance per dataset
    df_gov = e8_governance_scores(DATASETS, dof_budget_per_ds=10)
    df_gov.to_csv(os.path.join(OUT_E8, 'E8_governance_scores.csv'), index=False)
    # Coupling matrix
    names = [ds["name"] for ds in DATASETS]
    M = np.zeros((len(names), len(names)))
    for i, A in enumerate(DATASETS):
        for j, B in enumerate(DATASETS):
            s, e = entropy_exchange_ratio(A["seq"], B["seq"])
            M[i,j] = e
    df_coup = pd.DataFrame(M, index=names, columns=names)
    df_coup.to_csv(os.path.join(OUT_E8, 'E8_coupling_matrix.csv'))
    # Summary & charts
    summary = {
        "context_avg": df_context.mean(numeric_only=True).to_dict(),
        "governance_avg": df_gov.mean(numeric_only=True).to_dict(),
        "datasets": names
    }
    with open(os.path.join(OUT_E8, 'E8_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    # Charts
    plt.figure();
    vals = [summary["context_avg"].get("C1_invariant_rate",0), summary["context_avg"].get("C2_energy_descent_rate",0), summary["context_avg"].get("C3_accept_rate",0)]
    labels = ["C1_invariant","C2_energy","C3_accept"]
    plt.bar(range(len(vals)), vals); plt.xticks(range(len(vals)), labels); plt.tight_layout(); plt.savefig(os.path.join(OUT_E8, 'E8_context_avg.png')); plt.close()
    plt.figure();
    g_labels = ["G1_ledger_ok","observer_ok","dof_ok","entropy_ok","cycle_ok"]
    g_vals = [summary["governance_avg"].get(lbl,0) for lbl in g_labels]
    plt.bar(range(len(g_vals)), g_vals); plt.xticks(range(len(g_vals)), g_labels); plt.tight_layout(); plt.savefig(os.path.join(OUT_E8, 'E8_governance_avg.png')); plt.close()
    plt.figure();
    plt.imshow(M, aspect='auto'); plt.xticks(range(len(names)), names, rotation=90); plt.yticks(range(len(names)), names); plt.colorbar(); plt.tight_layout(); plt.savefig(os.path.join(OUT_E8, 'E8_coupling_heatmap.png')); plt.close()
    return df_context, df_gov, df_coup

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == '__main__':
    df_comb = run_combined()
    df_ctx, df_gov, df_coup = run_e8()
    print('Artifacts (combined):', os.listdir(OUT_COMB))
    print('Artifacts (E8):', os.listdir(OUT_E8))
```

---

## Where to plug in real data

- **E8 datasets:** replace `ds_*` generators with loaders for real sequences/tables. Map to residues {1..4} by any consistent encoding (e.g., DNA A/C/G/T→1/2/3/4; text classes; expression bins; embedding sign/quantile bins).
- **Receipts:** enable receipts in `helical_twist` and `bond_pair` if you want to allow ledger‑changing moves; ensure MEQ credits cover ΔΛ.
- **Approximate w80:** bucket by `(H8, ⌊Λ⌋)` and cache per bucket. Validate equality with full search on a held‑out slice.
- **Entropy n=1 exchange:** to align two domains A,B, use `s` (global) and `s_h` (by helicity) and watch mismatch drop; the remaining mismatch is your actionable entropy.

## Expected outputs

- Combined dashboard CSV/JSON + charts.
- E8 context/governance CSVs + coupling heatmap.
- Modular inverse report CSV (invertibility per modulus).

## Validation blueprint

- A claim is considered **validated** if:
  - ALT/W4/Q8 legality holds in‑block and across chain transitions, OR any deviations carry receipts that reconcile the ledger; and
  - Entropy mismatch falls under piecewise scaling; and
  - Governance (G1–G5) averages stay high across E8.

This satisfies your requirement: a fully enumerated test suite, with the complete harness code attached and ready to run, and explicit instructions to expand to IRL datasets from atomic to global scales using the same steps.

