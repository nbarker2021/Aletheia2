
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lattice Builder & Validator v1 (pure stdlib)
--------------------------------------------
- Build Gram matrices for ADE root lattices (A_n, D_n, E6/7/8) and direct sums.
- Validate integrality, evenness, determinant, unimodularity.
- Enumerate short vectors via branch-and-bound (Cholesky) to detect roots (||v||^2=2).
- Niemeier helper: recognize candidate root systems by spec; Leech check (rootless + even unimodular in 24D).

This is a math validator: it does *not* attempt full glue-code overlattice construction.
"""
from __future__ import annotations
import json, math, argparse, sys
from typing import List, Tuple, Dict

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

Matrix = List[List[float]]
Vector = List[float]

def mat_det(A: Matrix) -> float:
    n = len(A)
    M = [row[:] for row in A]
    det = 1.0
    for i in range(n):
        # pivot
        piv = i
        for r in range(i, n):
            if abs(M[r][i]) > abs(M[piv][i]): piv = r
        if abs(M[piv][i]) < 1e-12: return 0.0
        if piv != i:
            M[i], M[piv] = M[piv], M[i]; det *= -1
        det *= M[i][i]
        pivval = M[i][i]
        for j in range(i+1, n):
            fac = M[j][i] / pivval
            if fac == 0: continue
            for k in range(i, n):
                M[j][k] -= fac * M[i][k]
    return det

def is_integral(A: Matrix) -> bool:
    for i in range(len(A)):
        for j in range(len(A)):
            if abs(A[i][j] - round(A[i][j])) > 1e-10:
                return False
    return True

def is_even(A: Matrix) -> bool:
    # even lattice means x·x ∈ 2Z for all x; for root-lattice Gram (Cartan) this reduces to diag even
    return all(int(round(A[i][i])) % 2 == 0 for i in range(len(A)))

def cholesky(A: Matrix) -> Matrix:
    n = len(A)
    L = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i][k]*L[j][k] for k in range(j))
            if i == j:
                v = A[i][i] - s
                if v <= 0: raise ValueError("Matrix not positive definite")
                L[i][j] = math.sqrt(v)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L

def quad_norm(G: Matrix, x: Vector) -> float:
    # x^T G x
    n = len(G)
    s = 0.0
    for i in range(n):
        for j in range(n):
            s += x[i]*G[i][j]*x[j]
    return s

# ──────────────────────────────────────────────────────────────────────────────
# ADE root-lattice builders via Cartan matrices
# ──────────────────────────────────────────────────────────────────────────────

def cartan_A(n: int) -> Matrix:
    A = [[0]*n for _ in range(n)]
    for i in range(n):
        A[i][i] = 2
        if i>0: A[i][i-1] = -1
        if i<n-1: A[i][i+1] = -1
    return [list(map(float, r)) for r in A]

def cartan_D(n: int) -> Matrix:
    # D_n: chain with a fork at node n-2
    A = [[0]*n for _ in range(n)]
    for i in range(n):
        A[i][i] = 2
    for i in range(n-2):
        A[i][i+1] = A[i+1][i] = -1
    A[n-3][n-1] = A[n-1][n-3] = -1
    return [list(map(float, r)) for r in A]

def cartan_E6() -> Matrix:
    # numbering: chain 1-2-3-4-5 with 3 connected to 6
    A = [[2, -1, 0, 0, 0, 0],
         [-1, 2, -1, 0, 0, 0],
         [0, -1, 2, -1, 0, -1],
         [0, 0, -1, 2, -1, 0],
         [0, 0, 0, -1, 2, 0],
         [0, 0, -1, 0, 0, 2]]
    return [list(map(float, r)) for r in A]

def cartan_E7() -> Matrix:
    # chain 1-2-3-4-5-6 with 3 connected up to 7
    A = [[2, -1, 0, 0, 0, 0, 0],
         [-1, 2, -1, 0, 0, 0, 0],
         [0, -1, 2, -1, 0, 0, -1],
         [0, 0, -1, 2, -1, 0, 0],
         [0, 0, 0, -1, 2, -1, 0],
         [0, 0, 0, 0, -1, 2, 0],
         [0, 0, -1, 0, 0, 0, 2]]
    return [list(map(float, r)) for r in A]

def cartan_E8() -> Matrix:
    # chain 1-2-3-4-5-6-7 with 3 connected up to 8
    A = [[2, -1, 0, 0, 0, 0, 0, 0],
         [-1, 2, -1, 0, 0, 0, 0, 0],
         [0, -1, 2, -1, 0, 0, 0, -1],
         [0, 0, -1, 2, -1, 0, 0, 0],
         [0, 0, 0, -1, 2, -1, 0, 0],
         [0, 0, 0, 0, -1, 2, -1, 0],
         [0, 0, 0, 0, 0, -1, 2, 0],
         [0, 0, -1, 0, 0, 0, 0, 2]]
    return [list(map(float, r)) for r in A]

def block_diag(blocks: List[Matrix]) -> Matrix:
    n = sum(len(b) for b in blocks)
    M = [[0.0]*n for _ in range(n)]
    o = 0
    for B in blocks:
        m = len(B)
        for i in range(m):
            for j in range(m):
                M[o+i][o+j] = B[i][j]
        o += m
    return M

def parse_root_spec(spec: str) -> Matrix:
    """Parse like 'A8 + D16' or 'E8^3' or 'A1^24'."""
    tokens = spec.replace('*','^').replace('+',' ').replace(',',' ').split()
    blocks: List[Matrix] = []
    for tok in tokens:
        if '^' in tok:
            base, times = tok.split('^', 1)
            times = int(times)
        else:
            base, times = tok, 1
        base = base.strip().upper()
        for _ in range(times):
            if base.startswith('A'):
                n = int(base[1:])
                blocks.append(cartan_A(n))
            elif base.startswith('D'):
                n = int(base[1:])
                blocks.append(cartan_D(n))
            elif base == 'E6':
                blocks.append(cartan_E6())
            elif base == 'E7':
                blocks.append(cartan_E7())
            elif base == 'E8':
                blocks.append(cartan_E8())
            else:
                raise ValueError(f"Unknown base '{base}' in spec")
    return block_diag(blocks)

# ──────────────────────────────────────────────────────────────────────────────
# Enumeration of short vectors (Fincke–Pohst style, very small radius)
# ──────────────────────────────────────────────────────────────────────────────

def enumerate_short(G: Matrix, R2: float=2.0, limit: int=100000) -> List[Vector]:
    """Return integer coefficient vectors x with x^T G x <= R2 (excluding x=0).
    Warning: exponential in rank; good for small ranks or small R2.
"""
    n = len(G)
    L = cholesky(G)  # G = L L^T
    sol: List[Vector] = []
    x = [0]*n
    # Precompute for pruning: partial norms using L
    # We'll search in reverse order
    bounds = [0]*n
    def rec(k: int, residual: float):
        nonlocal sol, x
        if k < 0:
            if any(xi!=0 for xi in x):
                sol.append(x[:])
            return
        # compute bound on x_k from residual
        Lkk = L[k][k]
        max_abs = int(math.floor(math.sqrt(max(0.0, residual))/Lkk + 1e-9))
        for t in range(-max_abs, max_abs+1):
            # update residual: || L^T x ||^2 <= R2
            # compute contribution at level k
            s = t * L[k][k]
            for j in range(k+1, n):
                s += x[j]*L[j][k]
            new_res = residual - s*s
            if new_res >= -1e-12:
                x[k] = t
                rec(k-1, new_res)
                if len(sol) >= limit: return
        x[k] = 0
    rec(n-1, R2)
    return sol

def has_root(G: Matrix) -> bool:
    # root = vector of squared length 2 in root lattice basis
    sols = enumerate_short(G, R2=2.0, limit=100000)
    for v in sols:
        q = quad_norm(G, v)
        if abs(q-2.0) < 1e-9:
            return True
    return False

# ──────────────────────────────────────────────────────────────────────────────
# Niemeier helpers
# ──────────────────────────────────────────────────────────────────────────────

NIEMEIER_ROOT_SPECS = [
    "D24", "D16 E8", "E8^3", "A24", "D12^2", "A17 E7", "D10 E7^2",
    "A15 D9", "D8^3", "A12^2", "A11 D7 E6", "E6^4", "A9^2 D6",
    "D6^4", "A8^3", "A7^2 D5^2", "A6^4", "A5^4 D4", "D4^6",
    "A4^6", "A3^8", "A2^12", "A1^24"
]
# Leech is the unique even unimodular rank-24 lattice with no roots.

def validate_properties(G: Matrix) -> Dict:
    d = mat_det(G)
    return {
        "rank": len(G),
        "det": d,
        "integral": is_integral(G),
        "even": is_even(G),
        "unimodular": abs(round(d)-1)==1 and abs(d-1.0) < 1e-8
    }

def niemeier_check(G: Matrix) -> Dict:
    props = validate_properties(G)
    report = {"props": props, "rank": len(G)}
    if len(G) != 24:
        report["niemeier_candidate"] = False
        return report
    if props["even"] and props["unimodular"]:
        # Try to detect roots quickly
        root_present = has_root(G)
        report["root_present"] = root_present
        if not root_present:
            report["classification"] = "Leech (unique even unimodular rank-24 rootless lattice)"
        else:
            report["classification"] = "Even unimodular rank-24 with roots (some Niemeier overlattice)"
        report["niemeier_candidate"] = True
    else:
        report["niemeier_candidate"] = False
    return report

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    b = sub.add_parser("build"); b.add_argument("spec", help="e.g. 'E8^3' or 'A8 + D16'"); b.add_argument("--out", default=None)
    v = sub.add_parser("validate"); v.add_argument("--gram-json", required=True)
    r = sub.add_parser("roots"); r.add_argument("--gram-json", required=True); r.add_argument("--bound", type=float, default=2.0)
    n = sub.add_parser("niemeier"); n.add_argument("--gram-json", required=True)

    args = p.parse_args(argv)

    if args.cmd == "build":
        G = parse_root_spec(args.spec)
        out = json.dumps(G)
        if args.out:
            open(args.out, "w").write(out)
            print(json.dumps({"wrote": args.out, "rank": len(G)}))
        else:
            print(out)
        return

    if args.cmd == "validate":
        G = json.load(open(args.gram_json))
        print(json.dumps(validate_properties(G), indent=2))
        return

    if args.cmd == "roots":
        G = json.load(open(args.gram_json))
        sols = enumerate_short(G, R2=args.bound)
        cnt2 = sum(1 for v in sols if abs(quad_norm(G, v)-2.0) < 1e-9)
        print(json.dumps({"enumerated": len(sols), "roots_of_norm2_found": cnt2}, indent=2))
        return

    if args.cmd == "niemeier":
        G = json.load(open(args.gram_json))
        print(json.dumps(niemeier_check(G), indent=2))
        return

    p.print_help()

if __name__ == "__main__":
    main()
