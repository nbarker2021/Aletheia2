#!/usr/bin/env python3
# Apache-2.0
# CQE Controller — single-file runtime
# Now with overlays: HNF, DSC, PI and pattern miners: Pose Spectrum, Orbit Hash.
import argparse, json, os, sys, math, hashlib, datetime
from pathlib import Path
import numpy as np
import pandas as pd

def now():
    return datetime.datetime.utcnow().isoformat()+"Z"
def sha(s): return hashlib.sha256(s.encode()).hexdigest()

# -------- E8 machinery --------
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
    V = np.zeros_like(X); d_best = np.zeros(N); di = np.zeros(N); dh = np.zeros(N)
    coset = np.empty(N, dtype=object); altV = np.zeros_like(X)
    for i in range(N):
        vb, db, d0, d1, c, av = e8_nearest(X[i])
        V[i]=vb; d_best[i]=db; di[i]=d0; dh[i]=d1; coset[i]=c; altV[i]=av
    return V, d_best, di, dh, coset, altV

def coset_margin(di, dh, eps=1e-9):
    return np.abs(di - dh) / (di + dh + eps)

def hadamard8():
    H2 = np.array([[1,1],[1,-1]],float)
    H4 = np.kron(H2,H2)
    H8 = np.kron(H4,H2)
    return H8/np.sqrt(8.0)

E8_ROOTS = np.array([
    [ 1, -1,  0,  0,  0,  0,  0,  0],
    [ 0,  1, -1,  0,  0,  0,  0,  0],
    [ 0,  0,  1, -1,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  1, -1,  0,  0],
    [ 0,  0,  0,  0,  0,  1, -1,  0],
    [ 0,  0,  0,  0,  0,  1,  1,  0],
    [-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5, 0.5]
], dtype=float)

def pose_bits(X, V, roots, R=None):
    if R is None: R = np.eye(8)
    Rroots = roots @ R.T
    Rroots = Rroots / (np.linalg.norm(Rroots, axis=1, keepdims=True)+1e-9)
    Rres = X - V
    S = (Rres @ Rroots.T)
    return (S >= 0).astype(int)

def alignment_rate(P):
    powers = (1 << np.arange(P.shape[1]))[::-1]
    ints = P @ powers
    vals, counts = np.unique(ints, return_counts=True)
    return counts.max()/P.shape[0], ints

def fixed_rotations(seed=2025):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(8,8)); Q, _ = np.linalg.qr(A)
    H = hadamard8()
    Sflip = np.diag([1,1,1,1,-1,-1,-1,-1])
    return [np.eye(8), H, Q, Sflip@H]

# -------- Loaders --------
def load_matrix(path, dim=None):
    df = pd.read_csv(path)
    num = df.select_dtypes(include=["number"]).values
    if dim is not None:
        if num.shape[1] < dim:
            raise ValueError("Not enough numeric columns for requested dim")
        num = num[:,:dim]
    return num

def gen_synthetic(dim=16, n=8192, seed=42):
    rng = np.random.default_rng(seed)
    if dim==8:
        Theta = rng.random((n,4))*2*math.pi
        X = np.concatenate([np.cos(Theta), np.sin(Theta)], axis=1)
        return X
    if dim==16:
        ThetaA = rng.random((n,5))*2*math.pi
        ThetaB = rng.random((n,5))*2*math.pi
        A = np.concatenate([np.cos(ThetaA[:,:4]), np.sin(ThetaA[:,:4])], axis=1)
        B = np.concatenate([np.cos(ThetaB[:,:4]), np.sin(ThetaB[:,:4])], axis=1)
        return np.hstack([A,B])
    if dim==20:
        ThetaA = rng.random((n,5))*2*math.pi
        ThetaB = rng.random((n,5))*2*math.pi
        A = np.concatenate([np.cos(ThetaA), np.sin(ThetaA)], axis=1)
        B = np.concatenate([np.cos(ThetaB), np.sin(ThetaB)], axis=1)
        return np.hstack([A,B])
    raise ValueError("dim must be 8, 16, or 20")

# -------- Helpers --------
def block8s(X):
    D = X.shape[1]
    if D == 8:
        return [X]
    if D == 16:
        return [X[:,:8], X[:,8:16]]
    if D == 20:
        return [X[:,:8], X[:,10:18]]
    raise ValueError("Dimension must be 8, 16, or 20")

def ensemble_build(packs_dict, main_blocks):
    ensemble = {}
    ensemble.update({f"MAIN_B{bi}": blk for bi, blk in enumerate(main_blocks)})
    for k,v in (packs_dict or {}).items():
        ensemble[str(k)] = v
    return ensemble

def ensemble_choose_Rstar(ensemble, Rset):
    per_pack = {}
    for name, X8 in ensemble.items():
        V, d_best, di, dh, coset, altV = e8_snap_block(X8)
        margins = coset_margin(di, dh)
        per_pack[name] = {"X8":X8, "V":V, "margins":margins}
    best_rate = -1.0; best_R = None; perR_store = {}
    for R in Rset:
        rates = []
        for name, st in per_pack.items():
            P = pose_bits(st["X8"], st["V"], E8_ROOTS, R)
            r, ints = alignment_rate(P)
            rates.append(r)
        mean_rate = float(np.mean(rates))
        perR_store[str(id(R))] = float(mean_rate)
        if mean_rate > best_rate:
            best_rate = mean_rate; best_R = R
    return best_R, best_rate, per_pack, perR_store

def ensemble_metrics(ensemble, Rstar, per_pack):
    best_rates = []; ticket_rates = []; disc_ticket_rates = []
    for name, st in per_pack.items():
        Rset = fixed_rotations(2025)
        br = -1.0
        for R in Rset:
            P = pose_bits(st["X8"], st["V"], E8_ROOTS, R); r,_ = alignment_rate(P)
            if r>br: br=r
        best_rates.append(br)
        tickets = (st["margins"] <= 0.05)
        ticket_rates.append(float(tickets.mean()))
        Pstar = pose_bits(st["X8"], st["V"], E8_ROOTS, Rstar)
        rstar, ints = alignment_rate(Pstar)
        vals, counts = np.unique(ints, return_counts=True)
        modal = vals[np.argmax(counts)]
        disc = (ints != modal)
        disc_ticket_rates.append(float((tickets & disc).mean()))
    ensemble_pose_rate = float(np.mean([alignment_rate(pose_bits(st["X8"], st["V"], E8_ROOTS, Rstar))[0] for st in per_pack.values()]))
    mean_best_rate = float(np.mean(best_rates))
    pose_loss = mean_best_rate - ensemble_pose_rate
    ticket_rate = float(np.mean(ticket_rates))
    disc_ticket_rate = float(np.mean(disc_ticket_rates))
    synergy = ensemble_pose_rate * (1.0 - disc_ticket_rate)
    return {
        "ensemble_pose_rate": ensemble_pose_rate,
        "mean_best_rate": mean_best_rate,
        "pose_loss": pose_loss,
        "ticket_rate": ticket_rate,
        "discordant_ticket_rate": disc_ticket_rate,
        "synergy_index": synergy
    }

# -------- Overlays --------
def overlay_hnf(X8, V, R):
    _, _, di, dh, _, _ = e8_snap_block(X8)
    margin = coset_margin(di, dh)
    P = pose_bits(X8, V, E8_ROOTS, R)
    powers = (1 << np.arange(P.shape[1]))[::-1]
    ints = P @ powers
    vals, counts = np.unique(ints, return_counts=True)
    modal = vals[np.argmax(counts)]
    in_modal = (ints == modal)
    return (margin <= 0.02) & in_modal

def overlay_dsc(X8, V, R):
    P1 = pose_bits(X8, V, E8_ROOTS, R)
    Sflip = np.diag([1,1,1,1,-1,-1,-1,-1])
    Rm = Sflip @ R
    P2 = pose_bits(X8, V, E8_ROOTS, Rm)
    return np.all(P1 == P2, axis=1)

def overlay_pi(X, func_compute_tickets):
    tickets_a = func_compute_tickets(X)
    rng = np.random.default_rng(314)
    scales = rng.uniform(0.5, 2.0, size=X.shape[1])
    Xb = (X * scales)
    colmax = np.maximum(1.0, np.max(np.abs(Xb), axis=0))
    Xb = Xb / colmax
    tickets_b = func_compute_tickets(Xb)
    same = np.array_equal(tickets_a, tickets_b)
    return {"pi_equal": bool(same), "delta": int(np.sum(tickets_a ^ tickets_b))}

def pose_spectrum(X8, V, R):
    P = pose_bits(X8, V, E8_ROOTS, R)
    powers = (1 << np.arange(P.shape[1]))[::-1]
    ints = P @ powers
    vals, counts = np.unique(ints, return_counts=True)
    return pd.DataFrame({"pose_code": vals, "count": counts}).sort_values("count", ascending=False)

def orbit_hash(X8, V, R):
    P = pose_bits(X8, V, E8_ROOTS, R)
    Sflip = np.diag([1,1,1,1,-1,-1,-1,-1]); Rm = Sflip @ R
    Q = pose_bits(X8, V, E8_ROOTS, Rm)
    N = X8.shape[0]; cos = np.zeros(N, dtype=int)
    for i in range(N):
        _, _, d0, d1, c, _ = e8_nearest(X8[i])
        cos[i] = 0 if d0 <= d1 else 1
    powers = (1 << np.arange(P.shape[1]))[::-1]
    a = P @ powers; b = Q @ powers; c = cos.astype(int)
    sig = (a.astype(np.int64) << 9) | (b.astype(np.int64) << 1) | c
    vals, counts = np.unique(sig, return_counts=True)
    return pd.DataFrame({"orbit_sig": vals, "count": counts}).sort_values("count", ascending=False)

# -------- Controller --------
def controller_run(X, outdir, cycles=4, tau_w=0.05, tau_annih=0.01, seed=2025, packs_json=None, ensemble_auto=False):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    # Normalize
    Xn = X.copy().astype(float)
    colmax = np.maximum(1.0, np.max(np.abs(Xn), axis=0))
    Xn = Xn / colmax

    ensembles_rows = []; tickets_rows = []; cycles_rows = []
    prev_ticket_count = None; discovery_stalls = 0
    Rset = fixed_rotations(seed)

    def tickets_from_matrix(Z):
        blocks = block8s(Z)
        mlist = []
        for B in blocks:
            V, d_best, di, dh, coset, altV = e8_snap_block(B)
            m = coset_margin(di, dh); mlist.append(m)
        front = mlist[0] <= tau_w
        for k in range(1, len(mlist)):
            front |= (mlist[k] <= tau_w)
        return front

    for c in range(1, cycles+1):
        blocks = block8s(Xn)
        ensemble = {"MAIN_B0": blocks[0]}
        if len(blocks) > 1: ensemble["MAIN_B1"] = blocks[1]
        Rstar, mean_rate, per_pack, _ = ensemble_choose_Rstar(ensemble, Rset)
        em = ensemble_metrics(ensemble, Rstar, per_pack)
        ensembles_rows.append({"cycle": c, **em})

        # Snap blocks
        snap = []; margins = []
        for B in blocks:
            V, d_best, di, dh, coset, altV = e8_snap_block(B)
            m = coset_margin(di, dh); margins.append(m); snap.append((B,V,di,dh,coset,altV))

        # Tickets
        front = margins[0] <= tau_w
        for k in range(1, len(margins)): front |= (margins[k] <= tau_w)
        idxs = np.where(front)[0]
        cycles_rows.append({"cycle": c, "tickets": int(len(idxs))})

        # Overlays on first block
        X8, V8, di8, dh8, cos8, alt8 = snap[0]
        pd.DataFrame({"index": np.arange(len(X8)), "hnf": overlay_hnf(X8, V8, Rstar).astype(int)}).to_csv(Path(outdir)/"overlays_hnf.csv", index=False)
        pd.DataFrame({"index": np.arange(len(X8)), "dsc": overlay_dsc(X8, V8, Rstar).astype(int)}).to_csv(Path(outdir)/"overlays_dsc.csv", index=False)
        # PI
        pi = overlay_pi(Xn, tickets_from_matrix)
        Path(outdir/"overlays_pi.json").write_text(json.dumps(pi, indent=2))
        # Miners
        pose_spectrum(X8, V8, Rstar).to_csv(Path(outdir)/"pose_spectrum.csv", index=False)
        orbit_hash(X8, V8, Rstar).to_csv(Path(outdir)/"orbit_hash.csv", index=False)

        # Ticket rows
        if len(blocks)==1:
            Vcat = snap[0][1]; Altcat = snap[0][5]
        else:
            Vcat = np.hstack([s[1] for s in snap]); Altcat = np.hstack([s[5] for s in snap])
        move_cost = np.linalg.norm(Altcat - Vcat, axis=1)
        for i in idxs:
            margin_min = float(min([margins[k][i] for k in range(len(blocks))]))
            action = "hold"
            if margin_min <= 0.01: action = "annihilate_to_rails"
            elif move_cost[i] < 0.75: action = "consider_parity_flip"
            tickets_rows.append({"cycle": c, "index": int(i), "margin_min": margin_min,
                                 "move_cost": float(move_cost[i]), "proposed_action": action})

        if prev_ticket_count is not None and len(idxs) == prev_ticket_count:
            discovery_stalls += 1
        else:
            discovery_stalls = 0
        prev_ticket_count = len(idxs)
        if discovery_stalls >= 2: break

    # Write artifacts
    Path(outdir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ensembles_rows).to_csv(Path(outdir)/"ensembles.csv", index=False)
    pd.DataFrame(cycles_rows).to_csv(Path(outdir)/"cycles.csv", index=False)
    if len(tickets_rows)>0: pd.DataFrame(tickets_rows).to_csv(Path(outdir)/"tickets.csv", index=False)
    summary = {
        "cycles_run": int(pd.DataFrame(cycles_rows)["cycle"].max()) if len(cycles_rows)>0 else 0,
        "last_ticket_count": int(pd.DataFrame(cycles_rows)["tickets"].iloc[-1]) if len(cycles_rows)>0 else 0,
        "saturated": bool(discovery_stalls >= 2)
    }
    Path(outdir/"summary.json").write_text(json.dumps(summary, indent=2))
    return summary

def main():
    ap = argparse.ArgumentParser(description="CQE Controller — overlays enabled")
    ap.add_argument("--input", type=str, default="")
    ap.add_argument("--dim", type=int, default=16, choices=[8,16,20])
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--tau_w", type=float, default=0.05)
    ap.add_argument("--out", type=str, default="out")
    args = ap.parse_args()
    if args.input:
        X = load_matrix(args.input, dim=args.dim)
    else:
        X = gen_synthetic(args.dim)
    summary = controller_run(X, args.out, cycles=args.cycles, tau_w=args.tau_w)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
