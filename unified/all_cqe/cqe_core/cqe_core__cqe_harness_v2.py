#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CQE Controller Harness — single-file skeleton (stdlib-only)

This module implements a receipts-first, geometry-governed controller that:
  • Senses (slice calculus observables on wedge lattices W∈{80,240} for decagon/octagon viewers)
  • Plans (Socratic Q/A on objectives and invariants)
  • Acts (pose rotation/reflection, least-action repair, clone tiling, lattice switch)
  • Checks (ΔΦ monotonicity, validators across LATT/CRT/FRAC/SACNUM stubs)
  • Emits receipts (append-only JSONL ledger + latent pose cache row)

It is intentionally self-contained (stdlib only) and designed to be dropped into a repo as the spine.
Real slice validators can be wired in later by replacing stub methods.

Usage (CLI):
  python cqe_harness.py --text "some phrase" --policy channel-collapse --out runs/demo

Outputs:
  runs/<stamp>/ledger.jsonl   (receipts)
  runs/<stamp>/lpc.csv        (latent pose cache rows, '|' delimited)
  runs/<stamp>/summary.txt    (human-readable summary)

Author: CQE custodian
License: MIT
"""

from __future__ import annotations
import argparse
import dataclasses as dc
import hashlib
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# -----------------------------------------------------------------------------
# Utility: hash + timestamps
# -----------------------------------------------------------------------------

def now_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def _json_default(o: Any) -> str:
    try:
        return repr(o)
    except Exception:
        return f"<unrepr {type(o).__name__}>"

def sha256_hex(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=_json_default).encode("utf-8")
    return hashlib.sha256(b).hexdigest()

# -----------------------------------------------------------------------------
# Tokenization → faces (decagon/octagon) — minimal, deterministic
# -----------------------------------------------------------------------------

@dc.dataclass
class Face:
    """A 'face' is a small numeric stream view (mod 10 / mod 8) for slice calculus."""
    values: List[int]
    base: int
    label: str

def text_to_faces(text: str) -> Tuple[Face, Face]:
    """Map text into two aligned numeric streams: mod10 (decagon) and mod8 (octagon). Deterministic."""
    # FNV-1a 64-bit rolling hash over bytes; split into bases.
    h = 0xcbf29ce484222325  # FNV offset
    d10: List[int] = []
    d8: List[int] = []
    for ch in text.encode("utf-8", errors="ignore"):
        h ^= ch
        h = (h * 0x100000001b3) & ((1<<64)-1)  # FNV prime
        d10.append((h // 2654435761) % 10)
        d8.append((h // 11400714819323198485) % 8)
    if not d10:
        d10 = [0]; d8 = [0]
    return Face(d10, 10, "decagon"), Face(d8, 8, "octagon")

# -----------------------------------------------------------------------------
# Slice lattice & observables
# -----------------------------------------------------------------------------

@dc.dataclass
class SliceObservables:
    theta: List[float]                         # lattice angles (radians)
    extreme_idx: List[int]                     # i(θ): index of extreme sample (by projection on θ)
    quadrant_bins: List[Tuple[int,int,int,int]]  # q(θ): counts per quadrant-like bin
    chord_hist: List[Dict[int,int]]            # hΔ(θ): histogram of chord steps (constant in this simple model)
    perm: List[List[int]]                      # π(θ): top-k order (indices) by projection
    braid_current: List[int]                   # B(θ): adjacent transposition count per step
    energies: Dict[str, float]                 # Dirichlet energies over chosen signals

class SliceSensors:
    def __init__(self, W: int = 80, topk: int = 16):
        self.W = W
        self.topk = topk
        self.theta = [2.0 * math.pi * m / W for m in range(W)]

    # --- projections & helpers ---
    @staticmethod
    def _project_stream(vals: Sequence[int], base: int, theta: float) -> List[float]:
        # Treat each sample as a point on its base-gon; project onto direction θ
        out: List[float] = []
        for v in vals:
            ang = 2.0 * math.pi * (v % base) / base
            out.append(math.cos(ang - theta))
        return out

    @staticmethod
    def _argmax_idx(arr: Sequence[float]) -> int:
        best = -1e9; idx = 0
        for i, x in enumerate(arr):
            if x > best:
                best = x; idx = i
        return idx

    @staticmethod
    def _quadrant_bins(vals: Sequence[int], base: int, theta: float) -> Tuple[int,int,int,int]:
        # Bin positions after rotation; 4 equal arcs on the circle
        bins = [0,0,0,0]
        for v in vals:
            ang = (2.0 * math.pi * (v % base) / base - theta) % (2.0 * math.pi)
            q = int((ang / (2.0 * math.pi)) * 4.0) % 4
            bins[q] += 1
        return (bins[0], bins[1], bins[2], bins[3])

    @staticmethod
    def _chord_hist(vals: Sequence[int], base: int) -> Dict[int,int]:
        c: Dict[int,int] = {}
        for a, b in zip(vals, vals[1:]):
            step = (b - a) % base
            c[step] = c.get(step, 0) + 1
        return c

    @staticmethod
    def _perm_by_projection(vals: Sequence[int], base: int, theta: float, topk: int) -> List[int]:
        proj = SliceSensors._project_stream(vals, base, theta)
        order = sorted(range(len(vals)), key=lambda i: proj[i], reverse=True)
        return order[:min(topk, len(order))]

    @staticmethod
    def _adjacent_transpositions(prev: List[int], curr: List[int]) -> int:
        # Count inversions between adjacent elements moving from prev to curr (small topk, O(n^2) ok)
        pos_curr = {v: i for i, v in enumerate(curr)}
        common = [v for v in prev if v in pos_curr]
        mapped = [pos_curr[v] for v in common]
        inv = 0
        for i in range(len(mapped)):
            for j in range(i+1, len(mapped)):
                if mapped[i] > mapped[j]:
                    inv += 1
        return inv

    def compute(self, face: Face) -> SliceObservables:
        W, base, vals = self.W, face.base, face.values
        theta = self.theta
        extreme_idx: List[int] = []
        quadrant_bins: List[Tuple[int,int,int,int]] = []
        chord_hist: List[Dict[int,int]] = []
        perm: List[List[int]] = []
        braid_current: List[int] = []

        prev_order: Optional[List[int]] = None
        for th in theta:
            proj = self._project_stream(vals, base, th)
            extreme_idx.append(self._argmax_idx(proj))
            quadrant_bins.append(self._quadrant_bins(vals, base, th))
            chord_hist.append(self._chord_hist(vals, base))  # independent of θ in this simple model
            order = self._perm_by_projection(vals, base, th, self.topk)
            perm.append(order)
            if prev_order is None:
                braid_current.append(0)
            else:
                braid_current.append(self._adjacent_transpositions(prev_order, order))
            prev_order = order

        # Energies (Dirichlet) on discrete circle
        def dirichlet_energy_int(seq: Sequence[int]) -> float:
            n = len(seq); acc = 0.0
            for i in range(n):
                a = seq[(i+1) % n]; b = seq[i]; c = seq[(i-1) % n]
                acc += float((a - 2*b + c)**2)
            return acc / float(max(1, n))

        def q_imbalance_energy(qbins: Sequence[Tuple[int,int,int,int]]) -> float:
            e = 0.0
            for q in qbins:
                m = sum(q) / 4.0
                e += sum((qi - m)**2 for qi in q)
            return e / float(max(1, len(qbins)))

        energies = {
            "E_extreme": dirichlet_energy_int(extreme_idx),
            "E_quads": q_imbalance_energy(quadrant_bins),
            "Crossings": float(sum(braid_current)),
        }
        return SliceObservables(theta, extreme_idx, quadrant_bins, chord_hist, perm, braid_current, energies)

# -----------------------------------------------------------------------------
# Actuators
# -----------------------------------------------------------------------------

class Actuators:
    @staticmethod
    def least_action_repair(vals: List[int], base: int) -> Tuple[List[int], Dict[str, Any]]:
        """Odd-prime → next odd coprime mod base (toy). Returns repaired list + residue stats."""
        def next_odd_coprime(x: int) -> int:
            y = x
            for _ in range(base + 3):
                y = (y + 1) % base
                if (y % 2 == 1) and (math.gcd(y, base) == 1):
                    return y
            return x
        edits = 0; out: List[int] = []
        for v in vals:
            if (v % 2 == 1) and (math.gcd(v, base) == 1):
                out.append(v)
            else:
                out.append(next_odd_coprime(v)); edits += 1
        info = {"edits": edits, "edit_rate": edits / float(max(1, len(vals)))}
        return out, info

    @staticmethod
    def rotate(vals: List[int], steps: int) -> List[int]:
        if not vals: return vals
        s = steps % len(vals)
        return vals[-s:] + vals[:-s]

    @staticmethod
    def reflect(vals: List[int], base: int) -> List[int]:
        return [(base - v) % base for v in vals]

    @staticmethod
    def minK_to_balance(qbins: Sequence[Tuple[int,int,int,int]]) -> int:
        need = 0
        for q in qbins:
            need = max(need, max(q) - min(q))
        return need

# -----------------------------------------------------------------------------
# Validators (stubs)
# -----------------------------------------------------------------------------

@dc.dataclass
class GateResult:
    ok: bool
    escrow: bool = False
    reason: str = ""
    details: Optional[Dict[str, Any]] = None

class Validators:
    @staticmethod
    def delta_phi(prevJ: float, newJ: float) -> GateResult:
        return GateResult(ok=(newJ <= prevJ + 1e-12), escrow=(newJ > prevJ), reason=("J↑" if newJ > prevJ else ""))

    @staticmethod
    def latt_stub(face: Face) -> GateResult:
        return GateResult(ok=True)

    @staticmethod
    def crt_stub(face: Face) -> GateResult:
        return GateResult(ok=True)

    @staticmethod
    def frac_stub(obs: SliceObservables) -> GateResult:
        return GateResult(ok=True)

    @staticmethod
    def sacnum_stub(face: Face) -> GateResult:
        return GateResult(ok=True)

# -----------------------------------------------------------------------------
# Policy, State, Receipts, LPC
# -----------------------------------------------------------------------------

@dc.dataclass
class Policy:
    name: str
    alpha: float = 0.5
    beta: float = 0.1
    gamma: float = 0.3
    delta: float = 0.1
    kappa: float = 0.0
    dihedral_reflection: bool = True
    lattice_candidates: Tuple[int, ...] = (80, 240)
    viewers: Tuple[int, int] = (10, 8)  # decagon, octagon
    max_iter: int = 12

    @staticmethod
    def presets(kind: str) -> "Policy":
        kind = (kind or "channel-collapse").lower()
        if kind == "channel-collapse":
            return Policy("channel-collapse", 0.5, 0.1, 0.3, 0.1, 0.0, True, (80, 240), (10, 8), 12)
        if kind == "knot-sensitive":
            return Policy("knot-sensitive", 0.4, 0.35, 0.15, 0.1, 0.0, True, (80, 240), (10, 8), 12)
        if kind == "numerology-bridge":
            return Policy("numerology-bridge", 0.45, 0.1, 0.35, 0.05, 0.05, True, (80, 240), (10, 8), 12)
        return Policy(kind)

@dc.dataclass
class Receipt:
    claim: str
    pre: Dict[str, Any]
    post: Dict[str, Any]
    energies: Dict[str, float]
    keys: Dict[str, Any]
    braid: Dict[str, Any]
    validators: Dict[str, bool]
    parity64: str
    pose_salt: str
    merkle: Dict[str, Any]

@dc.dataclass
class LPCRow:
    face_id: str
    channel: str
    idx_range: Tuple[int,int]
    equalizing_angle_deg: float
    pose_key_W80: str
    d10_key: str
    d8_key: str
    joint_key: str
    writhe: int
    crossings: int
    clone_K: int
    quad_var_at_eq: float
    repair_family_id: str
    residues_hash: str
    proof_hash: str

# -----------------------------------------------------------------------------
# Keys & objective
# -----------------------------------------------------------------------------

class Keys:
    @staticmethod
    def pose_key_W(face: Face, obs: SliceObservables) -> str:
        # Rotation/reflection-invariant canonical key from extreme index sequence
        seq = list(obs.extreme_idx)
        W = len(seq)
        rots = [tuple(seq[i:]+seq[:i]) for i in range(W)]
        rets = [tuple(reversed(r)) for r in rots]
        canon = min(rots + rets)
        return json.dumps(list(canon), ensure_ascii=False)

    @staticmethod
    def delta_key(face: Face) -> str:
        vals = face.values
        if not vals:
            return "[]"
        steps = [int((b - a) % face.base) for a, b in zip(vals, vals[1:])]
        return json.dumps(steps[:128], ensure_ascii=False)

    @staticmethod
    def joint_key(dec_key: str, oct_key: str) -> str:
        return sha256_hex([dec_key, oct_key])

class Objective:
    @staticmethod
    def J(policy: Policy, obs: SliceObservables, d10_key: str, d8_key: str, repair_info: Dict[str,Any], pose_key: str) -> float:
        E_i = obs.energies.get("E_extreme", 0.0)
        Cross = obs.energies.get("Crossings", 0.0)
        # mismatch: naive Hamming distance between Δ-keys
        mismatch = 1.0
        try:
            a = json.loads(d10_key); b = json.loads(d8_key)
            n = min(len(a), len(b))
            mismatch = sum(1 for i in range(n) if a[i] != b[i]) / float(max(1, n))
        except Exception:
            mismatch = 1.0
        residue = float(repair_info.get("edits", 0))
        # pose dispersion proxy (hash spread)
        dispersion = (hash(pose_key) & 0xFFFF) / 65535.0
        return ( policy.alpha * E_i + policy.beta * Cross + policy.gamma * mismatch + policy.delta * residue + policy.kappa * dispersion )

# -----------------------------------------------------------------------------
# Receipt writer
# -----------------------------------------------------------------------------

class ReceiptWriter:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.out_dir / "ledger.jsonl"
        self.lpc_path = self.out_dir / "lpc.csv"
        if not self.lpc_path.exists():
            self.lpc_path.write_text(
                "|".join([
                    "face_id","channel","idx_lo","idx_hi","equalizing_angle_deg",
                    "pose_key_W80","d10_key","d8_key","joint_key","writhe","crossings",
                    "clone_K","quad_var_at_eq","repair_family_id","residues_hash","proof_hash"
                ]) + "\n",
                encoding="utf-8"
            )

    def append_ledger(self, rec: Receipt) -> None:
        with self.ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dc.asdict(rec), ensure_ascii=False, default=_json_default) + "\n")

    def append_lpc(self, row: LPCRow) -> None:
        fields = [
            row.face_id, row.channel, str(row.idx_range[0]), str(row.idx_range[1]), f"{row.equalizing_angle_deg:.6f}",
            row.pose_key_W80, row.d10_key, row.d8_key, row.joint_key, str(row.writhe), str(row.crossings),
            str(row.clone_K), f"{row.quad_var_at_eq:.6f}", row.repair_family_id, row.residues_hash, row.proof_hash
        ]
        with self.lpc_path.open("a", encoding="utf-8") as f:
            f.write("|".join(fields) + "\n")

# -----------------------------------------------------------------------------
# CQE Controller
# -----------------------------------------------------------------------------

class CQEController:
    def __init__(self, policy: Policy, out_dir: Path):
        self.policy = policy
        self.out = out_dir
        self.writer = ReceiptWriter(out_dir)

    # --- core loop on a single face ---
    def normalize_face(self, face: Face, channel: str, idx_range: Tuple[int,int]=(0,0)) -> Dict[str, Any]:
        pol = self.policy
        best: Optional[Dict[str, Any]] = None
        # Try repair OFF/ON and lattices (80 then 240)
        for repair_flag in (False, True):
            for W in pol.lattice_candidates:
                sens = SliceSensors(W=W)
                vals = list(face.values)
                rep_info: Dict[str, Any] = {"edits": 0}
                if repair_flag:
                    vals, rep_info = Actuators.least_action_repair(vals, face.base)
                obs = sens.compute(Face(vals, face.base, face.label))

                # Equalizer: choose θ index with minimal quadrant variance at that θ
                q_var = []
                for qb in obs.quadrant_bins:
                    m = sum(qb)/4.0
                    q_var.append(sum((x-m)**2 for x in qb))
                theta_star_idx = min(range(W), key=lambda i: q_var[i])
                theta_deg = 360.0 * theta_star_idx / W

                # Keys and objective
                d10_key = Keys.delta_key(Face(vals, 10, "decagon"))
                d8_key  = Keys.delta_key(Face(vals, 8, "octagon"))
                pose_key = Keys.pose_key_W(Face(vals, face.base, face.label), obs)
                J = Objective.J(pol, obs, d10_key, d8_key, rep_info, pose_key)

                candidate = {
                    "theta_deg": theta_deg,
                    "W": W,
                    "repair": repair_flag,
                    "clones_K": Actuators.minK_to_balance(obs.quadrant_bins),
                    "obs": obs,
                    "rep_info": rep_info,
                    "d10_key": d10_key,
                    "d8_key": d8_key,
                    "pose_key": pose_key,
                    "J": J,
                    "vals": vals,
                }
                if (best is None) or (candidate["J"] < best["J"]):
                    best = candidate
        assert best is not None

        # Validators (stubs for now)
        gates = {
            "ΔΦ": True,
            "LATT": Validators.latt_stub(face).ok,
            "CRT": Validators.crt_stub(face).ok,
            "FRAC": Validators.frac_stub(best["obs"]).ok,
            "SACNUM": Validators.sacnum_stub(face).ok,
        }

        # Receipt
        pre = {"J": best["J"], "theta": best["theta_deg"], "W": best["W"], "repair": best["repair"], "K": best["clones_K"]}
        post = dict(pre)  # single step
        energies = best["obs"].energies
        writhe = int(sum(best["obs"].braid_current))
        braid = {"writhe": writhe, "crossings": writhe, "windows": []}
        parity64 = hashlib.sha256((channel + str(idx_range) + str(best["vals"])).encode()).hexdigest()[:16]
        pose_salt = hashlib.md5(best["pose_key"].encode()).hexdigest()[:8]
        merkle = {"path": sha256_hex([pre, post, energies, braid])[:32]}
        rec = Receipt(
            claim="CQE.normalize",
            pre=pre, post=post,
            energies=energies,
            keys={"pose_W80": best["pose_key"], "d10": best["d10_key"], "d8": best["d8_key"], "joint": Keys.joint_key(best["d10_key"], best["d8_key"])},
            braid=braid,
            validators=gates,
            parity64=parity64,
            pose_salt=pose_salt,
            merkle=merkle,
        )
        self.writer.append_ledger(rec)

        # LPC row
        lpc = LPCRow(
            face_id=sha256_hex([channel, idx_range]),
            channel=channel,
            idx_range=idx_range,
            equalizing_angle_deg=best["theta_deg"],
            pose_key_W80=best["pose_key"],
            d10_key=best["d10_key"],
            d8_key=best["d8_key"],
            joint_key=Keys.joint_key(best["d10_key"], best["d8_key"]),
            writhe=writhe,
            crossings=writhe,
            clone_K=best["clones_K"],
            quad_var_at_eq=float(energies.get("E_quads", 0.0)),
            repair_family_id="odd-coprime@base",
            residues_hash=sha256_hex(best["vals"]),
            proof_hash=merkle["path"],
        )
        # Write LPC
        with open(self.writer.lpc_path, "a", encoding="utf-8") as f:
            f.write("|".join([
                lpc.face_id, lpc.channel, str(lpc.idx_range[0]), str(lpc.idx_range[1]), f"{lpc.equalizing_angle_deg:.6f}",
                lpc.pose_key_W80, lpc.d10_key, lpc.d8_key, lpc.joint_key, str(lpc.writhe), str(lpc.crossings),
                str(lpc.clone_K), f"{lpc.quad_var_at_eq:.6f}", lpc.repair_family_id, lpc.residues_hash, lpc.proof_hash
            ]) + "\n")

        return {
            "state": {k: best[k] for k in ("theta_deg","W","repair","clones_K")},
            "energies": energies,
            "keys": rec.keys,
            "validators": gates,
            "receipt_hash": rec.merkle["path"],
        }

    # High-level convenience
    def normalize(self, text: str) -> Dict[str, Any]:
        dec, octv = text_to_faces(text)
        out = {"policy": dc.asdict(self.policy), "faces": {}}
        out["faces"]["decagon"] = self.normalize_face(dec, channel="decagon", idx_range=(0, len(dec.values)-1))
        out["faces"]["octagon"] = self.normalize_face(octv, channel="octagon", idx_range=(0, len(octv.values)-1))
        # Human summary
        summary = self.out / "summary.txt"
        with summary.open("w", encoding="utf-8") as f:
            f.write(f"Policy: {self.policy.name}\n")
            for ch in ("decagon","octagon"):
                s = out["faces"][ch]["state"]
                f.write(f"{ch}: θ={s['theta_deg']:.2f}°, W={s['W']}, repair={s['repair']}, K={s['clones_K']}\n")
        return out

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="CQE Controller Harness (stdlib-only)")
    p.add_argument("--text", type=str, default="CQE makes pose a control knob.")
    p.add_argument("--policy", type=str, default="channel-collapse", choices=["channel-collapse","knot-sensitive","numerology-bridge"])
    p.add_argument("--out", type=str, default=str(Path("runs") / f"{now_stamp()}_demo"))
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pol = Policy.presets(args.policy)
    ctrl = CQEController(pol, out_dir)
    res = ctrl.normalize(args.text)
    print(json.dumps({"out": args.out, "policy": pol.name, "faces": {k: v["state"] for k,v in res["faces"].items()}}, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
