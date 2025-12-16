
from typing import List, Dict, Any, Tuple
import json, os, time, math, hashlib

def compute_phi(S: Dict[str,Any]) -> float:
    p, c = S["phases"], S["cartan"]
    geom = sum((p[i]-p[(i+1)%len(p)])**2 for i in range(len(p)))
    parity = sum(c) % 2
    spars  = sum(c)
    active = sum(1 for x in p if abs(x)>1e-9)
    kiss   = abs(active - 2)
    return geom + 5*parity + 0.5*spars + 0.1*kiss

def gate(prev: Dict[str,Any], cur: Dict[str,Any]) -> bool:
    return (sum(cur["cartan"])%2==0) and (compute_phi(cur) <= compute_phi(prev))

def op_apply(S: Dict[str,Any], op: str) -> Dict[str,Any]:
    p, c = S["phases"][:], S["cartan"][:]
    if op=='×': p = [round(x+0.10, 10) for x in p]
    elif op=='÷': p = [round((p[i]+p[(i+1)%len(p)])/2, 10) for i in range(len(p))]
    elif op=='%': p = [round(((x%1.0)+1.0)%1.0, 10) for x in p]
    elif op=='~':
        if sum(c)%2==1:
            i = min(range(len(c)), key=lambda i: c[i]); c[i]=1
    elif op=='#2':
        active = sum(1 for x in p if abs(x)>1e-9)
        if active<2:
            i=min(range(len(p)), key=lambda i: abs(p[i])); p[i]=round(p[i]+0.05, 10)
        elif active>2:
            i=max(range(len(p)), key=lambda i: abs(p[i])); p[i]=round(p[i]*0.8, 10)
    return {"phases": p, "cartan": c}

def eval_rung(S: Dict[str,Any], rails: List[List[str]]) -> Tuple[Dict[str,Any], Dict[str,Any], List[Dict[str,Any]]]:
    trials = []
    for seq in rails:
        cur = {"phases": S["phases"][:], "cartan": S["cartan"][:]}
        for op in seq:
            cur = op_apply(cur, op)
        phi = compute_phi(cur)
        ok  = gate(S, cur)
        trials.append({"seq":seq, "state":cur, "phi":phi, "ok":ok})
    accepted = [t for t in trials if t["ok"]]
    if not accepted:
        return S, None, trials
    accepted.sort(key=lambda t: (t["phi"], len(t["seq"]), "".join(t["seq"])))
    win = accepted[0]
    receipt = {
        "winner_seq": win["seq"],
        "phi_before": compute_phi(S),
        "phi_after":  win["phi"],
        "parity_even": sum(win["state"]["cartan"])%2==0,
        "accepted": True,
        "rails_considered": len(rails),
        "repairs_used": 0,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    return win["state"], receipt, trials

def run_ladder(S0: Dict[str,Any], ladder_specs: List[Dict[str,Any]]) -> Tuple[Dict[str,Any], List[Dict[str,Any]]]:
    receipts = []
    cur = S0
    for k, spec in enumerate(ladder_specs, 1):
        rails = spec.get("rails", [["÷"]])
        cur, rec, trials = eval_rung(cur, rails)
        rec = rec or {"winner_seq": [], "phi_before": None, "phi_after": None, "parity_even": False, "accepted": False, "rails_considered": len(rails), "repairs_used": 0, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        rec["rung"] = k
        receipts.append(rec)
    return cur, receipts


class RailGenerator:
    """
    Uses overlay registry + hyperperm oracle to propose rails.
    Priority: LOCKED overlays first, then PENDING within quota.
    Macro proposers are placeholders (evidence-only) governed by a small quota.
    """
    def __init__(self, overlay_registry, oracle, beam_width=8, pending_quota=0.25, macro_quota=0.10):
        self.ov = overlay_registry or {}
        self.oracle = oracle or {"items":{}}
        self.beam = beam_width
        self.pquota = pending_quota
        self.mquota = macro_quota

    def _overlay_rails(self):
        locked = []
        pending = []
        for glyph, entry in (self.ov.items() if isinstance(self.ov, dict) else []):
            ops = entry.get("ops") or entry.get("sequence") or []
            if not ops: continue
            (locked if entry.get("status")=="LOCKED" else pending).append(ops)
        # cap counts
        k_locked = min(len(locked), int(self.beam*(1.0-self.pquota)))
        k_pending = min(len(pending), int(self.beam*self.pquota))
        return locked[:k_locked] + pending[:k_pending]

    def _macro_rails(self):
        # placeholder macro proposers (evidence-only): very small quota
        candidates = [["%","÷"], ["÷","~"], ["×","÷","#2"]]
        k = max(1, int(self.beam*self.mquota))
        return candidates[:k]

    def propose(self):
        rails = self._overlay_rails()
        rails += self._macro_rails()
        # backfill if too few
        base = [["%","÷","~"], ["÷","~"], ["×","÷","#2"]]
        while len(rails) < self.beam:
            rails.append(base[len(rails)%len(base)])
        return rails[:self.beam]
