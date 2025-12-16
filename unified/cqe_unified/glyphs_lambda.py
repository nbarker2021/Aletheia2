
from typing import Dict, Any, List
import math, hashlib, time

def compute_phi(S: Dict[str, Any]) -> float:
    p, c = S["phases"], S["cartan"]
    geom = sum((p[i]-p[(i+1)%len(p)])**2 for i in range(len(p)))
    parity = sum(c) % 2
    spars = sum(c)
    active = sum(1 for x in p if abs(x)>1e-9)
    kiss = abs(active - 2)
    return geom + 5*parity + 0.5*spars + 0.1*kiss

def gate(S, S1) -> bool:
    return (sum(S1["cartan"]) % 2 == 0) and (compute_phi(S1) <= compute_phi(S))

def op_apply(S: Dict[str, Any], op: str) -> Dict[str, Any]:
    p, c = S["phases"][:], S["cartan"][:]
    if op=="×": p = [x+0.10 for x in p]
    elif op=="÷": p = [(p[i]+p[(i+1)%len(p)])/2 for i in range(len(p))]
    elif op=="%": p = [((x%1.0)+1.0)%1.0 for x in p]
    elif op=="~":
        if sum(c)%2==1:
            # flip minimal entry to 1
            i = min(range(len(c)), key=lambda i: c[i])
            c[i]=1
    elif op=="#2":
        active = sum(1 for x in p if abs(x)>1e-9)
        if active<2:
            i=min(range(len(p)), key=lambda i: abs(p[i])); p[i]+=0.05
        elif active>2:
            i=max(range(len(p)), key=lambda i: abs(p[i])); p[i]*=0.8
    return {"phases": p, "cartan": c}

def run_rung(S: Dict[str, Any], rails: List[List[str]]):
    trials=[]; from functools import reduce
    for seq in rails:
        S1 = S
        for op in seq:
            S1 = op_apply(S1, op)
        trials.append((seq, S1))
    trials.sort(key=lambda t: compute_phi(t[1]))
    winner_seq, S1 = trials[0]
    accepted = gate(S, S1)
    receipt = {
        "winner_seq": winner_seq,
        "phi_before": compute_phi(S),
        "phi_after": compute_phi(S1),
        "parity_even": sum(S1["cartan"])%2==0,
        "accepted": bool(accepted)
    }
    return (S1 if accepted else S), receipt

def init_state(n=8, seed=0) -> Dict[str, Any]:
    import random
    random.seed(seed)
    p = [random.uniform(-1,1) for _ in range(n)]
    c = [0]*n
    return {"phases": p, "cartan": c}

class OverlayRegistry:
    def __init__(self):
        self.db = {}  # glyph -> {"ops": [...], "status": "PENDING"/"LOCKED", "evidence": []}
    def set(self, glyph: str, ops: List[str], status="PENDING"):
        self.db[glyph] = {"ops": ops, "status": status, "evidence": []}
    def promote(self, glyph: str):
        if glyph in self.db: self.db[glyph]["status"]="LOCKED"

class HyperpermOracle:
    def __init__(self):
        self.items = {}  # atom_set_key -> {"orders":[{"sequence":seq,"channel":ch,"sig":sig}], "locked": False, "sigs": set()}
    def _key(self, atoms: List[str]):
        return "|".join(sorted(atoms))
    def add_order(self, atoms: List[str], seq: List[str], channel: str):
        key = self._key(atoms)
        if key not in self.items: self.items[key] = {"orders": [], "locked": False, "sigs": set()}
        sig = hashlib.sha256(("::".join(seq)+"||"+channel).encode()).hexdigest()
        if sig not in self.items[key]["sigs"]:
            self.items[key]["orders"].append({"sequence":seq, "channel":channel, "sig":sig})
            self.items[key]["sigs"].add(sig)
            if len(self.items[key]["sigs"])>=8:
                self.items[key]["locked"]=True
        return self.items[key]
