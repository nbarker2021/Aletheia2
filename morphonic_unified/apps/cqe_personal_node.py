
import sys, json, time
from typing import List
from morphonic_cqe_unified.core.cqe_math import (
    Vector, simple_roots_e8, cartan_from_simple_roots, metric_A_from_cartan,
    phi, try_internal_step, project_to_fundamental_chamber
)
from cqe_time import toroidal_step
from morphonic_cqe_unified.core.cqe_governance import BoundaryReceipt, AuditChain
from morphonic_cqe_unified.sidecar.cqe_sidecar_adapter import CQESidecar

HELP = (
    "CQE Personal Node (Phase 1)\n"
    "Commands:\n"
    "  /help                         - Show commands\n"
    "  /state                        - Show current state vector (8D)\n"
    "  /phi                          - Show \u03A6(state)\n"
    "  /classify                     - Project to Weyl chamber; show \u03B1-dots and reflection count\n"
    "  /time                         - Advance toroidal tick (closure enforced)\n"
    "  /scope <label>                - Set current consent scope label (default: personal)\n"
    "  /channel <3|6|9>              - Set current channel (default: 3)\n"
    "  /step dx1 ... dx8             - Internal step (\u0394\u03A6<=0 enforced, cached)\n"
    "  /boundary dx1 ... dx8 [note]  - Boundary step (receipt generated, cached)\n"
    "  /plan x1 ... x8               - Plan minimal internal move towards target (line-search), apply it\n"
    "  /receipts                     - Show audit chain status and last 5 receipts\n"
    "  /save <file>                  - Save node state + audit chain to JSON\n"
    "  /load <file>                  - Load node state + audit chain from JSON\n"
    "  /report                       - Show sidecar (SpeedLight) report\n"
    "  /exit                         - Quit\n"
)

def parse_vec(args: List[str]) -> Vector:
    if len(args) != 8:
        raise ValueError("Expected 8 numbers")
    return tuple(float(a) for a in args)  # type: ignore

class CQEPersonalNode:
    def __init__(self):
        self.S = simple_roots_e8()
        self.C = cartan_from_simple_roots(self.S)
        self.A = metric_A_from_cartan(self.C, scale=1.0)
        self.x: Vector = (0.0,)*8
        self.scope: str = "personal"
        self.channel: int = 3
        self.audit = AuditChain()
        self.sidecar = CQESidecar()

    def show_state(self):
        print("x =", " ".join(f"{v:+.6f}" for v in self.x))

    def show_phi(self):
        print(f"Phi = {phi(self.A, self.x):.12f}")

    def classify(self):
        v_proj, alpha_dots, refs, ok = project_to_fundamental_chamber(self.x, self.S)
        print("In fundamental chamber:", ok, "reflections:", refs)
        print("alpha·x =", " ".join(f"{d:+.6f}" for d in alpha_dots))

    def tick_time(self):
        x_new, closed = toroidal_step(self.x)
        print("Toroidal closure:", "OK" if closed else "FAIL")
        if closed:
            self.x = x_new

    def set_scope(self, s: str):
        self.scope = s
        print("scope =", self.scope)

    def set_channel(self, c: int):
        if c not in (3,6,9):
            print("Channel must be 3, 6, or 9")
            return
        self.channel = c
        print("channel =", self.channel)

    def step_internal(self, delta: Vector):
        payload = {"op": "internal_step", "x": self.x, "delta": delta}
        def compute():
            x_new, ok, attempts = try_internal_step(self.A, self.x, delta)
            return {"x_new": x_new, "accepted": ok, "attempts": attempts}
        res, cost, rid = self.sidecar.compute(payload, scope=self.scope, channel=self.channel, compute_fn=compute)
        if res["accepted"]:
            self.x = tuple(res["x_new"])
            print(f"ACCEPTED (attempts={res['attempts']}, cache_cost={cost:.6f}s, receipt={rid[:12]}...)")
        else:
            print("REJECTED (\u0394Phi would increase)")

    def step_boundary(self, delta: Vector, note: str = ""):
        payload = {"op": "boundary_step", "x": self.x, "delta": delta, "note": note}
        def compute():
            pre = self.x
            post = tuple(pre[i] + delta[i] for i in range(8))
            dphi = phi(self.A, post) - phi(self.A, pre)
            r = BoundaryReceipt(
                timestamp=time.time(),
                actor="CQE:PersonalNode",
                pre_state=pre,
                post_state=post,
                dphi=dphi,
                channel=9,
                scope=self.scope,
                note=note,
            )
            r.to_cnf_hash_and_sign()
            return {"post": post, "receipt": {
                "cnf_hash": r.cnf_hash, "crt_sig": r.crt_sig, "dphi": dphi, "scope": self.scope, "note": note
            }}
        res, cost, rid = self.sidecar.compute(payload, scope=self.scope, channel=9, compute_fn=compute)
        self.x = tuple(res["post"])
        r = BoundaryReceipt(
            timestamp=time.time(),
            actor="CQE:PersonalNode",
            pre_state=payload["x"],
            post_state=tuple(res["post"]),
            dphi=res["receipt"]["dphi"],
            channel=9,
            scope=self.scope,
            note=note,
        )
        r.cnf_hash = res["receipt"]["cnf_hash"]
        r.crt_sig = res["receipt"]["crt_sig"]
        entry = self.audit.append(r)
        print(f"BOUNDARY COMMIT (dPhi={res['receipt']['dphi']:+.12f})")
        print(f"  receipt_hash = {r.cnf_hash}")
        print(f"  crt_sig      = {r.crt_sig}")
        print(f"  chain_idx    = {entry.idx}, entry_hash={entry.entry_hash}")

    def plan_towards(self, target: Vector):
        delta = tuple(target[i] - self.x[i] for i in range(8))
        payload = {"op": "plan_internal", "x": self.x, "target": target}
        def compute():
            x_new, ok, attempts = try_internal_step(self.A, self.x, delta)
            return {"x_new": x_new, "accepted": ok, "attempts": attempts}
        res, cost, rid = self.sidecar.compute(payload, scope=self.scope, channel=self.channel, compute_fn=compute)
        if res["accepted"]:
            self.x = tuple(res["x_new"])
            print(f"PLANNED&STEPPED (attempts={res['attempts']}, receipt={rid[:12]}...)")
