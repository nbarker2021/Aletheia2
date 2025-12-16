
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Callable
from .objective import compute_phi, ObjectiveWeights
from .operators import rotate_Rtheta, weyl_reflect, midpoint_expand, parity_mirror, single_insert, OpResult

@dataclass
class PulseLog:
    step: int
    op: str
    phi_before: float
    phi_after: float
    accepted: bool
    reason_code: str

def apply_with_guard(state: Dict[str, Any], op_name: str, op_fn: Callable, weights: ObjectiveWeights):
    phi_b, _ = compute_phi(state, weights)
    s2 = op_fn(state)
    phi_a, _ = compute_phi(s2, weights)
    dphi = phi_a - phi_b
    accepted = dphi <= 0.0  # monotone acceptance; TODO: escrow policy
    reason = "strict_decrease" if dphi < 0 else ("plateau" if dphi == 0 else "delta_increase")
    return s2 if accepted else state, dphi, accepted, reason, phi_b, phi_a

def morsr_run(state0: Dict[str, Any], weights: ObjectiveWeights, max_steps: int = 64) -> (Dict[str, Any], List[PulseLog]):
    state = dict(state0)
    logs: List[PulseLog] = []
    ops = [
        ("Rtheta", lambda s: rotate_Rtheta(s)),
        ("WeylReflect0", lambda s: weyl_reflect(s, 0)),
        ("Midpoint", lambda s: midpoint_expand(s)),
        ("ParityMirror", lambda s: parity_mirror(s)),
        ("SingleInsert", lambda s: single_insert(s))
    ]
    for step in range(1, max_steps+1):
        improved = False
        for name, fn in ops:
            s2, dphi, acc, reason, pb, pa = apply_with_guard(state, name, fn, weights)
            logs.append(PulseLog(step=step, op=name, phi_before=pb, phi_after=pa, accepted=acc, reason_code=reason))
            if acc and dphi < 0:
                state = s2
                improved = True
        if not improved:
            break  # lane saturation in this toy scaffold
    return state, logs
