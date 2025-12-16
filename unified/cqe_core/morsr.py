
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
import time, math, random

from .state import CQEState
from .objective import Phi
from .operators import OperatorLibrary
from .ledger import Ledger
from .provenance import Provenance
from .validation import IntegrityPanel

@dataclass
class MORSR:
    phi: Phi
    ops: OperatorLibrary
    ledger: Ledger
    provenance: Provenance
    rs: random.Random = field(default_factory=lambda: random.Random(0))

    def run(self, state: CQEState, budget: int = 200, eps: float = 1e-9) -> Tuple[CQEState, IntegrityPanel]:
        run_id = self.provenance.start()
        self.ledger.open(self.provenance.manifold_id)

        ip = IntegrityPanel()
        best_phi = self.phi.total(state.lanes, state.parity_ok)
        state.phi_total = best_phi
        ip.phi_series.append(best_phi)

        op_names = self.ops.shortlist()
        t0 = time.time()

        for step in range(budget):
            op = self.rs.choice(op_names)

            # Sample parameters for certain ops
            if op == "R_theta":
                k = self.rs.randint(1, 7)
                new_lanes, new_parity = self.ops.R_theta(state.lanes, state.parity_ok, k=k)
            elif op == "Weyl_reflect":
                idx = self.rs.randint(0, len(state.lanes)-1)
                new_lanes, new_parity = self.ops.Weyl_reflect(state.lanes, state.parity_ok, idx=idx)
            elif op == "Midpoint":
                new_lanes, new_parity = self.ops.Midpoint(state.lanes, state.parity_ok)
            elif op == "ECC_parity":
                new_lanes, new_parity = self.ops.ECC_parity(state.lanes, state.parity_ok)
            elif op == "SingleInsert":
                idx = self.rs.randint(0, len(state.lanes)-1)
                val = self.rs.uniform(-0.15, 0.15)
                new_lanes, new_parity = self.ops.SingleInsert(state.lanes, state.parity_ok, idx=idx, val=val)
            elif op == "ParityMirror":
                strength = self.rs.uniform(0.1, 0.35)
                new_lanes, new_parity = self.ops.ParityMirror(state.lanes, state.parity_ok, strength=strength)
            else:
                new_lanes, new_parity = (list(state.lanes), state.parity_ok)

            new_phi = self.phi.total(new_lanes, new_parity)
            delta = new_phi - best_phi

            accepted = (delta <= eps)  # monotone acceptance
            if accepted:
                # track parity changes
                if (not state.parity_ok) and new_parity:
                    ip.parity_fixes += 1
                if state.parity_ok and (not new_parity):
                    ip.parity_breaks += 1

                state.lanes = new_lanes
                state.parity_ok = new_parity
                state.phi_total = new_phi
                best_phi = new_phi
                ip.accepts += 1
                ip.plateau_ticks = (ip.plateau_ticks + 1) if abs(delta) < 1e-12 else 0
            else:
                ip.rejects += 1
                ip.plateau_ticks = 0

            ip.phi_series.append(best_phi)

            # ledger receipt
            receipt = {
                "ts": time.time(),
                "run_id": run_id,
                "op": op,
                "delta_phi": delta,
                "accepted": accepted,
                "parity_status": "OK" if state.parity_ok else "VIOLATION",
                "phi": best_phi,
            }
            self.ledger.append(receipt)

        self.ledger.close()
        return state, ip
