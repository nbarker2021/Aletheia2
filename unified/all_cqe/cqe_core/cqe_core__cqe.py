
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple
import random

from .state import CQEState
from .objective import Phi
from .operators import OperatorLibrary
from .morsr import MORSR
from .ledger import Ledger
from .provenance import Provenance

@dataclass
class CQE:
    """Facade. Swap the internals with your real core by overriding apply() if needed."""
    phi: Phi
    ops: OperatorLibrary

    def apply(self, op_name: str, state: CQEState) -> CQEState:
        # Not used directly by orchestrator (MORSR drives ops), but kept for compatibility.
        if op_name == "R_theta":
            new, par = self.ops.R_theta(state.lanes, state.parity_ok, k=1)
        elif op_name == "Midpoint":
            new, par = self.ops.Midpoint(state.lanes, state.parity_ok)
        else:
            new, par = (state.lanes, state.parity_ok)
        state.lanes = new
        state.parity_ok = par
        state.phi_total = self.phi.total(new, par)
        state.steps += 1
        return state

    @staticmethod
    def run_manifold(manifold_id: str, seed: int, weights: Dict[str,float], init_state, out_dir: str, budget:int=200):
        rs = random.Random(seed)
        phi = Phi(
            w_geom=weights.get("geom", 1.0),
            w_parity=weights.get("parity", 1.0),
            w_sparsity=weights.get("sparsity", 0.1),
            w_lattice=weights.get("lattice", 0.2),
        )
        ops = OperatorLibrary(seed=seed)
        cqe = CQE(phi=phi, ops=ops)
        from .morsr import MORSR
        from .ledger import Ledger
        from .provenance import Provenance

        ledger = Ledger(path_dir=out_dir)
        prov = Provenance(manifold_id=manifold_id)
        runner = MORSR(phi=phi, ops=ops, ledger=ledger, provenance=prov)

        # initialize state
        state = init_state
        state.phi_total = phi.total(state.lanes, state.parity_ok)

        final_state, integrity = runner.run(state, budget=budget)
        return final_state, integrity
