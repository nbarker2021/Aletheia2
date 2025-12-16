
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
import os, json, time, statistics, hashlib

from cqe.core import CQEState, CQE

def sha256_json(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()

@dataclass
class Orchestrator:
    out_dir: str

    def run_portfolio(self, hypothesis: Dict[str, Any], budget: int = 200) -> Dict[str, Any]:
        os.makedirs(self.out_dir, exist_ok=True)
        manifolds = hypothesis.get("manifolds", [])
        portfolio = []
        for m in manifolds:
            mid = m["id"]
            seed = m.get("seed", 0)
            weights = m.get("weights", {})
            init = m.get("init_state", {"lanes": [0.0]*8, "parity_ok": True, "tags": {}})
            init_state = CQEState.from_dict(init)

            final_state, integrity = CQE.run_manifold(
                manifold_id=mid,
                seed=seed,
                weights=weights,
                init_state=init_state,
                out_dir=self.out_dir,
                budget=budget
            )

            summary = {
                "manifold_id": mid,
                "seed": seed,
                "weights": weights,
                "final_state": final_state.to_dict(),
                "integrity": integrity.to_dict(),
            }
            portfolio.append(summary)

        # Runbook rollup
        runbook = {
            "ts": time.time(),
            "hypothesis": hypothesis.get("name", "unnamed"),
            "portfolio_count": len(portfolio),
            "results": portfolio,
        }
        with open(os.path.join(self.out_dir, "runbook.json"), "w", encoding="utf-8") as f:
            json.dump(runbook, f, ensure_ascii=False, indent=2)

        # Ops integrity panel
        accepts = sum(r["integrity"]["accepts"] for r in portfolio)
        rejects = sum(r["integrity"]["rejects"] for r in portfolio)
        plateaus = sum((r["integrity"]["plateau_ticks"] or 0) for r in portfolio)
        ops_integrity = {
            "ts": time.time(),
            "accepts": accepts,
            "rejects": rejects,
            "plateau_ticks_total": plateaus,
            "manifolds": [r["manifold_id"] for r in portfolio],
        }
        with open(os.path.join(self.out_dir, "ops_integrity.json"), "w", encoding="utf-8") as f:
            json.dump(ops_integrity, f, ensure_ascii=False, indent=2)

        # Findings — promote those with parity OK and lowest phi
        ranked = sorted(portfolio, key=lambda r: (0 if r["final_state"]["parity_ok"] else 1,
                                                  r["final_state"]["phi_total"]))
        findings = {
            "ts": time.time(),
            "topline": ranked[:3],
            "promotion_rule": "Parity OK first; then minimal Φ",
        }
        with open(os.path.join(self.out_dir, "findings.json"), "w", encoding="utf-8") as f:
            json.dump(findings, f, ensure_ascii=False, indent=2)

        return {
            "runbook": runbook,
            "ops_integrity": ops_integrity,
            "findings": findings,
        }
