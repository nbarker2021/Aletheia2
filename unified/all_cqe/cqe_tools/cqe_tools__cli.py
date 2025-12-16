
from __future__ import annotations
import argparse, json, os
from cqe.harness.orchestrator import Orchestrator

def main():
    ap = argparse.ArgumentParser(description="CQE Unified System CLI")
    ap.add_argument("--hypothesis", required=True, help="Path to hypothesis JSON")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--budget", type=int, default=200, help="Step budget per manifold")
    args = ap.parse_args()

    with open(args.hypothesis, "r", encoding="utf-8") as f:
        hyp = json.load(f)

    orch = Orchestrator(out_dir=args.out)
    res = orch.run_portfolio(hypothesis=hyp, budget=args.budget)

    print("Run complete.")
    print("Runbook:", os.path.join(args.out, "runbook.json"))
    print("Ops Integrity:", os.path.join(args.out, "ops_integrity.json"))
    print("Findings:", os.path.join(args.out, "findings.json"))

if __name__ == "__main__":
    main()
