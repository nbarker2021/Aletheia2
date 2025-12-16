"""
CQE Harness CLI — runs the harness and prints the ledger path and receipt.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from cqe.harness.core import run_harness

def main() -> None:
    parser = argparse.ArgumentParser(description="Run CQE harness on text input")
    parser.add_argument("--text", required=True, help="Input text")
    parser.add_argument("--out", default="runs/demo", help="Output directory")
    args = parser.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    result = run_harness(args.text, outdir)
    print(f"✅ Receipt written → {outdir / 'ledger.jsonl'}")
    print(result)

if __name__ == "__main__":
    main()
