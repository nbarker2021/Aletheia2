"""Command-line entrypoint for the CQE harness."""
from __future__ import annotations
import argparse
from pathlib import Path
from cqe.harness.core import run_harness

def main():
    parser = argparse.ArgumentParser(description="CQE Harness CLI (receipts-first)")
    parser.add_argument("--text", type=str, default="demo", help="Input text/prompt")
    parser.add_argument("--out", type=str, default="runs/demo", help="Output directory")
    args = parser.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    receipt = run_harness(args.text, outdir)
    print("Harness complete. Receipt:", receipt)
