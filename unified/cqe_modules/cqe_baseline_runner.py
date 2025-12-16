#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CQE Baseline Runner (v0.1)

Orchestrates the baseline work order:
  S1: Stage-1 layout
  S2: Conceptual simulations & futures
  S3: Settings, diagonals, 24-plane & lanes

This runner does NOT fetch or move real tokens; it wires the receipts-first steps.
"""

import argparse, json, os, subprocess, sys, datetime

def exists(p): return os.path.exists(p)

def info(msg): print("[INFO]", msg)
def warn(msg): print("[WARN]", msg)
def die(msg):  print("[ERR ]", msg); sys.exit(1)

def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to cqe_baseline_manifest.json")
    ap.add_argument("--outdir", default="runs/baseline")
    ap.add_argument("--dry", action="store_true", help="Print plan only")
    args = ap.parse_args()

    m = load_manifest(args.manifest)
    os.makedirs(args.outdir, exist_ok=True)

    # S1 check
    s1 = m["paths"]["stage1_template"]
    if not exists(s1):
        warn("Stage-1 template not found; create with your Step-1 script or copy from prior session.")
    else:
        info(f"Stage-1 template OK: {s1}")

    # S2: futures
    s2_runner = m["paths"]["step2_runner"]
    s2_futures = m["paths"]["step2_futures"]
    s2_questions = m["paths"]["step2_questions"]
    s2_seed = m["paths"]["stage2_seed"]

    if exists(s1) and exists(s2_runner) and (not exists(s2_futures) or not exists(s2_questions) or not exists(s2_seed)):
        cmd = [sys.executable, s2_runner, "--in", s1, "--outdir", args.outdir]
        info("Plan: run Step-2 simulations & futures: " + " ".join(cmd))
        if not args.dry:
            subprocess.run(cmd, check=False)

    # S3: scaffold + lanes
    step3_scaffold = m["paths"]["step3_scaffold"]
    step3_lanes = m["paths"]["step3_lanes"]
    step3_builder = os.path.join(os.path.dirname(s2_runner), "cqe_step3.py")
    step2_fut = s2_futures if exists(s2_futures) else os.path.join(args.outdir, "cqe_step2_futures.json")

    if exists(step3_builder) and not exists(step3_scaffold):
        cmd = [sys.executable, step3_builder, "--futures", step2_fut, "--outdir", args.outdir]
        info("Plan: build Step-3 scaffold: " + " ".join(cmd))
        if not args.dry:
            subprocess.run(cmd, check=False)

    # Lanes file check (produced earlier in session; regen manual if needed)
    if not exists(step3_lanes):
        warn("Step-3 lanes JSON absent; regenerate via session tool or extend cqe_step3.py.")

    info("Baseline plan complete. Check outputs in '{}'.".format(args.outdir))

if __name__ == "__main__":
    main()
