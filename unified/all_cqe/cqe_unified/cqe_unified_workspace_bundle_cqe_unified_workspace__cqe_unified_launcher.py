# Auto-generated meta launcher
import importlib.util, sys, os, subprocess, json
from pathlib import Path

WORKSPACE = Path(__file__).parent
CANDIDATES = [
  "cqe_unified_snapshot/cqe_unified/cqe_unified/harness.py",
  "cqe_unified_snapshot/cqe_unified/cqe_unified/ms4_runner.py",
  "cqe_unified_snapshot/cqe_unified/tests/test_harness_spec_runner.py",
  "cqe_unified_repo_ms3_ms4/cqe_unified/cqe_unified/harness.py",
  "cqe_unified_repo_ms3_ms4/cqe_unified/cqe_unified/ms4_runner.py",
  "cqe_unified_repo_ms3_ms4/cqe_unified/tests/test_harness_spec_runner.py",
  "cqe_unified_repo_with_ms2/cqe_unified/cqe_unified/harness.py",
  "cqe_unified_repo_with_ms2/cqe_unified/tests/test_harness_spec_runner.py",
  "cqe_unified_repo_with_dwm_triplet/cqe_unified/cqe_unified/harness.py",
  "cqe_unified_repo_with_dwm_triplet/cqe_unified/tests/test_harness_spec_runner.py",
  "cqe_unified_repo_with_uvibs_pose_glyphs/cqe_unified/cqe_unified/harness.py",
  "cqe_unified_repo_with_uvibs_pose_glyphs/cqe_unified/tests/test_harness_spec_runner.py",
  "cqe_unified_repo_e8_full/cqe_unified/cqe_unified/harness.py",
  "cqe_unified_repo_e8_full/cqe_unified/tests/test_harness_spec_runner.py",
  "cqe_unified_repo_plus_docs/cqe_unified/cqe_unified/harness.py",
  "cqe_unified_repo_plus_docs/cqe_unified/tests/test_harness_spec_runner.py",
  "cqe_unified_repo_sliced_pro/cqe_unified/cqe_unified/harness.py",
  "cqe_unified_repo_sliced_adv/cqe_unified/cqe_unified/harness.py",
  "cqe_unified_repo_sliced/cqe_unified/cqe_unified/harness.py"
]

def list_candidates():
    print(json.dumps(CANDIDATES, indent=2))

def run_candidate(relpath, extra_args=None):
    p = WORKSPACE / relpath
    if not p.exists():
        print(f"[ERR] Candidate not found: {relpath}")
        sys.exit(2)
    # Prefer running as a script to preserve original assumptions
    cmd = [sys.executable, str(p)]
    if extra_args:
        cmd += extra_args
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKSPACE) + os.pathsep + env.get("PYTHONPATH","")
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=False)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="CQE Unified Meta Launcher")
    ap.add_argument("--list", action="store_true", help="List discovered harness candidates")
    ap.add_argument("--run", type=str, help="Relative path of candidate to run (from workspace root)")
    ap.add_argument("extra", nargs="*", help="Extra arguments passed to the candidate script")
    args = ap.parse_args()
    if args.list:
        list_candidates()
        return
    if args.run:
        run_candidate(args.run, args.extra)
        return
    # Default: print guidance
    print("CQE Unified Meta Launcher")
    print("Use --list to see harness scripts, then --run <path> [args] to execute one.")
    print("Example: python cqe_unified_launcher.py --list")
    print("         python cqe_unified_launcher.py --run cqe_unified_snapshot/tools/harness.py -- --help")

if __name__ == "__main__":
    main()
