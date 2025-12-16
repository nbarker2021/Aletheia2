
import subprocess, sys

def main():
    cmds = [
        [sys.executable, "-m", "tests.golden_suite"],
        [sys.executable, "-m", "tests.slices_suite"],
        [sys.executable, "-m", "tests.advanced_slices_suite"],
        [sys.executable, "-m", "tests.advanced_metrics_suite"],
        [sys.executable, "-m", "pytest", "tests/test_harness_spec_runner.py", "-q"],
        [sys.executable, "-m", "pytest", "tests/test_geometry_first_invariants.py", "-q"],
        [sys.executable, "-m", "pytest", "tests/test_persona_checklist.py", "-q"],
        [sys.executable, "-m", "pytest", "tests/test_e8_roots_full.py", "-q"],
        [sys.executable, "-m", "pytest", "tests/test_morsr_slice.py", "-q"],
        [sys.executable, "-m", "pytest", "tests/test_carlson_slice.py", "-q"],
        [sys.executable, "-m", "pytest", "tests/test_gov_recs.py", "-q"],
    ]
    ok = True
    for c in cmds:
        print("==>", " ".join(c))
        rc = subprocess.call(c)
        ok = ok and (rc == 0)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()

import subprocess, sys
subprocess.call([sys.executable,"-m","pytest","tests/test_dwm_triplet.py","-q"])
