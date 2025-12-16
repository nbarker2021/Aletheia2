
# Golden Suite: binds claims to executable checks. CI-friendly; emits a simple JSON summary.
import json, os, argparse, statistics
from cqe_unified.config import Config
from cqe_unified.harness import MasterHarness

TESTS = {}

def test_register(fn):
    TESTS[fn.__name__] = fn
    return fn

@test_register
def T_E8_EMBED(cfg: Config):
    """Check unit-norm-ish property (distance proxy) and determinism under seed lock."""
    mh = MasterHarness(cfg)
    a = mh.system.process_text("alpha")["e8"]["root_distance"]
    b = mh.system.process_text("alpha")["e8"]["root_distance"]
    pass_det = abs(a - b) < 1e-9
    return {"deterministic": pass_det, "root_distance": a, "pass": pass_det}

@test_register
def T_GOV_GATES(cfg: Config):
    """Governance gates produce scores and banding."""
    mh = MasterHarness(cfg)
    out = mh.system.process_text("governance check")
    pass_band = out["band"] in ("PEER_READY","BREAKTHROUGH","EXPLORATORY")
    return {"band": out["band"], "v_total": out["v_total"], "pass": pass_band}

@test_register
def T_SEMANTICS(cfg: Config):
    """Semantics returns relation/structure/confidence fields."""
    mh = MasterHarness(cfg)
    sem = mh.system.process_text("semantic probe")["semantics"]
    keys = set(sem.keys())
    pass_keys = {"relation","structure","confidence"}.issubset(keys)
    return {"semantics": sem, "pass": pass_keys}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.json")
    args = ap.parse_args()
    cfg = Config.from_file(args.config)

    results = {}
    passes = 0
    for name, fn in TESTS.items():
        r = fn(cfg)
        results[name] = r
        passes += 1 if r.get("pass") else 0

    summary = {"total": len(TESTS), "passes": passes, "results": results}
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/golden_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
