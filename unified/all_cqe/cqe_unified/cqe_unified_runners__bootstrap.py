
import argparse, os, json
from cqe_unified.config import Config
from cqe_unified.harness import MasterHarness
from cqe_unified.utils import jload

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--text", action="append", help="Text to process (can be repeated)")
    args = ap.parse_args()

    cfg = Config.from_file(args.config)
    if args.seed is not None:
        cfg.seed = args.seed

    mh = MasterHarness(cfg)
    texts = args.text or ["hello world", "geometry-first receipts", "CQE unified skeleton"]
    results = mh.run_on_texts(texts)

    # Persist a summary JSON to artifacts
    artifacts_dir = cfg.artifacts_dir
    os.makedirs(artifacts_dir, exist_ok=True)
    summary_path = os.path.join(artifacts_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print("Wrote:", summary_path)

if __name__ == "__main__":
    main()
