
import json, argparse
from cqe_unified.config import Config
from cqe_unified.harness import MasterHarness

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.json")
    args = ap.parse_args()
    cfg = Config.from_file(args.config)
    mh = MasterHarness(cfg)

    # Run sliced mode on a few texts
    texts = ["octet forcing", "parity lanes", "sliced execution"]
    oks = []
    for t in texts:
        out = mh.system.process_text_sliced(t)
        oks.append(out.get("octet_ok"))
    summary = {"texts": texts, "octet_ok_rate": sum(1 for x in oks if x)/len(oks)}
    with open("artifacts/slices_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
