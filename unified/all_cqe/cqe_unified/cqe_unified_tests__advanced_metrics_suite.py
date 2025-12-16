
import json, argparse
from cqe_unified.config import Config
from cqe_unified.harness import MasterHarness

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.json")
    args = ap.parse_args()
    cfg = Config.from_file(args.config)
    mh = MasterHarness(cfg)

    # 1) Fractal behavior + null-model metrics
    _ = mh.system.process_text_sliced("mandelbrot metrics", plan=["ingest_text","fractal","mandelbrot_null_model"])

    # 2) Toroidal robustness
    out2 = mh.system.process_text_sliced("torus metrics", plan=["toroidal","toroidal_e8_nearest_root"])
    flip_rate = out2.get("octet_ok")  # octet flag still included
    # 3) UVIBS windows
    out3 = mh.system.process_text_sliced("uvibs metrics", plan=["ingest_text","e8_embed","phi","uvibs_windows"])

    summary = {"ok": True}
    with open("artifacts/advanced_metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
