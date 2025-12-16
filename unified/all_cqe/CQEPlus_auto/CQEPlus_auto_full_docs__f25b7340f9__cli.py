import argparse, json, os
from pathlib import Path
from .worldforge.compose import compose
from .delta import run as delta_run
from .braid import run as braid_run
from .mdhg import MDHG
from .agrm import run as agrm_run

def main():
    ap = argparse.ArgumentParser("cqe")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("compose")
    sp.add_argument("--prompt", required=True)
    sp.add_argument("--modes", default="image")
    sp.add_argument("--lenses", default="digit:24,glyph:E8,topo:torus")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--out", default="runs")

    sd = sub.add_parser("delta")
    sd.add_argument("--root", default="/mnt/data")
    sd.add_argument("--max", type=int, default=128)
    sd.add_argument("--out", default="runs")

    sb = sub.add_parser("braid")
    sb.add_argument("--root", default="/mnt/data")
    sb.add_argument("--max", type=int, default=256)
    sb.add_argument("--out", default="runs")

    sm = sub.add_parser("mdhg-promote")
    sm.add_argument("--run", required=True)
    sm.add_argument("--top", type=int, default=16)

    sa = sub.add_parser("agrm")
    sa.add_argument("--problem", required=True)
    sa.add_argument("--budget", type=int, default=200)
    sa.add_argument("--seed", type=int, default=42)
    sa.add_argument("--out", default="runs")

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.cmd == "compose":
        modes=[m.strip() for m in args.modes.split(",") if m.strip()]
        lenses=[l.strip() for l in args.lenses.split(",") if l.strip()]
        rec = compose(args.seed, args.prompt, modes, lenses)
        Path(args.out,"compose_receipt.json").write_text(json.dumps(rec, indent=2))
        print(str(Path(args.out,"compose_receipt.json")))
    elif args.cmd == "delta":
        out = delta_run(args.root, args.max, args.out)
        Path(args.out,"delta_receipt.json").write_text(json.dumps(out["receipt"], indent=2))
        print(str(Path(args.out,"delta_ranking.csv")))
    elif args.cmd == "braid":
        out = braid_run(args.root, args.max, args.out)
        Path(args.out,"braid_receipt.json").write_text(json.dumps(out, indent=2))
        print(str(Path(args.out,"braid_receipt.json")))
    elif args.cmd == "mdhg-promote":
        import csv
        ranking_csv = Path(args.run)/"field_ranking.csv"
        if not ranking_csv.exists():
            ranking_csv = Path(args.run)/"delta_ranking.csv"
        rows=[]
        with open(ranking_csv, newline="") as f:
            for r in csv.DictReader(f): rows.append(r)
        rows = rows[:args.top]
        mdhg = MDHG(buckets=256)
        for r in rows:
            item={"doc_id": r.get("doc_id"), "path": r.get("path"), "rank": int(r.get("rank",0))}
            mdhg.put(item)
        man = mdhg.manifest()
        Path(args.run,"mdhg_promotion.json").write_text(json.dumps({"promoted": len(rows), "manifest": man}, indent=2))
        print(str(Path(args.run,"mdhg_promotion.json")))
    elif args.cmd == "agrm":
        prob = json.loads(Path(args.problem).read_text())
        res = agrm_run(prob, budget=args.budget, seed=args.seed)
        Path(args.out,"agrm_receipt.json").write_text(json.dumps(res["summary"], indent=2))
        print(str(Path(args.out,"agrm_receipt.json")))
