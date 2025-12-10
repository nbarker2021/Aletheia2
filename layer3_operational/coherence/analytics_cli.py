
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coherence/Decoherence Analytics CLI"""
import json, argparse, sys
from typing import List, Dict, Any
from coherence_metrics import composite_coherence, collapse_detector, embedding_alignment
from state_store import StateStore
from receipts_bridge import load_geolight, load_toklight, merge_timelines

def load_points(path: str):
    return json.load(open(path, "r", encoding="utf-8"))

def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    c = sub.add_parser("coherence"); c.add_argument("--points-json", required=True)
    d = sub.add_parser("collapse"); d.add_argument("--prev-json", required=True); d.add_argument("--curr-json", required=True)
    a = sub.add_parser("align"); a.add_argument("--emb-a-json", required=True); a.add_argument("--emb-b-json", required=True)
    t = sub.add_parser("timeline"); t.add_argument("--store", default="./deco_states"); t.add_argument("--receipts-json", required=True)

    l = sub.add_parser("ledger"); l.add_argument("--geo-ledger"); l.add_argument("--tok-ledger")

    args = p.parse_args(argv)

    if args.cmd == "coherence":
        pts = load_points(args.points_json)
        print(json.dumps(composite_coherence(pts), indent=2)); return

    if args.cmd == "collapse":
        A = load_points(args.prev_json); B = load_points(args.curr_json)
        print(json.dumps(collapse_detector(A, B), indent=2)); return

    if args.cmd == "align":
        A = json.load(open(args.emb_a_json)); B = json.load(open(args.emb_b_json))
        print(json.dumps({"alignment": embedding_alignment(A,B)}, indent=2)); return

    if args.cmd == "timeline":
        store = StateStore(args.store)
        recs = json.load(open(args.receipts_json))
        import callbacks as _cb
        print(json.dumps(_cb.timeline_metrics(store, recs), indent=2)); return

    if args.cmd == "ledger":
        tl = []
        if args.geo_ledger: tl += load_geolight(args.geo_ledger)
        if args.tok_ledger: tl += load_toklight(args.tok_ledger)
        print(json.dumps(merge_timelines(tl), indent=2)); return

    p.print_help()

if __name__ == "__main__":
    main()
