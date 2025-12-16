#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cqe_s5_planner.py
Reads a list of tokens + edge targets and emits a web-search plan (queries JSON) and a placement todo list.
This tool does not fetch the web; use web.run in your environment to execute the plan.
"""
import argparse, json, sys, datetime

def now():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True, help="JSON file: {tokens:[...]}")
    ap.add_argument("--edges", required=True, help="JSON file: {edges:[{edge_type, setting, bucket_ids:[]}, ...]}")
    ap.add_argument("--out_queries", default="cqe_web_queries_plan.json")
    ap.add_argument("--out_todo", default="cqe_web_todo.json")
    args = ap.parse_args()

    with open(args.tokens, "r", encoding="utf-8") as f:
        tokens = json.load(f)["tokens"]
    with open(args.edges, "r", encoding="utf-8") as f:
        edges = json.load(f)["edges"]

    queries = {"created_at": now(), "batches": []}
    todo = {"created_at": now(), "placements": []}

    for t in tokens:
        for e in edges:
            batch = {
                "token": t,
                "edge": e,
                "queries": [
                    {"role": "precision", "q": f"\"{t}\" {e['edge_type']}", "recency_days": 365, "domains": []},
                    {"role": "recall", "q": f"{t} {e['edge_type']} evidence", "recency_days": None, "domains": []},
                    {"role": "paraphrase_alt", "q": f"\"{t} alternative\" OR \"{t} parity\"", "recency_days": 1095, "domains": []},
                    {"role": "counterfactual", "q": f"\"{t}\" -{e['edge_type']} contradiction", "recency_days": None, "domains": []},
                ]
            }
            queries["batches"].append(batch)
            todo["placements"].append({
                "token": t,
                "edge": e,
                "expected_updates": ["core24|fringe8", "even16|odd16", "diagonals", "G1..G4 consensus"]
            })

    with open(args.out_queries, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    with open(args.out_todo, "w", encoding="utf-8") as f:
        json.dump(todo, f, ensure_ascii=False, indent=2)

    print("Wrote:", args.out_queries, args.out_todo)

if __name__ == "__main__":
    main()
