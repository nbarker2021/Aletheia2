
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unify GeoLight and TokLight ledgers for analytics."""
import os, json
from typing import Dict, Any, List, Optional

def read_jsonl(path: str) -> List[Dict[str,Any]]:
    out = []
    if not os.path.exists(path): return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

def load_geolight(ledger_path: str) -> List[Dict[str,Any]]:
    rows = read_jsonl(ledger_path)
    out = []
    for r in rows:
        out.append({
            "ts": r.get("ts"),
            "scope": r.get("scope","geom"),
            "channel": r.get("channel",3),
            "cost": r.get("cost",0.0),
            "input_hash": r.get("input_hash"),
            "result_hash": r.get("result_hash"),
            "receipt": r.get("entry"),
            "prev": r.get("prev"),
            "lane": "GeoLight",
        })
    return out

def load_toklight(ledger_path: str) -> List[Dict[str,Any]]:
    rows = read_jsonl(ledger_path)
    out = []
    for r in rows:
        out.append({
            "ts": r.get("ts"),
            "scope": r.get("scope","tokenizer"),
            "op": r.get("op"),
            "cost": r.get("cost",0.0),
            "input_hash": r.get("input_hash"),
            "result_hash": r.get("result_hash"),
            "receipt": r.get("entry"),
            "prev": r.get("prev"),
            "lane": "TokLight",
        })
    return out

def merge_timelines(*timelines: List[List[Dict[str,Any]]]) -> List[Dict[str,Any]]:
    merged = []
    for tl in timelines:
        merged.extend(tl)
    merged.sort(key=lambda r: (r.get("ts",0), r.get("lane","")))
    return merged
