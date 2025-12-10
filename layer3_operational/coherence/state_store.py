
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""State snapshots saved by receipt id."""
import os, json, time, uuid
from typing import Dict, Any, List, Optional

class StateStore:
    def __init__(self, root: str="./deco_states"):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def _path(self, rid: str) -> str:
        return os.path.join(self.root, f"{rid}.json")

    def save(self, *, receipt: str, points=None, tokens=None, embedding=None, meta: Dict[str,Any]=None):
        meta = meta or {}
        doc = {
            "receipt": receipt,
            "ts": time.time(),
            "points": points or [],
            "tokens": tokens or [],
            "embedding": embedding or [],
            "meta": meta,
        }
        with open(self._path(receipt), "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)

    def load(self, receipt: str) -> Optional[Dict[str,Any]]:
        p = self._path(receipt)
        if not os.path.exists(p): return None
        return json.load(open(p, "r", encoding="utf-8"))

    def list(self, limit: int=200):
        files = sorted([fn for fn in os.listdir(self.root) if fn.endswith(".json")])[-limit:]
        out = []
        for fn in files:
            try:
                j = json.load(open(os.path.join(self.root, fn), "r", encoding="utf-8"))
                out.append({"receipt": j.get("receipt"), "ts": j.get("ts"), "meta": j.get("meta",{})})
            except Exception:
                pass
        return out
