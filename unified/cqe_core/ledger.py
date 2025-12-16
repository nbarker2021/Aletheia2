
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List
import json, time, hashlib, os

def sha256_json(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()

@dataclass
class Ledger:
    path_dir: str
    file: Any = field(default=None, repr=False, compare=False)
    count: int = 0

    def open(self, manifold_id: str):
        os.makedirs(os.path.join(self.path_dir, manifold_id), exist_ok=True)
        p = os.path.join(self.path_dir, manifold_id, "receipts.jsonl")
        self.file = open(p, "a", encoding="utf-8")

    def append(self, rec: Dict[str, Any]):
        if not self.file:
            raise RuntimeError("Ledger not opened")
        self.file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.file.flush()
        self.count += 1

    def close(self):
        if self.file:
            self.file.close()
            self.file = None
