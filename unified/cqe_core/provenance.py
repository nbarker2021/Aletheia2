
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any
import time, json, hashlib

def sha256_json(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()

@dataclass
class Provenance:
    manifold_id: str
    started_ts: float = field(default_factory=time.time)
    run_id: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def start(self):
        base = {
            "manifold": self.manifold_id,
            "started_ts": self.started_ts,
        }
        self.run_id = sha256_json(base)[:12]
        return self.run_id
