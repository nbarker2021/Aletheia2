
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List
import json, hashlib, time

def _sha256(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()

@dataclass
class CQEState:
    # 8-lane Cartan proxy vector and some meta
    lanes: List[float] = field(default_factory=lambda: [0.0]*8)
    # parity vector (bool-ish); True means even/ok, False means violation
    parity_ok: bool = True
    # free-form tags; can hold adapter/domain info
    tags: Dict[str, Any] = field(default_factory=dict)
    # cumulative score (lower is better)
    phi_total: float = 0.0
    # steps counter
    steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lanes": list(self.lanes),
            "parity_ok": bool(self.parity_ok),
            "tags": self.tags,
            "phi_total": float(self.phi_total),
            "steps": int(self.steps),
        }

    def hash(self) -> str:
        payload = {
            "lanes": [round(x, 12) for x in self.lanes],
            "parity_ok": self.parity_ok,
            "tags": self.tags,
        }
        return _sha256(payload)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CQEState":
        return CQEState(
            lanes=list(d.get("lanes", [0.0]*8)),
            parity_ok=bool(d.get("parity_ok", True)),
            tags=dict(d.get("tags", {})),
            phi_total=float(d.get("phi_total", 0.0)),
            steps=int(d.get("steps", 0)),
        )
