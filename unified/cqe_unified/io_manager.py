
from typing import Any, Dict, Iterable
import json, os
from .utils import ensure_dir, sha256_bytes

class IOManager:
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = artifacts_dir
        ensure_dir(self.artifacts_dir)

    def ingest_text(self, text: str) -> Dict[str, Any]:
        b = text.encode("utf-8")
        return {"digest": sha256_bytes(b), "bytes": len(b), "type": "text"}

    def export_json(self, name: str, obj: Any) -> str:
        path = os.path.join(self.artifacts_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        return path
