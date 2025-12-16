
import os, json
from typing import Dict, Any, Optional
from .utils import ensure_dir, sha256_bytes

class Storage:
    def __init__(self, base_dir: str):
        self.base = base_dir
        ensure_dir(self.base)

    def put_blob(self, namespace: str, data: bytes) -> str:
        ns = os.path.join(self.base, namespace)
        ensure_dir(ns)
        digest = sha256_bytes(data)
        path = os.path.join(ns, digest + ".bin")
        with open(path, "wb") as f:
            f.write(data)
        return digest

    def get_blob(self, namespace: str, digest: str) -> Optional[bytes]:
        path = os.path.join(self.base, namespace, digest + ".bin")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
        return None

    def put_json(self, namespace: str, obj: Any) -> str:
        data = json.dumps(obj, sort_keys=True).encode("utf-8")
        return self.put_blob(namespace, data)
