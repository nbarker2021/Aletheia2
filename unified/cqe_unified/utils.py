
import os, json, time, hashlib, random, math, uuid, datetime
from typing import Any, Dict

def now_iso():
    return datetime.datetime.utcnow().isoformat()+"Z"

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def make_run_id(prefix="run"):
    return f"{prefix}_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)

def jdump(path: str, obj: Any):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def jload(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)
