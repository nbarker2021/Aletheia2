import json, hashlib, time
from typing import Any, Dict

def sha256_json(obj: Dict[str, Any]) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()

def now_receipt(obj: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(obj); out["ts"]=float(time.time()); out["etag"]=sha256_json(out); return out
