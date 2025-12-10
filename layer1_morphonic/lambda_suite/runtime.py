
from typing import Any, Dict, Tuple, Optional
from . import ast as A
from .eval import eval_normal
import json, hashlib

try:
    # Prefer unified build sidecar if installed
    from morphonic_cqe_unified.sidecar.speedlight_sidecar_plus import SpeedLightPlus as SpeedLight
except Exception:
    try:
        # Fallback to standalone file if user placed it
        from speedlight_sidecar_plus import SpeedLightPlus as SpeedLight  # type: ignore
    except Exception:
        SpeedLight = None  # type: ignore

def _hash_payload(payload: Dict[str, Any]) -> str:
    js = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(js.encode("utf-8")).hexdigest()

def eval_with_sidecar(term: Any, scope: str="lambda", channel: int=3, cache: Optional[Any]=None):
    payload = {"kind":"lambda_eval","term":repr(term)}
    if SpeedLight is None:
        # Direct eval if sidecar not present
        res, n = eval_normal(term)
        return {"result": res, "steps": n, "cached": False}, 0.0, _hash_payload(payload)
    sl = cache or SpeedLight(disk_dir=".speedlight-lambda/cache", ledger_path=".speedlight-lambda/ledger.jsonl")
    def compute():
        res, n = eval_normal(term)
        return {"result": res, "steps": n}
    return sl.compute(payload, scope=scope, channel=channel, compute_fn=compute)
