
import threading, json, hashlib
from typing import Any, Dict, Tuple
try:
    from morphonic_cqe_unified.sidecar.speedlight_sidecar_plus import SpeedLightPlus as SpeedLight
except Exception:
    from morphonic_cqe_unified.sidecar.speedlight_sidecar import SpeedLight  # type: ignore

class CQESidecar:
    def __init__(self):
        self._sl = SpeedLight()
        self._meta: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def _hash_payload(self, payload: Any) -> str:
        js = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(js.encode("utf-8")).hexdigest()

    def compute(self, payload: Any, scope: str, channel: int, compute_fn=None, *args, **kwargs) -> Tuple[Any, float, str]:
        with self._lock:
            rid = self._hash_payload({"payload": payload, "scope": scope})
            result, cost, receipt_id = self._sl.compute(payload, scope=scope, channel=channel, compute_fn=compute_fn, **kwargs)
            if cost > 0.0 and receipt_id not in self._meta:
                self._meta[receipt_id] = {"scope": scope, "channel": channel, "note": kwargs.get("note","")}
            return result, cost, receipt_id

    def get_meta(self, receipt_id: str) -> Dict[str, Any]:
        with self._lock:
            meta = dict(self._meta.get(receipt_id, {}))
            if not meta and hasattr(self._sl, "get_meta"):
                meta = self._sl.get_meta(receipt_id) or meta
            return meta

    def report(self) -> str:
        return self._sl.report()
