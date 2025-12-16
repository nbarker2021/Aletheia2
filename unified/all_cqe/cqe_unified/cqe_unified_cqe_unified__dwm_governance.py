
from typing import Dict, Any, List
from .dwm_support import seal_envelope

class WitnessRegistry:
    def __init__(self):
        self.rules = []  # callables: (receipts, ctx) -> List[witness]
    def register(self, fn):
        self.rules.append(fn)
    def gather(self, receipts, ctx):
        W = []
        for fn in self.rules:
            try:
                W.extend(fn(receipts, ctx) or [])
            except Exception:
                pass
        return W

def quorum_majority(witnesses: List[Dict[str,Any]], k: int=1) -> bool:
    return len(witnesses) >= max(k, 1)

def quorum_threshold(witnesses: List[Dict[str,Any]], k: int) -> bool:
    return len(witnesses) >= k

def quorum_strict_seal(seal_ok: bool) -> bool:
    return bool(seal_ok)

def seal_batch(system, receipts: List[Dict[str,Any]], config: Dict[str,Any], witnesses: List[Dict[str,Any]]):
    return seal_envelope(system.run_id, receipts, config, witnesses, system.cfg.artifacts_dir)
