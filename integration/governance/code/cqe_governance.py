
import json, hashlib
from dataclasses import dataclass, asdict
from typing import Any, Optional, Tuple, List

CRT_PRIMES = [1000003, 1000033, 1000037]
BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
assert len(BASE62) == 62

def to_cnf(obj: Any) -> str:
    def transform(x):
        if isinstance(x, dict):
            return {k: transform(v) for k,v in sorted(x.items())}
        elif isinstance(x, list):
            return [transform(v) for v in x]
        elif isinstance(x, float):
            return float(f"{x:.12f}")
        else:
            return x
    stable = transform(obj)
    return json.dumps(stable, separators=(",", ":"), sort_keys=True)

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _to_base62(n: int) -> str:
    if n == 0:
        return "0"
    out = []
    q = n
    while q > 0:
        out.append(BASE62[q % 62])
        q //= 62
    return "".join(reversed(out))

def crt_signature(data_hex: str) -> str:
    n = int(data_hex, 16)
    residues = [n % p for p in CRT_PRIMES]
    return ".".join(_to_base62(r) for r in residues)

@dataclass
class BoundaryReceipt:
    timestamp: float
    actor: str
    pre_state: tuple
    post_state: tuple
    dphi: float
    channel: int
    scope: str
    note: str = ""
    cnf_hash: Optional[str] = None
    crt_sig: Optional[str] = None

    def to_cnf_hash_and_sign(self) -> Tuple[str, str]:
        data = asdict(self).copy()
        data["cnf_hash"] = None
        data["crt_sig"] = None
        cnf = to_cnf(data)
        h = sha256_hex(cnf.encode("utf-8"))
        sig = crt_signature(h)
        self.cnf_hash = h
        self.crt_sig = sig
        return h, sig

@dataclass
class AuditEntry:
    idx: int
    prev_hash: str
    receipt: BoundaryReceipt
    entry_hash: str

class AuditChain:
    def __init__(self):
        self.entries: List[AuditEntry] = []
        self.tip_hash: str = "0"*64

    def append(self, r: BoundaryReceipt) -> AuditEntry:
        if not r.cnf_hash or not r.crt_sig:
            r.to_cnf_hash_and_sign()
        content = {
            "prev_hash": self.tip_hash,
            "cnf_hash": r.cnf_hash,
            "crt_sig": r.crt_sig,
            "timestamp": r.timestamp,
            "actor": r.actor,
            "channel": r.channel,
            "scope": r.scope,
        }
        h = hashlib.sha256(to_cnf(content).encode("utf-8")).hexdigest()
        entry = AuditEntry(idx=len(self.entries), prev_hash=self.tip_hash, receipt=r, entry_hash=h)
        self.entries.append(entry)
        self.tip_hash = h
        return entry

    def verify(self) -> bool:
        prev = "0"*64
        for e in self.entries:
            if e.prev_hash != prev:
                return False
            if not e.receipt.cnf_hash or not e.receipt.crt_sig:
                return False
            recompute = {
                "prev_hash": e.prev_hash,
                "cnf_hash": e.receipt.cnf_hash,
                "crt_sig": e.receipt.crt_sig,
                "timestamp": e.receipt.timestamp,
                "actor": e.receipt.actor,
                "channel": e.receipt.channel,
                "scope": e.receipt.scope,
            }
            h = hashlib.sha256(to_cnf(recompute).encode("utf-8")).hexdigest()
            if h != e.entry_hash:
                return False
            prev = h
        return True
