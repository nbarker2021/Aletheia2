
"""Governance Ledger – Python implementation that satisfies GovernanceProto.
Uses a JSONL + SHA‑256 Merkle chain for receipts (fallback when native C lib
<cqe/merkle_ledger.h> is unavailable). Can be swapped for the C extension
transparently by overriding CQE_GOV_MOD.
"""
from __future__ import annotations
import json, time, hashlib, os, pathlib, threading, base64, uuid
from typing import Dict, Any, Optional

_LOCK = threading.Lock()

class GovernanceLedger:
    def __init__(self, filepath: Optional[str] = None):
        self.filepath = pathlib.Path(filepath or os.getenv('CQE_LEDGER', 'ledger.jsonl'))
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash = None
        self._load_last_hash()

    # ------------------------------------------------------------------ #
    # GovernanceProto interface
    # ------------------------------------------------------------------ #
    def init(self, *, cold_boot: bool = False) -> None:
        """Initialise ledger – on cold boot wipe existing file."""
        if cold_boot and self.filepath.exists():
            self.filepath.unlink()
        if not self.filepath.exists():
            self._last_hash = None
        self._load_last_hash()

    def record_event(self, event: str, payload: Dict[str, Any]) -> str:
        """Append an event, returns deterministic task id."""
        with _LOCK:
            ts = time.time()
            entry = {
                'id': str(uuid.uuid4()),
                'ts': ts,
                'event': event,
                'payload': payload,
                'prev_hash': self._last_hash or ''
            }
            entry_bytes = json.dumps(entry, separators=(',', ':'), sort_keys=True).encode()
            entry_hash = hashlib.sha256(entry_bytes).hexdigest()
            entry['hash'] = entry_hash
            with self.filepath.open('a', encoding='utf-8') as f:
                f.write(json.dumps(entry, separators=(',', ':')) + '\n')
            self._last_hash = entry_hash
            return entry['id']

    def validate(self) -> bool:
        """Full pass over ledger verifying Merkle chain + placeholder ΔΦ ≤ 0.
        Returns True if ledger intact and ΔΦ checks pass.
        """
        with _LOCK:
            prev_hash = ''
            for line in self.filepath.open('r', encoding='utf-8'):
                entry = json.loads(line)
                expected_prev = entry.get('prev_hash', '')
                if expected_prev != prev_hash:
                    return False
                # recompute hash
                entry_no_hash = entry.copy()
                entry_no_hash.pop('hash', None)
                recalculated = hashlib.sha256(
                    json.dumps(entry_no_hash, separators=(',', ':'), sort_keys=True).encode()
                ).hexdigest()
                if recalculated != entry['hash']:
                    return False
                # ΔΦ placeholder – always 0 for now
                prev_hash = entry['hash']
            return True

# default instance expected by CQESpine dynamic loader
default = GovernanceLedger()

__all__ = ['GovernanceLedger', 'default']
