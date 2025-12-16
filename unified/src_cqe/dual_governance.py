"""
Dual Governance Bridge – ensures ChatGPT-native governance (local safety / policy
oracle) always communicates with system-level GovernanceLedger. Implements
GovernanceProto so the spine can swap it in transparently.

Design:
┌──────────────────────────┐
│ CQESpine                 │
│  record_event(...)       │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ DualGovernanceBridge     │
│  • system_gov (Ledger)   │→ ledger.jsonl
│  • native_gov (Proxy)    │→ native_ledger.jsonl
└──────────────────────────┘

The bridge fans out every call to both children, ensuring side‑car proof
from the CQE ΔΦ ledger is available when ChatGPT’s own governance checks
fire, and vice‑versa. A task ID is derived from the system ledger’s UUID
but mapped back to the native ID for auditing (bi‑directional lookup).
"""
from __future__ import annotations
import json, pathlib, time, uuid, hashlib, threading
from typing import Dict, Any, Optional

# Import interfaces
from .governance import GovernanceLedger, GovernanceProto

_BRIDGE_LOCK = threading.Lock()

class NativeGovernanceProxy(GovernanceProto):
    """
    Minimal stub emulating ChatGPT‑side governance receipts.
    Stores JSONL next to ledger as `native_ledger.jsonl`.
    """
    def __init__(self, filepath: Optional[str] = None):
        self.filepath = pathlib.Path(filepath or 'native_ledger.jsonl')
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash: Optional[str] = None
        self._load()

    def _load(self):
        if not self.filepath.exists():
            self._last_hash = None
            return
        with self.filepath.open('r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
            self._last_hash = entry.get('hash') if 'entry' in locals() else None

    # GovernanceProto impl
    def init(self, *, cold_boot: bool = False) -> None:
        if cold_boot and self.filepath.exists():
            self.filepath.unlink()
        self._load()

    def record_event(self, event: str, payload: Dict[str, Any]) -> str:
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
        # Basic hash chain validation, ΔΦ ≡ 0
        prev = ''
        for line in self.filepath.open('r', encoding='utf-8'):
            entry = json.loads(line)
            if entry['prev_hash'] != prev:
                return False
            e = entry.copy(); e.pop('hash')
            h = hashlib.sha256(json.dumps(e, separators=(',', ':'), sort_keys=True).encode()).hexdigest()
            if h != entry['hash']:
                return False
            prev = entry['hash']
        return True

class DualGovernanceBridge(GovernanceProto):
    def __init__(self,
                 system_gov: Optional[GovernanceProto] = None,
                 native_gov: Optional[GovernanceProto] = None):
        self.system = system_gov or GovernanceLedger()
        self.native = native_gov or NativeGovernanceProxy()
        self._map: Dict[str, str] = {}  # system_id → native_id

    # GovernanceProto
    def init(self, *, cold_boot: bool = False) -> None:
        self.system.init(cold_boot=cold_boot)
        self.native.init(cold_boot=cold_boot)

    def record_event(self, event: str, payload: Dict[str, Any]) -> str:
        with _BRIDGE_LOCK:
            sys_id = self.system.record_event(event, payload)
            nat_id = self.native.record_event(event, payload)
            self._map[sys_id] = nat_id
            return sys_id  # Spine uses system UUID

    def validate(self) -> bool:
        return self.system.validate() and self.native.validate()

    # Convenience accessor
    def get_native_id(self, system_id: str) -> Optional[str]:
        return self._map.get(system_id)

# Default instance for dynamic import
default = DualGovernanceBridge()

__all__ = ['DualGovernanceBridge', 'default']
