"""AssemblyLine Monitor – validates atomic boundaries & entropy futures.

• register_boundary_check(callback) – callback returns dict matching BoundaryValidationSchema
• register_entropy_check(callback) – callback returns dict matching EntropyFuturesSchema
• run_cycle() – executes all checks, logs events, feeds ValidationManager
• Auto-scheduled by background thread every INTERVAL seconds.
"""
from __future__ import annotations
import threading, time, json, pathlib, datetime
from typing import Callable, Dict, List, Any
import jsonschema
from .dual_governance import default as gov
from .validation import validator

SCHEMA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / 'schemas'
BOUNDARY_SCHEMA = json.loads((SCHEMA_DIR / 'BoundaryValidationSchema.json').read_text())
ENTROPY_SCHEMA = json.loads((SCHEMA_DIR / 'EntropyFuturesSchema.json').read_text())

class AssemblyLine:
    INTERVAL = 5.0  # seconds

    def __init__(self):
        self.boundary_checks: List[Callable[[], Dict[str, Any]]] = []
        self.entropy_checks: List[Callable[[], Dict[str, Any]]] = []
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def register_boundary_check(self, fn: Callable[[], Dict[str, Any]]):
        self.boundary_checks.append(fn)

    def register_entropy_check(self, fn: Callable[[], Dict[str, Any]]):
        self.entropy_checks.append(fn)

    def _loop(self):
        while True:
            self.run_cycle()
            time.sleep(self.INTERVAL)

    def run_cycle(self):
        ts = datetime.datetime.utcnow().isoformat()
        for fn in self.boundary_checks:
            data = fn()
            data['timestamp'] = ts
            jsonschema.validate(data, BOUNDARY_SCHEMA)
            gov.record_event('assembly_boundary', data)
            validator.register(f"boundary_{data['structure_id']}_{ts}", lambda: data['confinement_ok'] and data['chemical_specificity_ok'] and data['informational_boundary_ok'])
        for fn in self.entropy_checks:
            data = fn()
            data['timestamp'] = ts
            jsonschema.validate(data, ENTROPY_SCHEMA)
            gov.record_event('assembly_entropy', data)
            validator.register(f"entropy_{data['segment_id']}_{ts}", lambda: data['delta_S'] <= 0 and data['reversible'])

# singleton
default = AssemblyLine()

__all__ = ['default', 'AssemblyLine']
