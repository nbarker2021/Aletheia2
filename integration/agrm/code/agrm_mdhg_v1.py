"""AGRM+MDHG Harness Builder – satisfies AGRM_MDHGProto.

• bootstrap(governance) registers Governance handle for ΔΦ checks
• build_harness(slice_spec) validates against HarnessSliceSchema and returns
  a lightweight Harness object with ledger-aware execute().

This opens the entire slice-building toolkit: AGRM search + MDHG promotion.
"""
from __future__ import annotations
import json, pathlib, uuid, datetime
from typing import Any, Dict
import jsonschema

from .dual_governance import DualGovernanceBridge
from .governance import GovernanceProto

SCHEMA_PATH = pathlib.Path(__file__).resolve().parent.parent.parent / 'schemas' / 'HarnessSliceSchema.json'
HARNESS_SCHEMA = json.loads(SCHEMA_PATH.read_text())

class Harness:
    def __init__(self, spec: Dict[str, Any], gov: GovernanceProto):
        self.id = str(uuid.uuid4())
        self.spec = spec
        self.gov = gov
        self.gov.record_event('harness_created', {'id': self.id, 'spec': spec})

    def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder execution – logs start/end, returns dummy stats."""
        self.gov.record_event('harness_start', {'harness': self.id, 'plan': plan})
        # ... real AGRM + MDHG algorithms would run here ...
        result = {'status': 'ok', 'energy_used': 0.0, 'timestamp': datetime.datetime.utcnow().isoformat()}
        self.gov.record_event('harness_end', {'harness': self.id, 'result': result})
        return result

class HarnessBuilder:
    def __init__(self):
        self.gov: GovernanceProto | None = None

    # AGRM_MDHGProto
    def bootstrap(self, governance: GovernanceProto) -> None:
        self.gov = governance

    def build_harness(self, slice_spec: Dict[str, Any]) -> Harness:
        if self.gov is None:
            raise RuntimeError('HarnessBuilder.bootstrap() must be called first')
        jsonschema.validate(slice_spec, HARNESS_SCHEMA)
        return Harness(slice_spec, self.gov)

# default instance
default = HarnessBuilder()

__all__ = ['HarnessBuilder', 'default', 'Harness']
