from __future__ import annotations
import uuid, json
from typing import Dict, Any
from .dtt_orchestrator import default as dtt

class ThinkTankPlanner:
    """Minimal AletheiaProto planner that packages problem into IdeaPacket
    and submits to DTT."""
    def plan(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        packet = {
            'id': f'idea:{uuid.uuid4()}',
            'type': 'sandbox_scenario',
            'content': {'problem': problem},
            'context': context,
            'expected_outputs': {},
            'metadata': {'origin': 'ThinkTankPlanner'}
        }
        dtt.submit(packet)
        return {'world_spec': {'thinktank': True}}
default = ThinkTankPlanner()
__all__ = ['ThinkTankPlanner', 'default']
