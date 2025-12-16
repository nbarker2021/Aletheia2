from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from cqe_core.validation import validator
from cqe_core.assemblyline import default as assemblyline

from typing import Any, Dict, Optional, Protocol, runtime_checkable

logger = logging.getLogger("cqe.spine")

@runtime_checkable
class GovernanceProto(Protocol):
    def init(self, *, cold_boot: bool = False) -> None: ...
    def record_event(self, event: str, payload: Dict[str, Any]) -> str: ...
    def validate(self) -> bool: ...

@runtime_checkable
class AGRM_MDHGProto(Protocol):
    def bootstrap(self, governance: GovernanceProto) -> None: ...
    def build_harness(self, slice_spec: Dict[str, Any]) -> Any: ...

@runtime_checkable
class AletheiaProto(Protocol):
    def plan(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]: ...

@runtime_checkable
class WorldForgeProto(Protocol):
    def spawn_world(self, spec: Dict[str, Any]) -> Any: ...

@runtime_checkable
class Scene8Proto(Protocol):
    def render(self, world, output: Path, codec: str = "uesc") -> Path: ...

@dataclass
class SpineConfig:
    governance_mod: str = "cqe_core.governance"
    agrm_mod: str = "cqe_core.agrm_mdhg"
    aletheia_mod: str = "cqe_core.thinktank_planner"
    worldforge_mod: str = "worldforge"
    scene8_mod: str = "scene8_gvs"
    lazy: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

def _import(path: str):
    return importlib.import_module(path)

def _get(mod, attr: str):
    return getattr(mod, attr)

class CQESpine:
    def __init__(self, cfg: Optional[SpineConfig] = None):
        self.cfg = cfg or SpineConfig()
        self._load()

    def _load(self):
        self.governance = _get(_import(self.cfg.governance_mod), "default")
        self.agrm = _get(_import(self.cfg.agrm_mod), "default")
        self.aletheia = _get(_import(self.cfg.aletheia_mod), "default")
        self.worldforge = _get(_import(self.cfg.worldforge_mod), "default")
        self.scene8 = _get(_import(self.cfg.scene8_mod), "default")
        self.governance.init(cold_boot=True)

    def dispatch_problem(self, desc: str, output: Optional[Path] = None, codec: str = "uesc") -> Path:
        tid = self.governance.record_event("problem_received", {"desc": desc})
        harness = self.agrm.build_harness({})
        plan = self.aletheia.plan(desc, {"harness": harness})
        world = self.worldforge.spawn_world(plan.get("world_spec", {}))
        out = output or Path.cwd() / f"solve_{tid}.mp4"
        return self.scene8.render(world, out, codec=codec)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    spine = CQESpine()
    result = spine.dispatch_problem("apple falling from tree")
    print(result)
