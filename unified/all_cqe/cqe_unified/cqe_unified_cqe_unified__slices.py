
from typing import Callable, Dict, Any

SLICE_REGISTRY: Dict[str, Callable] = {}

def slice_op(name: str, category:str="GEN", lane:int=0, parity:str="E", requires=None, provides=None):
    requires = requires or []
    provides = provides or []
    def deco(fn):
        SLICE_REGISTRY[name] = fn
        fn.__meta__ = {"name":name,"category":category,"lane":lane,"parity":parity,"requires":requires,"provides":provides}
        return fn
    return deco

# Minimal charter slice to init overlay/oracle and artifacts dir on system.cfg
@slice_op("dwm_charter_init", category="DWM", lane=0, parity="E", requires=[], provides=["dwm_charter"])
def dwm_charter_init(system, payload, state):
    import os, time
    if not hasattr(system.cfg, "artifacts_dir"):
        class CfgShim: pass
        system.cfg.artifacts_dir = str((__import__("pathlib").Path("/mnt/data/cqe_unified/artifacts")))
        system.cfg.seed = 42
    # init registries if absent
    if not hasattr(system, "_overlay_registry") or not system._overlay_registry:
        system._overlay_registry = {
            "A": {"glyph":"A","category":"TRIANGLE","ops":["×","÷","#2"],"status":"PENDING","evidence":[]},
            "B": {"glyph":"B","category":"MULTI_STROKE","ops":["%","÷","~"],"status":"PENDING","evidence":[]},
            "C": {"glyph":"C","category":"LOOP","ops":["÷","~"],"status":"PENDING","evidence":[]}
        }
    if not hasattr(system, "_hyperperm_oracle") or not system._hyperperm_oracle:
        system._hyperperm_oracle = {"items":{}}
    state["dwm_charter"] = {"started_at": time.time()}
    system.emit("charter", {"overlays": len(system._overlay_registry)})
    return {"dwm_charter":"ok"}
