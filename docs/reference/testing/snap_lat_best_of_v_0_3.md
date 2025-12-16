# SnapLat — Best-of v0.3.7 Canvas

**Timestamp:** 2025-08-17 10:49:25

This canvas reflects the current "best-of" repo state with P0/P1 upgrades, plus tests and tooling. It includes: release snapshot, repo tree, artifacts, how-to-run, current work plan, and **full source code listing** for this repo.

## Downloads

- Latest: `sandbox:/mnt/data/snaplat_bestof_v0_3_7_P1progress.zip`
- Prior packages: `sandbox:/mnt/data/snaplat_bestof_v0_3_6_P1seed.zip`, `sandbox:/mnt/data/snaplat_bestof_v0_3_5_P0b.zip`, `sandbox:/mnt/data/snaplat_bestof_v0_3_4_P0.zip`
- Upstream reference (user): `sandbox:/mnt/data/snaplat_v0.2.8_full.zip`

## Quick Commands

```bash
# Run a fusion+gates tick (with optional RAG)
PYTHONPATH=src python scripts/cli.py tick --sectors 16 --boundary 0.9 --with-rag

# Generate shell tour (m=1) artifact
PYTHONPATH=src python scripts/shell_tour.py

# Emit artifact manifest and a release event
PYTHONPATH=src python tools/manifest.py
PYTHONPATH=src python tools/release_event.py

# Run tests
PYTHONPATH=src python -m pytest -q
```

## Start-of-Session Checklist

1. Verify zip and repo paths above.
2. `pytest -q` passes on this drop.
3. Review `artifacts/morsr_release.jsonl` and `morsr_ticks.jsonl` after tick.
4. Adjust `--sector-bounds` and detensor budgets as needed per scenario.

## P0/P1 Completed in this repo

- Diversity-weighted detector fusion + reliability.
- CPP pre-pass over sector anchors; coverage/leakage/ drift metrics.
- Detensor budget clamp wired into SAP loop; auto-reads Assembly weights from MDHG.
- MDHG decay + hotmap + namespaces view.
- CLI `--with-rag`; RAGAdapter scaffolding (local corpus scan).
- Verdict history (phi\_slope, leakage\_slope).
- Tests for clamp, sector-boundaries, history trends, TSP, nav pipeline.
- Release event emitter and artifact manifest tool.

## Next Planned (open)

- Shell(m>1) streaming enumerator behind a flag; integrate with navigator.
- MDHG typed lanes (schemas) + TTL eviction per namespace.
- Policy masks per subsystem (leakage/complexity thresholds by adapter).
- Auto-artifacts: tick emits `plan_step.json` and writes manifest checksums.
- Agent telemetry booster: include fusion weights and slopes in `keep{}`.

---

## Repo Tree

- Makefile
- README.md
- pyproject.toml
- scripts/cli.py
- scripts/shell\_tour.py
- scripts/tick\_demo.py
- src/snaplat/**init**.py
- src/snaplat/adapters/assembly\_adapter.py
- src/snaplat/adapters/dtt\_adapter.py
- src/snaplat/adapters/e8\_adapter.py
- src/snaplat/adapters/nav\_adapter.py
- src/snaplat/adapters/rag\_adapter.py
- src/snaplat/agrm/factory.py
- src/snaplat/agrm/plan\_adapter.py
- src/snaplat/agrm/simple.py
- src/snaplat/agrm/universal.py
- src/snaplat/assembly/core.py
- src/snaplat/bridges/mannequin\_e8.py
- src/snaplat/cpp/postman.py
- src/snaplat/config/load.py
- src/snaplat/detensor/core.py
- src/snaplat/dtt/harness.py
- src/snaplat/e8/core.py
- src/snaplat/e8/coxeter.py
- src/snaplat/e8/neighbor\_cache.py
- src/snaplat/e8/roots.py
- src/snaplat/e8/shells.py
- src/snaplat/e8/weyl.py
- src/snaplat/mdhg/adapter\_legacy.py
- src/snaplat/mdhg/bus.py
- src/snaplat/mdhg/core.py
- src/snaplat/morsr/enrich.py
- src/snaplat/morsr/log.py
- src/snaplat/morsr/ticklog.py
- src/snaplat/nav/pipeline.py
- src/snaplat/nav/sector.py
- src/snaplat/sap/gates.py
- src/snaplat/sap/history.py
- src/snaplat/sap/loop.py
- src/snaplat/sap/metrics.py
- src/snaplat/snap/state.py
- src/snaplat/superperm/c8.py
- src/snaplat/tsp/solver.py
- src/snaplat/universe/manager.py
- src/snaplat/utils/random.py
- tests/test\_detensor\_clamp.py
- tests/test\_history\_trend.py
- tests/test\_mdhg\_decay.py
- tests/test\_metrics\_aggregate.py
- tests/test\_nav\_pipeline.py
- tests/test\_sector\_boundaries.py
- tests/test\_tsp\_square.py
- tools/manifest.py
- tools/release\_event.py

---

## Full Source Code Listing

### `Makefile`

```makefile
PYTHON ?= python

.PHONY: test tick shell_tour manifest cli

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

tick:
	PYTHONPATH=src $(PYTHON) scripts/tick_demo.py

shell_tour:
	PYTHONPATH=src $(PYTHON) scripts/shell_tour.py

manifest:
	PYTHONPATH=src $(PYTHON) tools/manifest.py

cli:
	PYTHONPATH=src $(PYTHON) scripts/cli.py --help
```

### `README.md`

```markdown
# SnapLat Best-of (v0.3.4-P0)

P0 upgrades: fusion+budgets, CPP pre-pass, detensor↔SAP, keeplist audit.
```

### `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "snaplat"
version = "0.3.4"
description = "SnapLat: lattice-first reasoning system (best-of with P0 upgrades)"
readme = "README.md"
requires-python = ">=3.9"
authors = [{ name="Nick Barker" }]
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=7"]

[tool.pytest.ini_options]
pythonpath = ["src"]
addopts = "-q"
```

### `scripts/cli.py`

```python
import argparse, json
from snaplat.mdhg.bus import MDHGBus
from snaplat.adapters.e8_adapter import E8Adapter
from snaplat.adapters.dtt_adapter import DTTAdapter
from snaplat.adapters.assembly_adapter import AssemblyAdapter
from snaplat.adapters.nav_adapter import NavAdapter
from snaplat.adapters.rag_adapter import RAGAdapter
from snaplat.sap.loop import one_tick
from snaplat.morsr.ticklog import TickLogger

def cmd_tick(args):
    bus = MDHGBus()
    adapters = [E8Adapter(bus, base_point=[0.0]*8), DTTAdapter(bus), AssemblyAdapter(bus), NavAdapter(bus, base_point=[0.0]*8)]
    if args.with_rag:
        adapters.append(RAGAdapter(bus))
    cfg = {"agrm":{"alpha":0.25,"manifold_N":3}, "sap":{"epsilon":0.05,"max_frontier":2000,"max_glyph_dl":5.0}, "detectors":{"min_N":2,"min_diversity":0.5}, "mdhg":{"decay":0.95}, "detensor":{"max_rank":args.max_rank,"max_growth":args.max_growth}}
    def _parse_bounds(s):
        d={}
        for item in (s or '').split(','):
            item=item.strip()
            if not item: continue
            if ':' in item:
                k,v=item.split(':',1)
                try:
                    d[int(k.strip())]=float(v.strip())
                except Exception:
                    pass
        return d
    state = {"frontier_width": 240, "distinct_cells": 240, "word_length": 8, "glyph_dl": 1.0, "sectors": args.sectors, "policy_boundary": args.boundary, "tac": 0.98, "sector_boundaries": _parse_bounds(args.sector_bounds)}
    rec = one_tick(adapters, state, cfg, mdhg=bus)
    TickLogger('artifacts/morsr_ticks.jsonl').emit(rec)
    print(json.dumps({'verdict': rec.get('verdict'), 'fusion': rec.get('fusion'), 'keep': rec.get('keep')}, indent=2))

def main():
    ap = argparse.ArgumentParser(prog='snaplat')
    sub = ap.add_subparsers(dest='cmd')
    t = sub.add_parser('tick')
    t.add_argument('--sectors', type=int, default=16)
    t.add_argument('--boundary', type=float, default=0.9)
    t.add_argument('--sector-bounds', type=str, default='', help='comma list like 0:0.85,1:0.9')
    t.add_argument('--max-rank', type=int, default=16)
    t.add_argument('--max-growth', type=float, default=1.5)
    t.add_argument('--with-rag', action='store_true', help='include RAG adapter (scans data/corpus)')
    t.set_defaults(func=cmd_tick)
    args = ap.parse_args()
    if not hasattr(args, 'func'):
        ap.print_help(); return
    args.func(args)

if __name__ == '__main__':
    main()
```

### `scripts/shell_tour.py`

```python
import json
from pathlib import Path
from snaplat.nav.pipeline import plan_shell_tour
def main():
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    plan = plan_shell_tour([0.0]*8, sectors=16, boundary_ratio=0.9, stride_cross=0)
    (Path("artifacts")/"shell_tour_m1.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print("Wrote artifacts/shell_tour_m1.json")
if __name__ == "__main__":
    main()
```

### `scripts/tick_demo.py`

```python
import json
from snaplat.mdhg.bus import MDHGBus
from snaplat.adapters.e8_adapter import E8Adapter
from snaplat.adapters.dtt_adapter import DTTAdapter
from snaplat.adapters.assembly_adapter import AssemblyAdapter
from snaplat.adapters.nav_adapter import NavAdapter
from snaplat.sap.loop import one_tick
from snaplat.morsr.ticklog import TickLogger

def main():
    bus = MDHGBus()
    adapters = [E8Adapter(bus, base_point=[0.0]*8), DTTAdapter(bus, seed=99), AssemblyAdapter(bus, seed=123), NavAdapter(bus, base_point=[0.0]*8)]
    cfg = {"agrm":{"alpha":0.25,"manifold_N":3}, "sap":{"epsilon":0.05,"max_frontier":2000,"max_glyph_dl":5.0}, "detectors":{"min_N":2,"min_diversity":0.5}, "mdhg":{"decay":0.95}, "detensor":{"max_rank":16,"max_growth":1.5}}
    state = {"frontier_width": 240, "distinct_cells": 240, "word_length": 8, "glyph_dl": 1.0, "sectors": 16, "policy_boundary": 0.9, "tac": 0.98}
    rec = one_tick(adapters, state, cfg, mdhg=bus)
    TickLogger("artifacts/morsr_ticks.jsonl").emit(rec)
    print(json.dumps({"verdict": rec.get("verdict"), "fusion": rec.get("fusion"), "keep": rec.get("keep")}, indent=2))

if __name__ == "__main__":
    main()
```

### `src/snaplat/__init__.py`

```python
__all__ = []
```

### `src/snaplat/adapters/assembly_adapter.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
from math import log
from ..agrm.plan_adapter import PlanAdapterProto, Action, Trail
from ..mdhg.bus import MDHGBus
try:
    from ..assembly.core import stitch
    from ..superperm.c8 import produce_c8, C8Config
    from ..dtt.harness import DTT, DTTConfig
except Exception:
    stitch = None

def md_length(weights: List[float]) -> float:
    s = 0.0
    for w in weights:
        if w > 0.0: s += -w * (log(w)/log(2.0))
    return s

@dataclass
class AssemblyAdapter(PlanAdapterProto):
    mdhg: MDHGBus; seed: int = 99

    def actions(self) -> Dict[str, Action]:
        return {"stitch": Action(op="stitch", tag="C", m=0.5)}

    def step(self, schedule: List[Action], state: Dict[str, Any]) -> Trail:
        if stitch is None:
            glyph_dl = 1.0; self.mdhg.put("asm:last", {"glyph_dl": glyph_dl}, score=1.0, notes={})
            return Trail(metrics={"coverage": 1.0, "drift": 0.0, "glyph_dl": glyph_dl}, artifacts={"weights":[1.0]}, notes={"fallback": True})
        cands = produce_c8(C8Config(seed=self.seed)); ev = DTT(DTTConfig(seed=self.seed)).run(cands); out = stitch(cands, ev)
        w = out["DNA"].weights  # type: ignore
        nz = sum(1 for x in w if x > 0.0); coverage = nz / float(len(w) or 1); gdl = md_length(w)
        self.mdhg.put("asm:last_glyph", out["glyph"], score=coverage, notes={"glyph_dl": gdl})
        self.mdhg.put("asm:last_weights", w, score=coverage, notes={"glyph_dl": gdl})
        return Trail(metrics={"coverage": coverage, "drift": 0.0, "glyph_dl": gdl}, artifacts={"glyph": out["glyph"], "weights": w}, notes={})

    def learn(self, trace: Trail) -> None: return None
```

### `src/snaplat/adapters/dtt_adapter.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
from ..agrm.plan_adapter import PlanAdapterProto, Action, Trail
from ..mdhg.bus import MDHGBus
try:
    from ..dtt.harness import DTT, DTTConfig
    from ..superperm.c8 import produce_c8, C8Config
except Exception:
    DTT = None; produce_c8 = None

@dataclass
class DTTAdapter(PlanAdapterProto):
    mdhg: MDHGBus; seed: int = 42

    def actions(self) -> Dict[str, Action]:
        return {"produce": Action(op="produce", tag="E", m=1.0), "evaluate": Action(op="evaluate", tag="C", m=0.5)}

    def step(self, schedule: List[Action], state: Dict[str, Any]) -> Trail:
        if DTT is None or produce_c8 is None:
            cov = 0.5; drift = 0.05; self.mdhg.put("dtt:last", {"seed": self.seed}, score=cov, notes={"drift": drift})
            return Trail(metrics={"coverage": cov, "drift": drift}, artifacts={}, notes={"fallback": True})
        cands = produce_c8(C8Config(seed=self.seed))
        ev = DTT(DTTConfig(seed=self.seed)).run(cands)
        utils = [e.metrics.get("utility", 0.0) for e in ev]  # type: ignore
        good = sum(1 for u in utils if u >= 0.5)
        coverage = good / float(len(utils) or 1)
        drift = abs(sum(utils)/len(utils) - 0.5)
        self.mdhg.put("dtt:last_utils", utils, score=coverage, notes={"drift": drift})
        return Trail(metrics={"coverage": coverage, "drift": drift}, artifacts={"utils": utils}, notes={})

    def learn(self, trace: Trail) -> None: return None
```

### `src/snaplat/adapters/e8_adapter.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
from ..agrm.plan_adapter import PlanAdapterProto, Action, Trail
from ..mdhg.bus import MDHGBus
try:
    from ..e8.core import nearest, edges, coxeter_plane
except Exception:
    def nearest(v): return ([0.0]*8, 0.0)
    def edges(x): return [[0.0]*8 for _ in range(240)]
    def coxeter_plane(x): return (0.0, 0.0)

def slice_id_for(vec: List[float]) -> str:
    p, _ = nearest(vec); return "cell:" + ",".join(str(int(round(x*2))) for x in p)

@dataclass
class E8Adapter(PlanAdapterProto):
    mdhg: MDHGBus
    base_point: List[float]

    def actions(self) -> Dict[str, Action]:
        return {"neighbors": Action(op="neighbors", tag="E", m=1.0), "project2d": Action(op="project2d", tag="C", m=0.2)}

    def step(self, schedule: List[Action], state: Dict[str, Any]) -> Trail:
        p, _ = nearest(self.base_point or [0.0]*8)
        nbrs = edges(p); distinct_cells = len(nbrs)
        coverage = min(1.0, distinct_cells / 240.0)
        u,v = coxeter_plane(p)
        prev = self.mdhg.get("e8:last_proj", {"u": u, "v": v})
        du, dv = abs(prev["u"]-u), abs(prev["v"]-v)
        drift = min(1.0, (du+dv)/10.0)
        self.mdhg.put("e8:last_proj", {"u": u, "v": v}, score=coverage, notes={"drift": drift})
        self.mdhg.put("e8:last_cell", slice_id_for(self.base_point or [0.0]*8), score=coverage, notes={})
        return Trail(metrics={"coverage": coverage, "drift": drift}, artifacts={"neighbors": distinct_cells}, notes={"u": u, "v": v})

    def learn(self, trace: Trail) -> None: return None
```

### `src/snaplat/adapters/nav_adapter.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
from ..agrm.plan_adapter import PlanAdapterProto, Action, Trail
from ..mdhg.bus import MDHGBus
from ..nav.pipeline import plan_shell_tour

@dataclass
class NavAdapter(PlanAdapterProto):
    mdhg: MDHGBus
    base_point: List[float]

    def actions(self) -> Dict[str, Action]:
        return {"sectorize": Action(op="sectorize", tag="E", m=1.0), "tour": Action(op="tour", tag="C", m=0.5)}

    def step(self, schedule: List[Action], state: Dict[str, Any]) -> Trail:
        sectors = int(state.get("sectors", 16)); boundary = float(state.get("policy_boundary", 0.9))
        stride = int(state.get("stride_cross", 0))
        sb = state.get("sector_boundaries")
        plan = plan_shell_tour(self.base_point or [0.0]*8, sectors=sectors, boundary_ratio=boundary, stride_cross=stride, sector_boundaries=sb)
        cov = plan["coverage"]; coverage = 0.6*cov["sector_coverage"] + 0.4*cov["node_coverage"]
        prev = self.mdhg.get("nav:tour_meta", {"tour_len": plan["length"], "sector_cov": cov["sector_coverage"]})
        drift = min(1.0, abs(prev.get("tour_len", plan["length"]) - plan["length"]) / (prev.get("tour_len", 1.0) + 1e-9) + abs(prev.get("sector_cov", 0.0) - cov["sector_coverage"]))
        self.mdhg.put("nav:tour_meta", {"tour_len": plan["length"], "sector_cov": cov["sector_coverage"]}, score=coverage, notes={})
        self.mdhg.put("nav:anchors", plan["anchors"], score=coverage, notes={"length": plan["length"]})
        self.mdhg.put("nav:tour", plan["tour"], score=coverage, notes={"leakage": plan["leakage"]})
        return Trail(metrics={"coverage": coverage, "drift": drift, "leakage": plan["leakage"]}, artifacts={"tour_len": plan["length"]}, notes={"sectors": sectors})

    def learn(self, trace: Trail) -> None: return None
```

### `src/snaplat/adapters/rag_adapter.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path
import os

from ..agrm.plan_adapter import PlanAdapterProto, Action, Trail
from ..mdhg.bus import MDHGBus

@dataclass
class RAGAdapter(PlanAdapterProto):
    mdhg: MDHGBus
    root: str = "data/corpus"
    index_tag: str = "rag:index_bytes"

    def actions(self) -> Dict[str, Action]:
        return {"scan": Action(op="scan", tag="E", m=1.0)}

    def _scan(self) -> Dict[str, int]:
        base = Path(self.root)
        total = 0; files = 0
        if base.exists():
            for p in base.rglob('*'):
                if p.is_file():
                    files += 1
                    try:
                        total += p.stat().st_size
                    except Exception:
                        pass
        return {"files": files, "bytes": total}

    def step(self, schedule, state: Dict[str, Any]) -> Trail:
        cur = self._scan()
        prev = self.mdhg.get(self.index_tag) or {"val": {"files": 0, "bytes": 0}}
        prevv = prev["val"] if isinstance(prev, dict) else {"files": 0, "bytes": 0}
        # coverage: fraction of files "indexed" — naive: indexed==files (1.0) unless a cutoff is given by state
        cutoff = int(state.get("rag_index_cutoff", cur["files"]))
        coverage = 0.0 if cur["files"]==0 else min(1.0, cutoff/float(max(1, cur["files"])))
        # drift: byte growth ratio
        growth = 0.0 if prevv["bytes"]==0 else max(0.0, (cur["bytes"]-prevv["bytes"])/float(prevv["bytes"]))
        drift = min(1.0, growth)
        leakage = 0.0  # local corpus, no policy boundary by default
        self.mdhg.put(self.index_tag, cur, score=coverage, notes={"drift": drift})
        return Trail(metrics={"coverage": coverage, "drift": drift, "leakage": leakage}, artifacts=cur, notes={})

    def learn(self, trace: Trail) -> None: return None
```

### `src/snaplat/agrm/factory.py`

```python
from .simple import plan as simple_plan, AGRMState

def get_planner(name: str):
    name = (name or "").lower()
    if name == "cmplx":
        try:
            from .cmplx_adapter import CmplxAGRM
            return CmplxAGRM()
        except Exception:
            pass
    class _Simple:
        def plan(self, state: AGRMState):
            return simple_plan(state)
        def learn(self, trace):
            return None
    return _Simple()
```

### `src/snaplat/agrm/plan_adapter.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Dict, Any, List

@dataclass
class Action:
    op: str
    tag: str
    m: float
    est_cost: float = 1.0

@dataclass
class Trail:
    metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    notes: Dict[str, Any]

class PlanAdapterProto(Protocol):
    def actions(self) -> Dict[str, Action]: ...
    def step(self, schedule: List[Action], state: Dict[str, Any]) -> Trail: ...
    def learn(self, trace: Trail) -> None: ...
```

### `src/snaplat/agrm/simple.py`

```python
from dataclasses import dataclass
from typing import Dict
import math

PHI = (1 + 5**0.5) / 2

@dataclass
class AGRMState:
    step: int
    tac: float

def plan(state: AGRMState) -> Dict:
    r = 1 + int(math.floor((state.step / PHI) % 2))
    floors = 1 if state.tac >= 0.5 else 2
    elevators = 1
    schedule = ["neighbors", "edges", "reflect"]
    return {"radius": r, "floors": floors, "elevators": elevators, "schedule": schedule}
```

### `src/snaplat/agrm/universal.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Dict
import math

@dataclass
class DetectCfg:
    manifold_N: int = 2

@dataclass
class BudgetCfg:
    lambda_: float = 0.25

@dataclass
class Scorecard:
    frontier_width: int
    radius: int
    distinct_cells: int
    word_length: int
    glyph_dl: float
    phi_before: float = 0.0
    phi_after: float = 0.0
    def phi(self, alpha: float) -> float:
        return self.frontier_width * alpha - self.glyph_dl

def _clamp(x: float, lo=0.0, hi=1.0)->float:
    return max(lo, min(hi, x))

class UniversalPlanner:
    alpha: float = 0.25

    def detect(self, adapters, state: Dict[str, Any], cfg: DetectCfg):
        out = []
        for a in adapters:
            tr = a.step([], state)
            cov = float(tr.metrics.get("coverage", 0.0))
            drift = float(tr.metrics.get("drift", 0.0))
            leak = float(tr.metrics.get("leakage", 0.0))
            out.append(type("Det", (), {"name": a.__class__.__name__, "coverage": cov, "drift": drift, "leakage": leak}))
        return out

    def fuse_with_diversity(self, detectors, manifold_meta=None):
        if not detectors:
            return {"consensus": 0.0, "diversity": 0.0, "N": 0, "weights": []}
        covs = [d.coverage for d in detectors]
        mu = sum(covs)/len(covs)
        rel = [_clamp(1.0 - getattr(d, "drift", 0.0)) * _clamp(1.0 - getattr(d, "leakage", 0.0)) for d in detectors]
        dev = [abs(c - mu) for c in covs]
        if sum(dev) == 0: dev = [1.0 for _ in dev]
        maxd = max(dev) if dev else 1.0
        w = [r * (0.5 + 0.5 * (d/maxd)) for r,d in zip(rel, dev)]
        sw = sum(w) or 1.0
        w = [wi/sw for wi in w]
        consensus = sum(ci*wi for ci,wi in zip(covs, w))
        var = sum(wi*(ci-consensus)**2 for ci,wi in zip(covs, w))
        diversity = _clamp(1.0 - var*4.0)
        return {"consensus": consensus, "diversity": diversity, "N": len(detectors), "weights": w}

    def score(self, util: float, sc: Any, budget: BudgetCfg) -> float:
        return util * (1.0 - budget.lambda_) - sc.glyph_dl * budget.lambda_
```

### `src/snaplat/assembly/core.py`

```python
from typing import List, Dict, Any
from ..snap.schema import Candidate, DNA, dna_from

def stitch(candidates: List[Candidate], evidences: List[Dict[str, Any]]) -> Dict[str, Any]:
    assert len(candidates) == len(evidences) and len(candidates) > 0
    utils = [e.metrics.get("utility", 0.0) for e in evidences]  # type: ignore
    s = sum(utils) if sum(utils) > 0 else 1.0
    w = [u/s for u in utils]
    dna = dna_from(w, candidates)
    glyph = {"type": "barymix", "weights": w, "parts": [c.payload for c in candidates]}
    return {"glyph": glyph, "DNA": dna}

def replay(dna: DNA, candidate_lookup: Dict[str, Any]) -> Dict[str, Any]:
    parts = [candidate_lookup[cid] for cid in dna.candidates]
    glyph = {"type": "barymix", "weights": dna.weights, "parts": parts}
    return {"glyph": glyph, "DNA": dna}
```

### `src/snaplat/bridges/mannequin_e8.py`

```python
from typing import List, Tuple
from ..e8.core import nearest, edges, coxeter_plane

def slice_id_for(vec) -> str:
    p, _ = nearest(vec)
    return "cell:" + ",".join(str(int(x*2)) for x in p)

def neighbors_for(vec, k: int = 8) -> List:
    p, _ = nearest(vec)
    nbrs = edges(p)
    return nbrs[:k]

def project2d(vec) -> Tuple[float,float]:
    p, _ = nearest(vec)
    return coxeter_plane(p)
```

### `src/snaplat/cpp/postman.py`

```python
from __future__ import annotations
from typing import List, Tuple, Dict
from collections import defaultdict
import math

def build_graph(pts: List[Tuple[float,float]], anchors: List[int], stride_cross: int = 0) -> Dict[int, Dict[int,int]]:
    G: Dict[int, Dict[int,int]] = defaultdict(lambda: defaultdict(int))
    n = len(anchors)
    if n==0: return G
    for i in range(n):
        a = anchors[i]; b = anchors[(i+1)%n]
        G[a][b]+=1; G[b][a]+=1
    if stride_cross and n>2:
        for i in range(n):
            a = anchors[i]; b = anchors[(i+stride_cross)%n]
            G[a][b]+=1; G[b][a]+=1
    return G

def _odd_nodes(G: Dict[int, Dict[int,int]]) -> List[int]:
    return [u for u in G if (sum(G[u].values()) % 2)==1]

def _pair_odds_nearest(G, pts, odds):
    used=set(); pairs=[]
    for i,u in enumerate(odds):
        if u in used: continue
        best=None; bestd=1e18
        for v in odds[i+1:]:
            if v in used: continue
            d = math.hypot(pts[u][0]-pts[v][0], pts[u][1]-pts[v][1])
            if d < bestd: bestd=d; best=v
        if best is not None:
            pairs.append((u,best)); used.add(u); used.add(best)
    for u,v in pairs:
        G[u][v]+=1; G[v][u]+=1

def _euler_tour(G, start: int) -> List[int]:
    g = {u: dict(nei) for u,nei in G.items()}
    stack=[start]; path=[]
    while stack:
        u=stack[-1]
        if g[u]:
            v = next(iter(g[u].keys()))
            g[u][v]-=1
            if g[u][v]==0: del g[u][v]
            g[v][u]-=1
            if g[v][u]==0: del g[v][u]
            stack.append(v)
        else:
            path.append(stack.pop())
    return path[::-1]

def cpp_tour(pts: List[Tuple[float,float]], anchors: List[int], stride_cross: int=0) -> List[int]:
    if not anchors: return []
    G = build_graph(pts, anchors, stride_cross=stride_cross)
    odds = _odd_nodes(G)
    if odds:
        _pair_odds_nearest(G, pts, odds)
    return _euler_tour(G, anchors[0])
```

### `src/snaplat/config/load.py`

```python
from __future__ import annotations
from typing import Dict, Any
import os
def load_config(path: str = None) -> Dict[str, Any]:
    defaults = {"agrm":{"alpha":0.25,"manifold_N":3},"sap":{"epsilon":0.05,"max_frontier":2000,"max_glyph_dl":5.0},"detectors":{"min_N":2,"min_diversity":0.5},"mdhg":{"decay":0.95}}
    try:
        if path and os.path.exists(path):
            return defaults
        env = os.environ.get("SNAPLAT_CONFIG")
        if env and os.path.exists(env):
            return defaults
        if os.path.exists("config/default.yaml"):
            return defaults
    except Exception:
        pass
    return defaults
```

### `src/snaplat/detensor/core.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class DetensorBudget:
    max_rank: int = 64
    max_growth: float = 2.0

@dataclass
class DetensorReport:
    rank_before: int
    rank_after: int
    growth: float
    clamped: bool
    notes: Dict[str, float]

def estimate_rank(weights: List[float]) -> int:
    return sum(1 for w in weights if w > 1e-6)

def clamp(weights: List[float], budget: DetensorBudget) -> DetensorReport:
    rb = estimate_rank(weights)
    growth = float(rb) / float(max(1, min(rb, budget.max_rank)))
    clamped = False; ra = rb
    if rb > budget.max_rank or growth > budget.max_growth:
        idx = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)[:budget.max_rank]
        ra = len(idx); clamped = True
    return DetensorReport(rank_before=rb, rank_after=ra, growth=growth, clamped=clamped, notes={"lambda": 1.0/growth if growth>0 else 1.0})
```

### `src/snaplat/dtt/harness.py`

```python
from typing import List
from dataclasses import dataclass
import hashlib, random
from ..snap.schema import Candidate, Evidence

def _stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)

@dataclass
class DTTConfig:
    seed: int = 0
    perf_scale: float = 1.0

class DTT:
    def __init__(self, cfg: DTTConfig):
        self.cfg = cfg
        self._rng = random.Random(cfg.seed)

    def run(self, candidates: List[Candidate]) -> List[Evidence]:
        out: List[Evidence] = []
        for c in candidates:
            raw = f"{c.id}:{c.payload}"
            h = _stable_hash(raw)
            score = ((h % 10_000_000) / 10_000_000.0)
            stability = (((h//7) % 10_000_000) / 10_000_000.0)
            utility = 0.5*score + 0.5*stability
            metrics = {"score": score * self.cfg.perf_scale, "stability": stability, "utility": utility}
            notes = {"dtt": "simulated", "seed": self.cfg.seed}
            out.append(Evidence(candidate_id=c.id, metrics=metrics, notes=notes))
        return out
```

### `src/snaplat/e8/core.py`

```python
from dataclasses import dataclass
from typing import List, Tuple
import math
from .roots import e8_roots

Vec = List[float]

def _round_half(x: float) -> float:
    return round(x - 0.5) + 0.5

def _parity_even_sum(vec: Vec) -> bool:
    s = sum(vec)
    return int(round(2*s)) % 2 == 0

def is_member(x: Vec, tol: float = 1e-9) -> bool:
    ints = all(abs(c - round(c)) < tol for c in x)
    halves = all(abs((c - 0.5) - round(c - 0.5)) < tol for c in x)
    if not (ints or halves):
        return False
    return _parity_even_sum(x)

def _candidate_Z(v: Vec) -> Vec:
    z = [round(c) for c in v]
    if not _parity_even_sum(z):
        diffs = [abs(vc - zc) for vc, zc in zip(v, z)]
        j = max(range(8), key=lambda i: diffs[i])
        step = 1 if v[j] > z[j] else -1
        z[j] += step
    return [float(c) for c in z]

def _candidate_H(v: Vec) -> Vec:
    y = [_round_half(c) for c in v]
    if not _parity_even_sum(y):
        diffs = [abs(vc - yc) for vc, yc in zip(v, y)]
        j = max(range(8), key=lambda i: diffs[i])
        step = 1 if v[j] > y[j] else -1
        y[j] += step
    return [float(c) for c in y]

def nearest(v: Vec) -> Tuple[Vec, float]:
    z = _candidate_Z(v)
    y = _candidate_H(v)
    dz = sum((vc - zc)**2 for vc, zc in zip(v, z))
    dy = sum((vc - yc)**2 for vc, yc in zip(v, y))
    best = z if dz <= dy else y
    dist2 = min(dz, dy)
    assert is_member(best)
    return best, dist2

def project(v: Vec) -> Vec:
    return nearest(v)[0]

_ROOTS = None

def roots():
    global _ROOTS
    if _ROOTS is None:
        _ROOTS = e8_roots()
    return _ROOTS

def edges(x: Vec):
    assert is_member(x)
    nbrs = []
    for r in roots():
        y = [xi + ri for xi, ri in zip(x, r)]
        if is_member(y):
            nbrs.append(y)
    assert len(nbrs) > 0
    return nbrs

def reflect(x: Vec, r: Vec) -> Vec:
    dot = sum(xi*ri for xi, ri in zip(x, r))
    return [xi - (dot)*ri for xi, ri in zip(x, r)]

_B1 = [1,1,1,1, -1,-1,-1,-1]
norm = math.sqrt(sum(c*c for c in _B1))
_B1 = [c/norm for c in _B1]
_B2 = [1,-1,1,-1, 1,-1,1,-1]
dot12 = sum(a*b for a,b in zip(_B1,_B2))
_B2 = [b - dot12*a for a,b in zip(_B1,_B2)]
norm2 = math.sqrt(sum(c*c for c in _B2))
_B2 = [c/norm2 for c in _B2]

def coxeter_plane(x: Vec) -> Tuple[float, float]:
    return (sum(a*b for a,b in zip(_B1, x)), sum(a*b for a,b in zip(_B2, x)))
```

### `src/snaplat/e8/coxeter.py`

```python
from typing import Tuple, List
try:
    import sympy as sp  # type: ignore
except Exception:  # pragma: no cover
    sp = None  # type: ignore
from .core import coxeter_plane as fallback_project

def get_projector():
    if sp is None:
        return lambda x: fallback_project(x)
    def proj(x: List[float]) -> Tuple[float,float]:
        B1 = sp.Matrix([1,1,1,1,-1,-1,-1,-1])
        B1 = B1 / B1.norm()
        B2 = sp.Matrix([1,-1,1,-1,1,-1,1,-1])
        B2 = B2 - (B2.dot(B1))*B1
        B2 = B2 / B2.norm()
        xv = sp.Matrix(x)
        u = float(B1.dot(xv))
        v = float(B2.dot(xv))
        return (u, v)
    return proj
```

### `src/snaplat/e8/neighbor_cache.py`

```python
from typing import Dict, List, Tuple
from .core import nearest, edges

class NeighborCache:
    def __init__(self):
        self.cache: Dict[Tuple[float,...], List[List[float]]] = {}

    def get_first_shell(self, x: List[float]) -> List[List[float]]:
        px, _ = nearest(x)
        key = tuple(px)
        if key not in self.cache:
            self.cache[key] = edges(px)
        return self.cache[key]
```

### `src/snaplat/e8/roots.py`

```python
"""E8 root system (240 roots) and helpers.
We generate the 240 roots of norm^2 = 2 using the standard construction:
- 112 roots of type A: permutations of (±1, ±1, 0, ..., 0) with an even number of minus signs
- 128 roots of type B: vectors (±1/2, ..., ±1/2) with an even number of minus signs
"""
from itertools import combinations

def e8_roots():
    roots = []
    # Type A
    for i, j in combinations(range(8), 2):
        for s1 in (+1, -1):
            for s2 in (+1, -1):
                vec = [0]*8
                vec[i] = s1
                vec[j] = s2
                if (int(s1<0)+int(s2<0)) % 2 == 0:
                    roots.append(vec)
    # Type B
    for mask in range(1<<8):
        bits = [(mask>>k)&1 for k in range(8)]
        negs = sum(bits)
        if negs % 2 == 0:
            vec = [(-0.5 if b else 0.5) for b in bits]
            roots.append(vec)
    assert len(roots) == 240
    for r in roots:
        n2 = sum(x*x for x in r)
        assert abs(n2 - 2.0) < 1e-9
    return roots
```

### `src/snaplat/e8/shells.py`

```python
from typing import List
from .roots import e8_roots

def sigma3(n: int) -> int:
    s = 0
    for d in range(1, n+1):
        if n % d == 0:
            s += d**3
    return s

def oracle_count(m: int) -> int:
    return 240 * sigma3(m)

def root_shell() -> List[List[float]]:
    return e8_roots()
```

### `src/snaplat/e8/weyl.py`

```python
from typing import List, Iterable
from .core import reflect
from .roots import e8_roots

def apply_reflection(x: List[float], r_index: int) -> List[float]:
    r = e8_roots()[r_index % 240]
    return reflect(x, r)

def apply_sequence(x: List[float], seq: Iterable[int]) -> List[float]:
    y = list(x)
    for idx in seq:
        y = apply_reflection(y, idx)
    return y
```

### `src/snaplat/mdhg/adapter_legacy.py`

```python
try:
    from vendor.mdhg.agrmmdhg import MDHGHashTable as _VendorMDHG  # type: ignore
except Exception:
    _VendorMDHG = None
from .core import MDHG as _SimpleMDHG

class MDHG:
    def __init__(self, use_vendor: bool = True):
        if use_vendor and _VendorMDHG is not None:
            self.backend = _VendorMDHG()
            self.vendor = True
        else:
            self.backend = _SimpleMDHG()
            self.vendor = False

    def put(self, k, v):
        return self.backend.put(k, v)

    def get(self, k, default=None):
        return self.backend.get(k) if hasattr(self.backend, "get") else default

    def has(self, k) -> bool:
        return getattr(self.backend, "has", lambda kk: kk in getattr(self.backend, "kv", {}))(k)

    def hotmap(self):
        if hasattr(self.backend, "hotmap"):
            return self.backend.hotmap()
        return getattr(self.backend, "access", {})
```

### `src/snaplat/mdhg/bus.py`

```python
from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict

class MDHGBus:
    def __init__(self):
        self.kv: Dict[str, Any] = {}
        self.access = defaultdict(int)
    def put(self, key: str, val: Any, score: float=0.0, notes: Dict[str,Any]=None):
        self.kv[key] = {"val": val, "score": score, "notes": notes or {}}
        self.access[key] += 1
    def get(self, key: str, default=None):
        self.access[key] += 1
        v = self.kv.get(key)
        return v if v is not None else default
    def hotmap(self):
        return dict(self.access)
    def decay(self, gamma: float = 0.95):
        for k in list(self.access.keys()):
            self.access[k] = max(0, int(self.access[k] * gamma))
    def top_hot(self, n: int = 10):
        return sorted(self.access.items(), key=lambda kv: kv[1], reverse=True)[:n]

class NSView:
    def __init__(self, bus: MDHGBus, ns: str):
        self.bus = bus
        self.prefix = (ns.strip()+":") if ns else ""
    def _k(self, k: str) -> str:
        return self.prefix + k
    def put(self, key, val, score: float=0.0, notes=None):
        return self.bus.put(self._k(key), val, score, notes)
    def get(self, key, default=None):
        return self.bus.get(self._k(key), default)
    def hotmap(self):
        # filter keys by prefix
        hm = self.bus.hotmap()
        return {k[len(self.prefix):]: v for k,v in hm.items() if k.startswith(self.prefix)}
    def decay(self, gamma: float = 0.95):
        # decay globally; caller can ignore
        return self.bus.decay(gamma)
```

### `src/snaplat/mdhg/core.py`

```python
from collections import defaultdict
from typing import Any, Dict

class MDHG:
    def __init__(self):
        self.kv: Dict[str, Any] = {}
        self.access = defaultdict(int)

    def put(self, k: str, v: Any): self.kv[k] = v
    def get(self, k: str, default=None):
        self.access[k] += 1
        return self.kv.get(k, default)
    def has(self, k: str) -> bool: return k in self.kv
    def hotmap(self): return dict(self.access)
```

### `src/snaplat/morsr/enrich.py`

```python
from typing import Dict, Any, Iterable

DEFAULT_KEEPLIST = [
    "coverage","drift","leakage","consensus","diversity",
    "glyph_dl","neighbors_count","frontier_width","distinct_cells",
    "tac","boundary"
]

def enrich(record: Dict[str, Any], keeplist: Iterable[str] = DEFAULT_KEEPLIST) -> Dict[str, Any]:
    r = dict(record); r.setdefault("metrics", {})
    keep = {k: record.get(k, r["metrics"].get(k)) for k in keeplist if (record.get(k) is not None or r["metrics"].get(k) is not None)}
    if keep: r["keep"] = {k: v for k,v in keep.items() if v is not None}
    return r
```

### `src/snaplat/morsr/log.py`

```python
import json, time, os
from typing import Dict, Any

class MORSR:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
    def emit(self, event: Dict[str, Any]):
        event = dict(event)
        event["ts"] = time.time()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")
```

### `src/snaplat/morsr/ticklog.py`

```python
import json, time, os
from typing import Dict, Any

class TickLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
    def emit(self, event: Dict[str, Any]):
        ev = dict(event); ev["ts"] = time.time()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, sort_keys=True) + "\n")

def make_tick_record(**kwargs) -> Dict[str, Any]:
    return dict(kwargs)
```

### `src/snaplat/nav/pipeline.py`

```python
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import math
try:
    from ..e8.core import nearest, edges, coxeter_plane
except Exception:
    def nearest(v): return ([0.0]*8, 0.0)
    def edges(x): return [[0.0]*8 for _ in range(240)]
    def coxeter_plane(x): return (0.0, 0.0)

from .sector import sectorize, radial_stats
from ..tsp.solver import euclid, nearest_neighbor, two_opt, route_length
from ..cpp.postman import cpp_tour

Point2D = Tuple[float,float]

def shell_points_2d(base_point: List[float]) -> List[Point2D]:
    p, _ = nearest(base_point or [0.0]*8)
    nbrs = edges(p); pts2d: List[Point2D] = []
    for v in nbrs:
        u,v2 = coxeter_plane(v); pts2d.append((u,v2))
    return pts2d

def sector_anchors(pts: List[Point2D], S: int) -> List[int]:
    buckets = sectorize(pts, S); anchors: List[int] = []
    for s in range(S):
        idxs = buckets.get(s, [])
        if not idxs: continue
        best, br = None, 1e18
        for i in idxs:
            r = math.hypot(pts[i][0], pts[i][1])
            if r < br: br=r; best=i
        anchors.append(best)
    seen=set(); out=[]
    for i in anchors:
        if i not in seen: seen.add(i); out.append(i)
    return out

def tour_over_anchors_cpp(pts: List[Point2D], anchors: List[int], stride_cross: int=0) -> List[int]:
    walk = cpp_tour(pts, anchors, stride_cross=stride_cross)
    comp=[walk[0]] if walk else []
    for i in range(1, len(walk)):
        if walk[i] != comp[-1]: comp.append(walk[i])
    return comp

def leakage_metric(pts: List[Point2D], visited: List[int], sectors: int, boundary_ratio: float=0.9, sector_boundaries: Dict[int, float] = None) -> float:
    if not visited or not pts: return 0.0
    S = max(1, sectors)
    buckets = sectorize(pts, S)
    # per-sector max radius
    sec_max = {s: 0.0 for s in range(S)}
    for s, idxs in buckets.items():
        if idxs:
            sec_max[s] = max(math.hypot(pts[i][0], pts[i][1]) for i in idxs)
    R_global = max(math.hypot(x,y) for (x,y) in pts)
    point_sector = {}
    for s, idxs in buckets.items():
        for i in idxs:
            point_sector[i] = s
    over = 0
    for i in visited:
        sidx = point_sector.get(i, 0)
        ratio = boundary_ratio
        if sector_boundaries and sidx in sector_boundaries:
            try:
                ratio = float(sector_boundaries[sidx])
            except Exception:
                ratio = boundary_ratio
        R = sec_max.get(sidx, 0.0) or R_global
        boundary = max(0.0, min(1.0, ratio)) * R
        if math.hypot(pts[i][0], pts[i][1]) > boundary:
            over += 1
    return min(1.0, over / float(len(visited)))

def coverage_metrics(pts: List[Point2D], tour: List[int], sectors: int) -> Dict[str, float]:
    if not pts: return {"sector_coverage": 0.0, "node_coverage": 0.0}
    S = max(1, sectors)
    buckets = sectorize(pts, S); visited = set(tour)
    sec_hit = sum(1 for s in range(S) if any(i in visited for i in buckets.get(s, [])))
    node_cov = len(visited) / float(len(pts))
    return {"sector_coverage": sec_hit / float(S), "node_coverage": node_cov}

def plan_shell_tour(base_point: List[float], sectors: int=16, boundary_ratio: float=0.9, stride_cross: int=0, sector_boundaries: Dict[int, float] = None) -> Dict[str, Any]:
    pts = shell_points_2d(base_point); anchors = sector_anchors(pts, sectors)
    tour = tour_over_anchors_cpp(pts, anchors, stride_cross=stride_cross)
    cov = coverage_metrics(pts, tour, sectors); leak = leakage_metric(pts, tour, sectors, boundary_ratio, sector_boundaries)
    total_len = route_length(tour, pts)
    return {"pts2d": pts, "anchors": anchors, "tour": tour, "length": total_len, "coverage": cov, "leakage": leak, "stats": {"sectors": sectors, **radial_stats(pts)}}
```

### `src/snaplat/nav/sector.py`

```python
from __future__ import annotations
from typing import List, Tuple, Dict
import math
Point2D = Tuple[float, float]

def angles(pts: List[Point2D]) -> List[float]:
    return [math.atan2(y,x) for (x,y) in pts]

def sectorize(pts: List[Point2D], S: int) -> Dict[int, List[int]]:
    if S <= 0: S = 8
    angs = angles(pts); out = {i: [] for i in range(S)}
    for idx, a in enumerate(angs):
        aa = (a + 2*math.pi) % (2*math.pi); sec = int((aa/(2*math.pi))*S) % S
        out[sec].append(idx)
    return out

def radial_stats(pts: List[Point2D]):
    rs = [math.hypot(x,y) for x,y in pts]
    return {"min": min(rs) if rs else 0.0, "max": max(rs) if rs else 0.0, "mean": sum(rs)/len(rs) if rs else 0.0}
```

### `src/snaplat/sap/gates.py`

```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class GateOutcome:
    ok: bool
    reason: str

@dataclass
class C
```
