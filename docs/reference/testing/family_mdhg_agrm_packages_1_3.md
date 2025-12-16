# Family — MDHG & AGRM (Packages #1–#3)

**Linkage:** Running Log Reconstruction (Fresh Pass v3) → Packages #1–#3.

---

## Scope (doc‑grounded)
- Consolidate **MDHG** (Hierarchy/HotMap/Elevators/Persistence/Policy) and **AGRM** (Controller/Sweeps/Planner/Message‑Passing) as implemented across early versions.
- Bind **Mannequin + Sentinel (Safe‑Cube)** and **Universe materializers** (to_points/to_graph) to MDHG/AGRM flows.
- Capture **PolicyBus + Run Manifests** as the governance spine.

---

## Components & Evidence

### 1) Governance Spine — PolicyBus + Run Manifests
- Wire‑ins at mannequin, controller, and promotion; manifests emitted for runs/events; HotMap recording. (Files: `policy_bus.py`, `run_manifest.py`, `mdhg/hotmap.py`, mannequin/controller/promotion integration.)
- Default behavior: conservative; allow unless policy denies. *Doc evidence captured in main log.*

### 2) Mannequin + Sentinel (Safe‑Cube) + Porter custody
- Toggleable, slice‑activated lattice; states DORMANT → WARM → ACTIVE_SLICE; Sentinel gate enforces policy (e.g., `safe_cube_requires_tags`).
- Universe policy profiles (YAML) + loader; slice SNAPs persisted for lineage.

### 3) Universe → MDHG/AGRM Materializers
- `to_points(...)` (8‑D coords from mannequin slices) and `to_graph(...)` (neighbor graph + elevators quota), streaming only touched data.
- Safe‑Cube policy example (deny when tags omitted) and quick usage snippet included in the log.

### 4) MDHG — Hierarchy, Heat/Decay, Splits, Tagging, Elevators, Persistence
- Data model: Building → Floor → Room; `HierarchyManager` with `build_initial`, `apply_heat`, `snapshot`.
- Policies: `decay_half_life_hours`, `split_policy` (family/type deny e.g. fiction), `elevator_*` thresholds; room tagging via glyph three‑word base; `persist=True` loads/saves latest hierarchy.
- Events: `mdhg_event::elevator` emitted for candidates when enabled.

### 5) AGRM — Controller + Sweeps + Planner + Message‑Passing
- Controller v0_7: policy precheck, optional sweeps (hemisphere/quadrant/x‑arms), hotmap capture, run manifest; fast‑lane toggle.
- Planner chooses retrieval radius, floors, elevator budgets; bounded‑radius propagation and sheaf‑style consistency; feedback updates MDHG heat/stability.

---

## Interfaces (as in log)
- `materialize.to_points(...)`, `materialize.to_graph(...)`
- `mannequin.activate_stream(criteria, wave_size, max_nodes)` (SAP‑gated)
- `PromotionManager.stage_result(...)`, `promote_after_full(...)`
- MDHG hooks in controller: `persist`, `decay_half_life_hours`, `split_policy`, `promote_elevators`, `elevator_score_min`; emits `mdhg_event` when enabled.

---

## Acceptance Criteria & Test Surfaces (as in log)
- Unit tests for samplers/adjacency/materializers/controller; Safe‑Cube demo denies without tags.
- MDHG persistence/policy test: saves/loads hierarchy; verifies split blocking; docs for persistence/decay/tagging/elevators.
- Hierarchy test: inject cross‑room heat → expect elevators; controller summary exposes `{floors, rooms, elevators}`.
- AGRM acceptance: recall/MRR, grounding rate, latency p50/p95, stability, explainability, cost; targets vs vector‑DB baseline.

---

## Risks & Mitigations (from log text)
- Cold‑start, concept drift, latency spikes; mitigations: bootstrap clusters, sliding windows/shadow builds, batch elevator proposals, rate limits, cache negatives.
- Determinism & lineage; policy enforcement at all boundaries; structured SNAP meta schema.

---

## TODOs (family doc)
- Extract concrete config examples for MDHG (`cfg['mdhg']`) and Safe‑Cube policies; park in `configs/` examples.
- Collate controller summaries and MDHG snapshots into a minimal **ops report** template.
- Map inverse‑glyph exploration flow into elevator candidate enrichment (review‑only here).

