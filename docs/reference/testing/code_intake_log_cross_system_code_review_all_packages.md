# Code Intake Log — Cross‑System Code Review (All Packages)

**Purpose:** A living intake and audit log focused on **code**: modules, versions, bundles, and how code is used across systems (MDHG, AGRM, Superperm/FS2, TPG, Trails/W5H, Policy/Mannequin/Universes). Built from the running log; no invention.

---

## 0) Scope & Rules (LOCK)
- Intake only what’s evidenced in the text (file paths, module names, versions/bundles, config flags).  
- Cross‑link each code item to its **Idea Package**(s) and **family docs**.  
- Stage items needing source retrieval (e.g., from `code bits.zip`) as **Pending Source** until we pivot to the code batch review.

---

## 1) Version & Bundle Map (as observed)
- **v0.30.22** — naive superperm baseline (n=7 length 35277).
- **v0.30.23** — AGRM/MDHG‑guided superperm (n=7 length 5913; silver).
- **v0.31.1** — pkg/workspace zip; strict batch (n=7) with envelopes + caches + uniqueness.
- **v0.31.2** — scaled batch (8 seeds × 90k) + per‑seed prefill (~400 hints) from 5906.
- **v0.31.3** — scaled batch (10 × 120k), dynamic neg thresholds, trail‑keyed cache, doc‑intake smoke.
- **Ops/A‑B** — v0.29.0 FAST; v0.29.1 Golden (time‑shift, leakage, superperm sanity); v0.30.1→0.30.6 Manifest/A‑B runs.

**File checksums (attached now):** see `code_intake_index.json` (and interactive table "Code Intake — Initial File Scan").

> Artifacts to attach to this log when sourced: `snaplat_pkg_v0_31_1*.zip`, `snaplat_v0_31_2_scaled_batch.zip`, `snaplat_v0_31_3_scaled_batch_and_intake.zip`, Golden/Manifest/A‑B CSV/HTML bundles.

---

## 2) Code Families → Units (inventory)
### AGRM + MDHG (User drop: `agrmmdhg.py`) — **Source Attached**
- **SHA256:** *(see index)*  
- **Imports:** core stdlib; optional `sklearn.neighbors` (KDTree/BallTree) with fallback; `numpy` present.  
- **Key classes/areas observed:** `MDHGHashTable` (multi‑building, velocity+core+conflict regions; GR sizing; Hamiltonian‑like paths; promotion to velocity); hashing (`murmur`, `fnv`), location map, conflict structures; dynamic adaptation stubs.  
- **Immediate flags:** uses `print` warnings; heavy console noise; relies on `numpy` but minimal usage; **telemetry/logging hooks missing**; resizing/opt thresholds partially hardcoded; path cache/shorcuts unpersisted; no unit tests.  
- **Placement (proposed):** `agrm/mdhg/` split across `hash_table.py`, `paths.py`, `conflict.py`, `adaptation.py`; exports via `agrm/mdhg/__init__.py`.

### Superpermutation Solver (User drop: `superperm code.py`) — **Source Attached**
- **SHA256:** *(see index)*  
- **Imports:** `pandas`, `networkx`, `sklearn` (linear_model, model_selection, metrics); simulated sandbox; logging; os/subprocess.
- **Modules conflated in one file:** `config.py`, `utility.py`, `graph_utils.py`, `layout_memory.py`, plus DTT/sandbox classes.  
- **Immediate flags (concrete):**
  - **Missing import:** uses `hashlib` in `compute_checksum` but not imported.  
  - **Undefined symbol:** `AntiLaminate` referenced in `LayoutMemory` ctor but not defined.  
  - **Inconsistent config access:** `RestrictedEnvironment` expects `sandbox_timeout` et al., while config provides `auto_adjust_params.sandbox_timeout_base` etc.  
  - **Mixed concerns:** DTT/sandbox/graph/layout memory/ML all co‑located; simulated execution paths (`SIMULATED`) may mask runtime issues.  
  - **Potential cyclics:** `from config import ConfigManager` while `ConfigManager` also defined inline.  
- **Placement (proposed):** move into `plugins/edges/superperm/` with proper package split: `config.py`, `utility.py`, `graph_utils.py`, `layout_memory.py`, `dtt.py`, `sandbox.py`, `solver.py`, `experiments/`.

### Superperm / FS2 (P#4)
- `uniqueness.py`, `caches.py`, `laminates.py`, `promotion.py`, `batch_runner.py` (to be mapped when source arrives/exists in bundles).

### AGRM / Orchestrator / TPG (P#5)
- `agrm/orchestrator/iterate.py`, `tpg/surgery.py`, `tpg/run.py`, `agrm/config/neg_beacons.py` (to be mapped when source arrives).

### MDHG / Mannequin / Universes (P#1–#3)
- `mdhg/*`, `mannequin/*`, `policy_bus.py`, `run_manifest.py`, `materialize.py` (pending source).

### Trails / W5H & Beacons (P#6)
- `agrm/snap/trails.py`, `w5h/*`, `beacons/*` (pending source).

### Repo Scaffold & Tools (P#3)
- `tools/cli.py`, `core/agrm/solver_legacy.py`, tests/, fixtures/ (pending source).

## 3) Code Intake Row — schema
`{package_id, family, module_path, version_tag, artifact_bundle, deps[], config_flags[], tests[], fixtures[], cross_system_links[], status(pending|ready), notes}`

- **cross_system_links** examples: (MDHG ↔ AGRM), (FS2 ↔ Uniqueness/Caches/Envelopes), (TPG ↔ A/B/Neg‑Beacons), (Trails ↔ Reports), (Mannequin ↔ PolicyBus/Universes).
- **status**: *pending* until source is attached; *ready* after retrieval and checksum.

---

## 4) Review & Diff Plan
- **Static checks:** imports, missing deps, circulars, policy_hash pins, logging to Trails.
- **Interfaces:** materializers (`to_points/to_graph`), controller flags (elevators, decay, persist), tpg config, FS2 gates (envelope/uniqueness), W5H score panel exposure.
- **Snapshots & invariants:** canonicalization tests (relabel/rotation/reversal), envelope checkpoints, cache write‑gating, neg‑beacon enforcement.
- **Diffs:** version deltas (v0.31.1→0.31.2→0.31.3), A/B harness 0.30.3→0.30.6; MDHG scoring changes when W5H term added.
- **Cross‑system usage:** confirm Trails/Reports stitching; verify Safe‑Cube/PolicyBus gating across systems.

---

## 5) TODOs (intake)
- Populate **Code Intake Rows** for all units above; mark **Pending Source** where needed.  
- Attach available bundles/CSV/HTML artifacts; compute checksums; record `artifact_bundle` fields.  
- Prepare **diff tasks** per version hop; generate an **issues ledger** for code smells and interface drifts.  
- Stage **unit/functional test** harness pointers (fixtures, seeds, policies).

---

## 6) Open Questions (to resolve at code‑batch time)
- Source of truth for `materialize.py` signatures and quota logic.  
- Exact boundaries between `mannequin` vs `porter` custody responsibilities.  
- Canonicalization fallback order when structural fingerprints disagree.

