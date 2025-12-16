# Code Consolidation — Batches & Winners

**Goal:** Stage a deep consolidation pass: group exact/structural/near‑duplicate code, form review **batches**, select provisional **winners**, then test winners against all non‑named similar/structural neighbors.

---

## Pipeline (LOCK)
1) **Discover & extract** nested zips → `/mnt/data/_extracted/*` (breadth‑first).  
2) **Index** all `.py` files → size, lines, checksums (`raw_sha256`), **structural signature** (`normsig_sha`) via token‑shingle normalization, **basename clusters** (name‑based).  
3) **Group**: (a) exact dup groups, (b) structural dup groups (same `normsig_sha`), (c) **near‑dup pairs** within basename cluster using Jaccard ≥ 0.80 over 5‑gram shingles.  
4) **Batch creation**: one batch per basename cluster; within batch, sub‑batches for structural dup groups; attach near‑dup pairs as “challenge edges”.  
5) **Winner rubric (first pass)**: completeness, compile/import‑clean, config fidelity, logging/telemetry hooks, testability, boundary coverage, performance hints, docstrings/type hints, coupling to other systems.  
6) **Face‑off**: run lightweight static checks & smoke tests; winners compared against challenge edges.  
7) **Promote**: winners → **Best‑Of staging**; non‑winners archived with Trails.

---

## Artifacts (generated now)
- **Index CSV:** `/mnt/data/_intake_outputs/code_index.csv`  
- **Exact duplicates:** `/mnt/data/_intake_outputs/exact_duplicates.csv`  
- **Structural duplicates:** `/mnt/data/_intake_outputs/structural_duplicates.csv`  
- **Near‑duplicate pairs:** `/mnt/data/_intake_outputs/near_duplicate_pairs.csv`  
- **Zip extraction log:** `/mnt/data/_intake_outputs/zip_extraction_log.json`

> Interactive tables have been shared to your workspace for quick inspection.

---

## Batch Assembly (seed rules)
- **Batch key:** `basename_cluster`.  
- **Sub‑batch A:** files sharing `normsig_sha` (identical structure).  
- **Sub‑batch B:** near‑dup pairs at Jaccard ≥ 0.80 (review order by score desc).  
- **Outliers:** same basename but no structural similarity → append as “challenge candidates”.

---

## Winner Selection Rubric (detail)
- **Correctness & completeness:** no missing imports/symbols; matches config contract.  
- **Design hygiene:** separation of concerns; no conflated modules; minimal global state.  
- **Observability:** clear logging, error taxonomy; Trail hooks (where applicable).  
- **Testability:** deterministic seams; fixture‑friendly; reproducible seeds.  
- **Performance posture:** avoids O(N^2) hotspots unless justified; cache/boundary awareness.  
- **Integration fitness:** clear boundaries to MDHG/AGRM/TPG/Trails/Policy; config via manifest; no silent fallbacks.  
- **Docs & types:** docstrings/type hints where feasible.

---

## Next Steps
1) **Batch Sheets — Pass 2 (generated):** Per‑cluster sheets with flags + near‑dup edges saved under `/mnt/data/_intake_outputs/batches/<cluster>/`.  
   - Global provisional table: `/mnt/data/_intake_outputs/provisional_rankings_pass2.csv`.
2) **Provisional nominees:** Top 1–2 per cluster chosen with preference for clean candidates (no parse errors, no missing imports).  
3) **Upcoming (Pass 3 — smoke tests):** Prepare importability checks and config‑surface sanity tests for nominees vs. challengers (pure Python; no external calls).  
4) **Promotion criteria:** Nominee must pass smoke suite and integrate cleanly (no risky IO), then promoted to **Best‑Of staging** with Trail notes.  
5) **Hold on editing/build:** No refactors yet. We stay intake‑only until you approve Pass 3 execution.

---

## Pass 3 — Smoke Tests (nominees vs challengers)
**Method:** Safe import/exec with **stubs** for heavy/risky deps (`sklearn`, `networkx`, `requests`, `urllib*`, `subprocess`) and a blocked `open()` during import. Import executed via `importlib.util.spec_from_file_location` into an isolated module. No network/filesystem/process calls permitted.

**Inputs:**
- Provisional nominees: `/mnt/data/_intake_outputs/provisional_rankings_pass2.csv` (top‑1 per `basename_cluster`).
- Challengers: near‑dup edges with **Jaccard ≥ 0.90** from `/mnt/data/_intake_outputs/near_duplicate_pairs.csv`.

**Outputs:**
- **Smoke results:** `/mnt/data/_intake_outputs/smoke_results_pass3.csv` (status=`ok|parse_error|exec_error`, classes/functions discovered).  
- **Suggested winners:** `/mnt/data/_intake_outputs/winners_pass3_suggested.csv` (prefers `ok` status; tie‑break by lines).

**Notes:**
- Any attempt to use `open()`/`subprocess`/blocked modules at import time raises a SAFE‑STUB error, recorded under `exec_error`.
- This is a **pure intake** action—no edits or refactors performed.

**Next (Pass 4 — Review & Promote):**
1) Inspect smoke table for failures; for `exec_error` cases, categorize by cause (missing dep vs risky operation).  
2) Confirm **winner per cluster**; non‑winners archived with Trails (to be wired).  
3) Prepare **Best‑Of staging** tree mirroring the proposed repo layout; no code changes yet—just **selected sources** copied into place for review.

---

## Pass 4 — Review & Promote (Best‑Of staging)
**Action:** Selected winners per `basename_cluster` (from Pass 3) staged into a **repo‑shaped tree** (no edits) under:  
`/mnt/data/_best_of_staging/src/<cluster>/<original_filename>`

**Manifest:** `/mnt/data/_best_of_staging/best_of_manifest.jsonl` (JSONL per file with cluster, original path, staged path, checksums, line counts, structural sigs, smoke status, readiness score).

**Table shared:** *Best‑Of Staging — Winners Manifest* (interactive).

**Notes:**
- This is **not** the final repo; it’s a review staging. No refactors or code changes yet.  
- Non‑winners remain in intake; promotion Trails to be wired during build phase.

**Next (Pass 5 — Interface Diff & Fit):**
1) Per‑cluster **Interface Sheet**: public symbols (classes/functions), config surfaces, and cross‑system touchpoints.  
2) Flag conflicts/missing surfaces vs. reconstructed design (MDHG/AGRM/TPG/Trails/W5H).  
3) Prepare **repo layout mapping** (from “possible file structure.txt”) and show where each staged winner would live.  
4) Gate build: once mapping is approved, initialize the git repo and start refactors with full downloadable bundle per your rule.

---

## Pass 5 — Interface Diff & Fit (no edits)
**Outputs created:**
- **Interface Sheets:** `/mnt/data/_intake_outputs/pass5/interface_sheets.csv` (per staged winner: classes, methods, functions, constants, imports, mentions, docstrings/types, parse status).  
- **Interface Details (JSON):** `/mnt/data/_intake_outputs/pass5/interface_details.json` (deep per‑module structures for diffing).  
- **Proposed Repo Mapping:** `/mnt/data/_intake_outputs/pass5/proposed_mapping.csv` (target path per file with mapping reason; heuristics + layout hint).

**Heuristics (transparent):**
- `superperm*/superpermutation` → `plugins/edges/superperm/`  
- `mdhg` → `agrm/mdhg/` ; `agrm` → `agrm/` ; `tpg` → `tpg/`  
- `trail(s)` / `w5h` / `beacon(s)` → `agrm/snap/`  
- `uniqueness/laminates/caches/promotion/batch_runner` → `plugins/edges/superperm/`  
- `mannequin/porter` → `mannequin/` ; `policy + hash` → `ops/policy/` ; else → `misc/`.

**Next (Pass 5b — Interface Fit Report):**
- Generate a **Fit Report** overlay: for each file, highlight cross‑system touchpoints (e.g., MDHG↔AGRM, Trails↔Reports) and **gaps** vs. reconstructed design (columns missing, hooks absent).  
- Propose **minimal shims** (names only, no code) required for clean integration.

---

## Pass 5b — Interface Fit Report (no edits)
**Generated files:**
- **Interface Fit Report:** `/mnt/data/_intake_outputs/pass5/interface_fit_report.csv`  
  Columns: cluster, file, proposed_repo_path, mentions, **expected_touchpoints**, **detected_signals**, **gaps**, **minimal_shims** (names/signatures only), docstring/type‑hint flags, parse status.

**Touchpoint model (transparent):**
- AGRM/MDHG → expect `policy_hash` & `trails_hooks`; MDHG also `to_points/to_graph` and `promotion_breakdown`.  
- TPG → expect `tpg_config` + `neg_beacon_support`.  
- Trails/W5H/Beacons → expect `trails_hooks`, `w5h_alignment`, `beacons_registry`.  
- Superperm/FS2 → expect `fs2_uniqueness_cache_envelope` surfaces.  
- Mannequin/Porter → expect `safe_cube_sentinel` + `porter_custody`.

**Next (Pass 6 — Repo Layout & Build Gate):**
1) Render **repo mapping plan** using `proposed_mapping.csv` (show final folders/files).  
2) Attach **shims file** per package (names only) to guarantee clean integration when we begin refactors.  
3) With your go‑ahead, initialize **git repo** and copy staged winners into place + shim skeletons; produce the first downloadable bundle.

---

## Pass 6 — Repo Layout & Build Gate (executed)
**What I did (no refactors yet):**
- Created **git‑ready repo skeleton** at `/mnt/data/_repo_snaplat/` with `src/` tree.  
- Copied each **staged winner** into its **proposed repo path** under `src/` (packages initialized with `__init__.py`).  
- Generated **shim skeletons** (names/signatures only) per directory to satisfy missing integration surfaces flagged in Pass 5b.  
- Wrote **repo manifests** for provenance.

**Artifacts:**
- **Repo root:** `/mnt/data/_repo_snaplat/`  
- **Copied sources index:** `/mnt/data/_repo_snaplat/_build_indexes/copied_sources.csv`  
- **Shims index:** `/mnt/data/_repo_snaplat/_build_indexes/shims_index.csv`  
- **Repo manifest (provenance):** `/mnt/data/_repo_snaplat/_build_indexes/repo_manifest.csv`  
- **Downloadable bundle:** `/mnt/data/snaplat_best_of_repo.zip`

**Notes:**
- Shims live as `_shims.py` in the target package dirs; they include only **minimal signatures** (`...` bodies or `pass`).  
- This stage fulfills your "start build → provide full downloadable repo" gate while keeping code **unchanged** aside from directory placement and shim files.

**Next (Pass 7 — Planning for Refactor):**
1) Generate a **Refactor Plan** per package (rename/split, dependency reduction, telemetry insertion, config alignment).  
2) Stage **unit harness skeletons** aligned with shims/interfaces so we can test while refactoring.  
3) Propose **milestone cuts**: MDHG core, AGRM orchestrator, Trails/W5H, FS2/superperm plugin, Policy/Mannequin.

---

## Pass 7 — Refactor Planning (no logic changes yet)
**Created in repo:** `/mnt/data/_repo_snaplat/PLANS/`
- `REFACTOR_PLAN.md` — principles, intake summary, refactor waves, risks/mitigations, DoD.  
- `PLAN_<family>.md` — per‑package file lists, known gaps from Fit Report, minimal shims, first steps.  
- `MILESTONES.md` — M1..M5 milestone cuts.

**Tests (skeletons):** `/mnt/data/_repo_snaplat/tests/`  
- `test_mdhg_api.py`, `test_agrm_tpg_api.py`, `test_trails_w5h_api.py`, `test_fs2_superperm_api.py`, `test_policy_mannequin_api.py` (placeholders; to be enabled during refactor).

**Next (Pass 8 — Execution Plan & Work Queue):**
- Derive a **prioritized work queue** from gaps → shims → refactor waves.  
- Attach **Trail IDs** templates and logging schema to instrument as we go.  
- Propose **branch strategy** and change‑review checklist.

---

## Pass 8 — Execution Plan & Work Queue (no logic changes)
**Deliverables added to repo** `/mnt/data/_repo_snaplat/PLANS/`:
- `WORK_QUEUE_PASS8.csv` — prioritized tasks from Fit Report gaps (milestone, priority, size, target path, gap, shim hint).  
- `EXECUTION_PLAN.md` — overview of priorities (P0→P3) and milestone routing.  
- `TRAILS_LOGGING_SCHEMA.json` & `TRAIL_TEMPLATES.json` — structured logging schema + begin/append/finalize templates.  
- `BRANCHING.md` — branch strategy and PR gates.  
- `REVIEW_CHECKLIST.md` — criteria for acceptance.

**Next (Pass 9 — Kickoff M1 sandbox)**
- Create a **M1 scratch branch** and stand up a minimal, isolated sandbox that wires Trails stubs and config injection around MDHG `to_points/to_graph` and `promotion_breakdown` surfaces, without altering winner code yet.  
- Prepare a first **change list** proposal and corresponding test fixtures.

