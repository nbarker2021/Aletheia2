# SnapLat — Running Log Reconstruction (Fresh Pass v3)

**Scope:** Rebuild a *single running log* from the main log text; treat the compressed doc as a concise reworded summary. This canvas will be the authoritative, continuously-updated reconstruction for recovery and R&D.

---

## Standing Protocol v3 (LOCK)
- **Log-aware segmentation:** `You said:` → *Directive/intent* (human). `ChatGPT said:`/`Thought for …` → *Action/result* (AI). Each segment becomes a **Directive SNAP** or **Action SNAP** with lineage.
- **Idea Package segmentation (NEW):** Group contiguous work into **Idea Packages**: a Human-led idea (one or more `You said:`) + all responding AI work blocks, ending at the next **task pivot**.
- **Backtracking & foresight:** Every pass re-verifies earlier slices and proposes **forward bindings** strictly derivable from text (no invention). Deltas are logged.
- **Keyword expansion (rolling):** Each pass harvests tokens from headers, filenames, modules, reason codes, and tables; grows a **keyword ledger**, **alias map**, and **facet index**. Monotonic growth required.
- **Eight‑way completeness:** Weyl 8‑point coverage (positive + inverse) is required before any slice or package is marked “complete”; interim Safe‑Cube rule recorded with TTL.
- **SNAP everywhere:** All decisions/artifacts are mirrored as SNAP entries (manifests, overlays, diffs, trails).
- **Backfilling = polish:** Line‑range bookmarks, cross‑links, and citation polish are performed opportunistically during review and again as a **final finishing pass** after segmentation is complete.

---



## Sources (D1/D2)
- **D1:** `full session SnapLat build.txt` (primary work log)
- **D2:** `compressed full conversation.txt` (summary/cheat‑sheet derived from D1)

---

## Ingestion Outputs (this canvas emits)
- **/ledger/keyword_ledger.snap** — Append‑only list of newly harvested terms per pass, with provenance pointers.
- **/ledger/alias_map.json** — Canonical → aliases (synonyms, renames, hyphenation/snake/kebab variants).
- **/ledger/facet_index.json** — Term → {module | policy | reason | universe | artifact} bindings.
- **/runs/run_manifest.snap** — Per reconstruction pass: seeds, policies, toggles, counts.
- **/weyl/pass_ledger.snap** — 8‑direction sweep coverage per idea and interior lattices.

> These are *logical* artifacts for traceability; the content is tracked in this canvas until repo storage exists.

---

## Pass‑0 Bootstrap (fresh sweep)
**Goal:** Establish structure, seed ledgers, and lay down a time‑ordered index of major themes/artifacts without interpretation beyond the text.

### A) Time‑ordered milestone index (seed)
- **Branding & namespace:** SnapLat brand adoption; alias/CLI shim; public API re‑exports.
- **Governance:** Personal‑use license pack; PolicyBus central gate; run manifests; Safe‑Cube sentinel; policy pinning.
- **Shelling/Glyphs:** `shell(meta,max_n=5)`, 3‑word base; `compress/inflate/invert`; inverse hooks + tests.
- **AGRM & Sweeps:** Hemisphere/quadrant/x‑arm sweeps with feedback; controller hooks.
- **MDHG:** Hierarchy/hotmap with decay; elevator events; later bandit/learner and persistence across runs.
- **TPG:** Telemetry‑only integration; bridge/proposal SNAP emission; proposal‑as‑hint canary.
- **W5H & Beacons:** Beacons/Sub‑Beacons; room‑level W5H tagging; scoring integration; Trails.
- **Ops & Reports:** A/B and Iterative runners; HTML reports; Trail anchors; per‑universe policies; fast‑path gating.
- **Iterative pipeline:** Staging worker; DNA overrides; FreshStart loops; re‑audit (FS2); baseline_proposal on threshold.
- **Superperm track:** n=7 baseline; AGRM/MDHG‑guided builder with overlap histogram; multi‑run stats.

### B) Artifact index (seed)
- **Bundles/patches:** `snaplat_full_v*`, `snaplat_patch_v*`, `snaplat_workspace_v*`.
- **Reports:** `ab_report.html`, `iter_report.html`, demo summary/trials.
- **CSVs/JSON:** `ab_results.csv`, `ab_deltas.csv`, `iter_results.csv`, `iter_summary.csv`, ops/test logs, superperm stats.
- **Modules:** `policy_bus.py`, `run_manifest.py`, `mdhg/hotmap.py`, `beacons/registry.py`, `w5h/extract.py`, `snap/trails.py`, `orchestrator/iterate.py`, `agrm/agrm/sweeps.py`, `shelling/engine.py`, `glyphs/codec.py`, `workers/staging_worker.py`.

### C) Keyword ledger (seed, non‑exhaustive)
`snaplat`, `policy_bus`, `safe_cube`, `sentinel_pass`, `run_manifest`, `overlay`, `universe_diff`, `hotmap`, `elevator_candidate`, `fastlane`, `promotion_manager`, `dna_overrides`, `staging_worker`, `trail_id`, `neg_beacons`, `w5h_alignment`, `beacon`, `subbeacon`, `baseline_candidate`, `baseline_proposal`, `fs2`, `freshstart`, `tpg_bridge`, `tpg_order`, `proposal_as_hint`, `golden_runner`, `n=7`, `overlap_6`, `superperm_stats`.

### D) Alias map (seed examples)
- **Playbook** ↔ **TPG path** (context: A/B comparisons)
- **Direct** ↔ **baseline/chrono** (context: A/B)
- **Bridge SNAP** ↔ **family=bridge, type=tpg_bridge**

### E) Facet bindings (seed examples)
- **Reason codes:** `sentinel_pass` → Mannequin/PolicyBus allow path.
- **Policies:** `safe_cube_requires_tags` → mannequin gate; `policy_hash` → trail/SNAP pinning.
- **Universes:** Governance/Docs/Work/User (policy_by_universe; per‑universe weights/thresholds).

---

## Pass‑1 — Sequential reconstruction (doc‑grounded)
**Goal:** Exhaustively enumerate versioned cuts, artifacts, modules, and runtime behaviors in execution order (no invention). The compressed doc (D2) serves only as cross‑check phrasing.

### v0.16.x (2025‑08‑13)
- **v0.16.0 (alias demo):** `snaplat` namespace & CLI shim; demo bundle.
- **v0.16.2 (governance core):** PolicyBus, run manifests, HotMap logging threaded through mannequin/controller/promotion; patch package.

### v0.20.0 (SNAPOps + Universes + FastLane) — patch
- **Overlays & diffs:** `attach_mdhg_overlay`, `write_universe_diff`.
- **FastLane:** policy‑driven elevator review → `ElevatorDecision`.
- **Ops Center:** minimal wrapper; controller emits enriched elevator candidates.

### v0.21.0 (promotion workflow + end‑to‑end)
- **Promotion pass:** `promote_elevators(...)` scans `mdhg_event::elevator` → runs Ops review → writes `result::elevator` → updates universe overlay; test & docs.

### v0.23.0 (FreshStart + Iterative Orchestrator + telemetry fix)
- **FreshStart:** repo/UM wipe, reset overlays, `freshstart::` SNAP.
- **Iterative Orchestrator:** multi‑sweep with sanity gates (`min_promoted`, `stability_eps`, fail‑fast on persist=False/event emission), `iter_report::` SNAP.
- **Telemetry fix:** accurate `candidates_emitted`.

### v0.24.0 (P0 fixes + presets + 8‑universe stress)
- **RepoQuery:** safer listing by universe; geometry presets (ring/bimodal/line); promotion summaries into overlays; success criteria for multi‑universe runs.

### v0.30.9 → v0.30.11 (DNA hardening → staging worker → taxonomy/mannequin)
- **v0.30.9:** DNA policy & audit; staged candidates with TTL/decay; colorized tables; iter artifacts.
- **v0.30.10:** `dna_overrides` per scenario; **staging_worker** scanning family=staging; outcomes (`staging_promote/expired/still`); reports + bundles.
- **v0.30.11:** SNAP Taxonomy Manager; MDHG dual‑mode hashing; Repo Projector; mannequin universes; smoke artifacts.

### v0.30.20 → v0.30.21 (FS2 promotions, fast‑path gating, per‑universe policy, richer reports)
- **v0.30.20:** FS2 promotion audit SNAPs; auto‑flip to `baseline_candidate` on threshold; fast‑path `promote_if_equal`; consolidated HTML.
- **v0.30.21:** Archivist hooks emit `baseline_proposal`; per‑universe policy thresholds; fast‑path gating enforced; charts/badges.

---

## Pass‑1 — Ledgers (growth)
**Keyword ledger (additions):** `policy_hash`, `promotion_summary`, `ring|bimodal|line`, `elevator_threshold`, `compat_policy`, `freshstart::`, `iter_report::`, `mdhg_event::elevator`, `result::elevator`, `baseline_proposal`, `fast_path_mode`, `promote_if_equal`, `mannequin_universe`, `projector`, `taxonomy_manager`, `neg_beacons`, `dna_policy`, `staging_ttl_hours`.

**Alias map (additions):** Fast‑path ↔ quick review lane; Full‑path ↔ full evaluation; Inline hashing ↔ native; MDHG hashing ↔ structured.

**Facet bindings (additions):**
- **SNAP families/types:** `mdhg_event::elevator`, `result::elevator`, `universe_diff::overlay`, `audit::promotion_note`, `freshstart::`, `iter_report::`.
- **Policies:** per‑universe `policy_by_universe`; `dna_overrides` precedence; `neg_beacons` injection.

---

## Pass‑1 — Fit & connections (doc‑grounded)
- **Governance backbone:** PolicyBus aligns mannequin gate, controller pre/post, and promotion finalize; manifests unify provenance.
- **On‑demand materialization:** mannequin streams; controller/MDHG consume; overlays journal universe state; Ops Center reviews.
- **Iterative lifecycle:** FreshStart → Iterative → Staging worker → Promotion → FS2 re‑audit → Archivist (`baseline_candidate` → `baseline_proposal`).
- **Observability:** Trails/Reports/CSVs enable replay; promotion summaries attached to overlays for quick Ops review.

---

## Validation & TODOs — Pass‑1 adds
- **V3‑VAL‑001:** Assert presence of per‑version artifacts (bundles, reports, CSVs) in the running log index.
- **V3‑VAL‑002:** Ensure promotion summary SNAPs attach to overlays post‑iter runs.
- **V3‑VAL‑003:** Staging worker outcomes present in reports (promoted/still/expired) with TTL/decay.
- **V3‑OPS‑004:** Add stress‑suite acceptance counters (per‑universe stability, promotions, emissions).

---

## Next Pass Plan (active)
1) Continue D1 sweep to backfill any missing intermediate versions and artifacts; lock SNAP family/type taxonomy enumerations.
2) Cross‑check D2 phrasing against each milestone; only add clarifying notes when D2 introduces a reworded concept already in D1.
3) Expand ledgers (keywords/aliases/facets); flag collisions to E‑DBSU; attach weyl coverage entries per idea cluster.

---

## Pass‑2 — Sequential reconstruction (continued)
**Goal:** Fill gaps between v0.20 → v0.24 and v0.30.9 → v0.30.14; extract acceptance criteria, test coverage, CLI/docs, and policy defaults.

### A) Features & behaviors (doc‑grounded)
- **MDHG candidates & policies**: emits `mdhg_event::elevator` when `promote_elevators=True`; split blocking via `split_policy` (e.g., `family_deny:["fiction"]`); persistence + decay half‑life; tags (family/type/glyph). 
- **Overlays & diffs**: `attach_mdhg_overlay(um, universe, snap_id)` idempotent; `write_universe_diff(repo, universe, before, after)` writes `universe_diff::overlay` with before/after pointers; tests verify thresholds + family denial.
- **Promotion workflow**: `promote_elevators(repo, um, universe, threshold, compat_policy)` scans events → runs Ops (`SnapOpsCenter`) → writes `result::elevator` → updates overlays; tiny CLI; docs.
- **FreshStart + Iterative**: `fresh_start(...)` wipes mdhg/events/results/overlays and records `freshstart::`; `run_iterations(...)` loops with gates (`min_promoted`, `stability_eps`, fail‑fast under persist=False emission) and writes `iter_report::` SNAP.
- **DNA policy & staging**: `dna_policy.py` (required_families, min_lineage_depth, min_w5h_total, staging_ttl_hours); `promote_with_fastpath(...)` stages with TTL/decay; v0.30.10 adds **per‑scenario dna_overrides** and **staging_worker** with `staging_promote/expired/still`.
- **Taxonomy/Projector/Mannequin universes**: taxonomy manager enforces families/types (incl. *neg* flags); projector & mannequin universes introduced; smoke artifacts accompany releases.
- **RepoQuery + telemetry**: repo query helpers (universe/family/type), hash‑mode telemetry via `choose_mode(...)` recorded in Trails.

### B) Acceptance criteria & tests (extracted)
- **Controller invariants**: increment `candidates_emitted` only after successful save; assert `persist=False & elevators>0 ⇒ events>0`; add e2e for persist=False universe.
- **Promotion summaries**: emit `promotion_summary::<universe>::{ts}` and **attach to overlays** post‑run.
- **Policy block coverage**: include synthetic block (e.g., family=fiction split) and DocUniverse retune until elevators>0; record parameters.
- **Unit tests**: `tests/unit/test_elevator_fastlane_and_overlay.py`; `tests/unit/test_mdhg_persistence_policy.py`; docs for SNAPOps/Universes and Iterative orchestration.

### C) CLI & docs (extracted)
- `scripts/run_elevator_promotion.py` and `docs/ELEVATOR_PROMOTION.md`.
- `scripts/run_freshstart_iterations.py` and `docs/ITERATIVE_ORCHESTRATION.md` (flags: `--max-iters`, `--min-promoted`, `--stability-eps`).

### D) Stress toggles & readiness gates (extracted)
- Geometry presets: **ring/bimodal/line**; increase `mdhg_alpha`/`w5h_alpha`, add **sweep jitter**, tighten consensus; watch **Staging Queue TTL/decay**; factory picks MDHG when `node_count ≥ max_nodes`.

---

## SNAP family/type taxonomy — enumeration (draft from D1)
- **Events:** `mdhg_event::elevator`
- **Results:** `result::elevator`
- **Diffs:** `universe_diff::overlay`
- **Audits/Reports:** `freshstart::…`, `iter_report::…`, `promotion_summary::…`
- **Staging:** `staging::candidate` (with TTL/decay)
- **Taxonomy:** neg‑class flags for fiction et al. (enforced on stitch)

> Boundaries: this list mirrors D1 usage; it will expand as more families/types appear in later slices.

---

## Ledgers — Pass‑2 growth
**Keywords (additions):** `policy_hash`, `promotion_summary`, `ring`, `bimodal`, `line`, `elevator_threshold`, `compat_policy`, `freshstart::`, `iter_report::`, `mdhg_event::elevator`, `result::elevator`, `universe_diff::overlay`, `fast_path_mode`, `promote_if_equal`, `mannequin_universe`, `projector`, `taxonomy_manager`, `dna_policy`, `staging_ttl_hours`, `choose_mode`, `hash_mode`, `sweep_jitter`, `stress_circle_128`, `mdhg_alpha`, `w5h_alpha`, `consensus.max_imperf_best`.

**Aliases (additions):** `fast‑path` ↔ `quick review lane`; `full‑path` ↔ `full evaluation`; `inline hashing` ↔ `native`; `MDHG hashing` ↔ `structured`.

**Facet bindings (additions):**
- **SNAP families/types:** `mdhg_event::elevator`, `result::elevator`, `universe_diff::overlay`, `freshstart::`, `iter_report::`, `promotion_summary::`, `staging::candidate`.
- **Policies & reasons:** `safe_cube_requires_tags`, `family_deny`, `sentinel_pass`.
- **CLI/Docs facets:** elevator_promotion CLI/docs; freshstart_iterations CLI/docs.

---

## Validation — Pass‑2 adds
- **V3‑VAL‑004:** Promotion summary SNAP attaches to overlays (presence + counts match report tables).
- **V3‑VAL‑005:** Controller invariants enforced (`candidates_emitted` emission order; persist=False e2e).
- **V3‑VAL‑006:** Staging worker outcomes present with TTL/decay columns in reports.
- **V3‑VAL‑007:** Hash‑mode telemetry present in Trails for each iterative run.

## TODOs — Pass‑2 adds
- **V3‑OPS‑005:** RepoQuery exact‑universe filters integrated in Ops dashboards.
- **V3‑RUN‑002:** Stamp `policy_hash` into all Action SNAPs and overlay diffs.
- **V3‑MDHG‑005:** Persist compact index→room membership; add `rehydrate_index_map(N)`; regression tests (carryover, P1).

---

## Next Pass Plan (queued)
1) Continue D1 sweep forward from **Package #4 (open)** until the next non‑superperm pivot; then stamp mini‑hash and close #4.
2) Begin segmentation of **Package #5+** using pivot detectors; keep ledgers growing; attach Weyl counters per package.
3) Opportunistic backfill only when it unblocks accuracy (no broad polish yet).
4) Final pass (after segmentation complete): apply the **Backfill Protocol** below.

---

## Backfill Protocol — polish & finishing (v1, LOCK)
**When:** After all packages are segmented (plus opportunistic fixes during review where needed).

**What we backfill:**
- **Line‑range bookmarks:** Add `[Lstart–Lend]` markers to each package’s first/last blocks (in addition to char ranges). 
- **Mini‑hash audit:** Recompute and store `mini_hash(start,end,contents)` for each package manifest.
- **Cross‑links:** Link packages to artifacts (bundles/docs/fixtures), reason codes, and manifests; add package→SNAP family/type references.
- **Ledger reconciliation:** Resolve alias collisions into `alias_map.json`; ensure `facet_index.json` covers every keyword.
- **Weyl accounting:** Verify **8×2** coverage where applicable; record `weyl_coverage_snap` pointers in package manifests.
- **Taxonomy completeness:** Ensure every observed SNAP family/type appears in the **taxonomy enumeration** with at least one in‑text example.

**Validation hooks:**
- **V3‑BF‑001:** Package manifests include both **char and line** boundaries.
- **V3‑BF‑002:** `mini_hash` values present and consistent with stored boundaries.
- **V3‑BF‑003:** Each package has ≥1 cross‑link to artifacts/policies/reasons where applicable.
- **V3‑BF‑004:** Alias collisions resolved or quarantined to **E‑DBSU**.

**TODOs:**
| ID | Description | Priority | Status |
|---|---|---|---|
| V3‑BF‑001 | Emit line‑range bookmarks for all packages; store in package manifests. | P1 | Open |
| V3‑BF‑002 | Recompute `mini_hash` across all packages during final pass. | P1 | Open |
| V3‑BF‑003 | Reconcile `alias_map.json` collisions; update `facet_index.json`. | P1 | Open |
| V3‑BF‑004 | Verify 8×2 Weyl coverage where applicable; attach `weyl_coverage_snap` refs. | P1 | Open |


---

## Package #4 — enrichment (Superpermutation pilot)
**Char range:** [460093–… ) *(open; mini‑hash on close)*  
**Role ledger:** *(in progress)*  
**Observed (doc‑grounded):** greedy pilots at n=3..6; rising imperfect‑join counts with n; planned heavy test battery (overlap, coverage, length, failure taxonomy); explicit intent to integrate results back into AGRM/MDHG policies.  
**Topic anchors (auto):** superperm_pilot, greedy, overlap_histogram, laminates, AGRM_bias, measurement_plan, acceptance_criteria, fixtures, egan_sequences  
**Next pivot (expected):** non‑superperm artifact or explicit scope handoff; to be detected.  
**Weyl coverage:** pos=0/8, inv=0/8 (init)

## Ledgers — additions (Package #4 growth)
**Keywords (added):** universal_superpermutation_principle, cross_field_framework, superperm_pilot, imperfect_join_rate, overlap_histogram, coverage_metric, path_length, greedy_bias, laminate_merge, n3..n6, egan_fixture, acceptance_criteria.

**Facet bindings (added):**
- **Artifacts:** universal_superpermutation_principle.pdf → facet: artifact.doc; cross_field_framework → facet: artifact.code; egan_fixture → facet: artifact.fixture
- **Metrics:** imperfect_join_rate, overlap_histogram, coverage_metric, path_length → facet: metric

## Validation — Package #4 adds
- **V3‑SPM‑001:** Presence of pilot artifacts (logs/tables) for n=3..6; verify `imperfect_join_rate` trend captured.
- **V3‑SPM‑002:** Overlap histogram computed and snapshot saved; coverage and path length recorded.
- **V3‑SPM‑003:** Heavy test plan SNAP present with acceptance criteria and failure taxonomy.
- **V3‑SPM‑004:** AGRM/MDHG policy feedback notes emitted (no behavior change in this review phase).

## TODOs — Package #4 adds
| ID | Description | Priority | Status |
|---|---|---|---|
| V3‑SPM‑A01 | Extract pilot metrics table(s) and bind to Package #4 manifest. | P1 | Open |
| V3‑SPM‑A02 | Persist overlap histogram + coverage SNAP; attach lineage to fixtures. | P1 | Open |
| V3‑SPM‑A03 | Draft acceptance criteria block (from doc text) and store in plan SNAP. | P1 | Open |
| V3‑SPM‑A04 | Record AGRM/MDHG feedback notes and link to PolicyBus deltas (review‑only). | P2 | Open |

