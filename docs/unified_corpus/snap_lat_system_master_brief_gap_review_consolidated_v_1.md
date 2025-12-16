# SnapLat — Running Worklog & Design Ledger (live)

**Timestamp:** Aug 15, 2025 (America/Los_Angeles)

This ledger tracks **current/previous/future** work aligned to the Master Brief. It is the single source of truth for: inputs, diffs, decisions, tests, gates, and next actions. All changes in this session are appended here.

---

## Authorship & Attribution
- **Primary owner/author:** Nick Barker.
- **Attribution rule:** Credit Nick Barker on all SnapLat outputs and artifacts.
- **AI involvement:** Various AI systems may assist; note as *assistance* only when attribution is needed. Human ownership remains with Nick Barker.

---

## A) Inputs processed this checkpoint
- Zips: `SnapLat_Pass41.zip`, `SnapLat_Pass42.zip`, `SnapLat_Pass44.zip`
- Intake snapshot: `manus_zip_intake_snapshot.json`

**Artifacts produced** (paths available in-session):
- Coverage delta CSV → `sandbox:/mnt/data/snaplat_coverage_delta_report.csv`
- README/PLANS diffs index → `sandbox:/mnt/data/readme_plans_diffs_index.json`
- TODO checklist (gate‑mapped) → `sandbox:/mnt/data/snaplat_todo_checklist.md`

---

## B) Presence & Gaps (from scans)
**Clearly present** across latest drops: Trails; Policy/Safe‑Cube; Porter; MDHG ops; W5H/Beacons; RandomWalk/E8; ThinkTank; DTT; Assembly; Scout; SnapOps; Space (Universe/Lattice); Schemas/validators; CLI/CI/Docs.

**Missing / stubbed:**
- **FS2 runtime** (UniquenessRegistry, StepCache, Envelope) — *not detected as code in 41/42/44*; referenced in specs/tests only.
- **TPG surgery** (two‑opt/apply_surgery) — seams/mentions present; no concrete implementation in 41/42/44.

**Manus intake**: Large corpus indexed; many `LSDT` paths. Policy remains **reference‑only** for LSDT; advisory adoption is blocked unless explicitly whitelisted.

---

## C) Decisions @ this checkpoint
1. Adopt **SnapLat Personal Research License (Proprietary)** — owner retains all rights; no redistribution/commercial use without written permission.
2. Default classification = **CONFIDENTIAL–PERSONAL**; LSDT remains **REFERENCE‑ONLY**; add a *citation‑whitelist* path for exceptions (no advisory adoption).
3. Bind **Scoring weight changes** to Policy Hash; log diffs in Trails for reproducibility.
4. Approve **CiteSnap v0.1** as the citation format (URN + JSON/YAML) tied to Snap/Universe/Lattice/Weyl with “force node” edges.

---

## D) Tests executed (session)
- Feature family presence scan across 41/42/44 (tables attached in-session).
- README/PLANS diffs saved per file (index JSON above).

**Next test expansions (queued):** Weyl metamorphic multi‑lattice suite; coverage streaming degradation tests; FS2 canonicalization invariants; TPG surgery property tests.

---

## E) Backlog by Gate (extract)

**Gate B — Spec**
- N/Bridge FSM – transitions/demotion/merge; validator & proofs.
- E‑DBSU promotion math – corroboration vs. neg_conflict thresholds; property tests.

**Gate D — Budgets/Streaming**
- Document MDHG quotas (points/edges) & streaming `to_graph`; add degrade path tests.

**P0 (Immediate)**
- Implement FS2 runtime; invoke Envelope gate in MDHG wrapper under flag; CI check present.
- Scoring units + monotonicity properties; tie weights to Policy Hash changes.

**P1**
- TPG two‑opt/apply_surgery with revert guards + tiny demos.
- Trail schema enforcement in CI for all touched modules.
- Security: minimal roles (ops/analyst/system) + secrets handling notes.

---

## F) Ready‑to‑Run Next Steps (this session)
- Assign owners/dates to the **P0/P1 checklist** and create a short execution plan.
- (Optional) Pull selected `.diff` files into a single **delta digest** section here.
- (Optional) Whitelist specific Manus doc IDs for **citation‑only**.

> To trigger: say **“assign owners”** with names/handles, **“digest diffs”** with file list, or **“whitelist these Manus docs”** with paths.

---

## G) Links & Artifacts (quick)
- Coverage delta CSV → `sandbox:/mnt/data/snaplat_coverage_delta_report.csv`
- Diffs index JSON → `sandbox:/mnt/data/readme_plans_diffs_index.json`
- TODO checklist → `sandbox:/mnt/data/snaplat_todo_checklist.md`
- **License** → `sandbox:/mnt/data/LICENSE_SnapLat_PROPRIETARY.txt`
- **Confidentiality Policy** → `sandbox:/mnt/data/CONFIDENTIALITY_POLICY.md`
- **CiteSnap spec** → `sandbox:/mnt/data/CITATION_CITESNAP.md`

---

# SnapLat — System Master Brief & Gap Review (Consolidated v1.0)

> **Purpose**: One coherent, senior‑engineer‑ready brief that fuses your Intake v1 (incl. Passes 9–48) into a stable reference before refactor/wrapper work. It captures the operating model, core formalism, current runtime surfaces, tests, governance, and a crisp gap→proof plan with decision gates.

---

## 0) Executive Summary

**Intent**: Deterministic, explainable reasoning at bounded complexity via explicit N‑levels, shell expansion, E8/Weyl evaluation, and compression into **Snaps** with full **Trails** lineage. Cross‑domain composition is achieved by **Bridges** and safe governance through **Safe‑Cube**, **Policy Hash**, **Porter/Mannequin** custody.

**What exists (from your staged passes)**
- **Contracts** (.pyi) are emitted for core surfaces (MDHG/AGRM/TPG, Trails, W5H/Beacons, FS2, Policy/Mannequin).  
- **Runtime seeds & wrappers**: MDHG ops (`to_points`, `to_points_n`, `to_graph`, `promotion_breakdown`), ThinkTank, DTT (dry‑run), Assembly Line, SnapOps (policy‑aware orchestrator), Scout (rendezvous + advisory hints), Beacons registry + loader, W5H alignment, Safe‑Cube DSL, Porter (custody + DLQ + inspector), Archivist v1 (expand/contract), E‑DBSU lifecycle, Universe/Lattice tags & Weyl stubs, Random‑Walk (E8‑aware) + MCMC/SMC scaffolds, metrics (coverage/shelling/neg_conflict), schemas/validators, CI setup, README and experiment harnesses.
- **Observability**: Trails schema + validator, ring buffer, query helpers, chain‑of‑custody fingerprints (agent/snap), policy‑stamped decisions.

**Top strengths**
- Clear contracts + schema guards; deny‑by‑default governance; test‑first wiring; orchestration proved E2E (ThinkTank→DTT→Assembly) with DLQ behavior.
- Coverage gating + “Approach‑phase” consolidation; advisory MC/RW hints gated by diagnostics and policy; LSDT treated reference‑only with explicit policy blocker.

**Critical gaps** (pre‑refactor blockers)
1) **Formal N/Bridge FSM** (transitions/merge/demotion) and **E‑DBSU** promotion math (confidence update).  
2) **Scoring panel** (config‑calibrated, W5H + (neg) beacons + heat/bridge terms) with principled units + invariants.  
3) **FS2 canonicalization & Envelope** truth table wired to MDHG expansion gates.  
4) **Trail taxonomy enforced project‑wide** (all modules emit begin→append→finalize with required fields).  
5) **Quotas & complexity budgets** for MDHG/TPG (streaming mode & degradation paths).  

**Decision gates**
- **Gate A — Contract Freeze**: §11 surfaces emitted as .pyi + Trail schema bound ✅ (met).  
- **Gate B — Spec Complete**: N/Bridge FSM, E‑DBSU lifecycle, Snap v1 schema ✅ (specs exist) → needs proofs.  
- **Gate C — Harness Ready**: Weyl metamorphic tests + Trail replay + toy universes ✅ (exists as stubs/fixtures) → expand coverage.  
- **Gate D — Budgets Pinned**: MDHG/TPG quotas + coverage watermark policy → partial; finalize before refactor.  

**Readiness**: Safe to proceed with **wrapper‑first refactor** once §1–§5 below are satisfied (P0/P1 proofs).  

---

## 1) First Principles & Design Maxims
- **Everything is a lattice**: all data expandable; lattices nest; **Bridges** connect lattices (emerge at N≥5).  
- **Trace → compress**: expand shells to reason; compress into **Snaps**; maintain lineage.  
- **Eight‑way truth**: at N=5 evaluate 8 Weyl points across interior lattices; Safe‑Cube sits central.

**N‑levels (working semantics)**
- **N=0**: Unchangeables + baseline policy/no touch.  
- **Bridges**: base weight 0.25 (convention); **double‑known → N=0.5**, **double‑unknown → E‑DBSU**.  
- **N=1..4**: increasing saturation; N=4: one perfect outcome.  
- **N=5**: ambiguity (exactly 8 outcomes assumed); bridges crystallize.

> **Proof obligations**: Formal FSM for N/Bridge transitions (incl. demotions), constructive E8 mapping, counterexample harness for “exactly 8”.

---

## 2) Operating Model (AGRM, Arms, Rendezvous, Ticks)
- **AGRM** chooses regions; **Main arms** follow plan; **Scout arms** counter‑probe (spawn on N≥4 hardness or resource slack).  
- **Ticks**: pause zones publish to **ThinkTank**; ThinkTank returns guidance for next tick.  
- **Greedy vs JIT**: Greedy safe at N≤2; at N≥3 pre‑inform **MDHG** and keep scouts warm.

**Always‑on services**: ThinkTank (panel critique), **DTT** (sandbox/dry), **Assembly Line** (staging); all guarded by Safe‑Cube; policies hashed & logged to Trails.

---

## 3) Data Intake & Knowledge Growth
- **MORSR + Wave Pool**: RAG‑like parsing with lattice/shell awareness; idea packaging; lexicon growth; probability maps steering review.  
- **Inclusiveness**: expanded keyword families & inverse/negation search to avoid narrow spaces.

---

## 4) Graph & Scoring Core (MDHG) + Orchestrator
- **MDHG ops**: `to_points`, `to_points_n`, `to_graph` (quotas), `promotion_breakdown` (degree + alignment; config‑driven weights when flag set).  
- **Caches/Uniqueness**: step/beacon caches; FS2 hooks to enforce uniqueness and canonicalization (stubbed).  
- **AGRM/TPG**: `TPGConfig` locked; 2‑opt surgery seams pinned (no‑ops until enabled).  
- **Metrics**: coverage, redundancy, neg_conflict (bridge‑aware), shelling checks.

> **Proof obligations**: Type‑precise MDHG API with complexity budgets; calibrated scoring panel with invariants; TPG surgery protocol.

---

## 5) Observability & Governance
- **Trails**: schema‑enforced begin/append/finalize; viewer/query helpers; ring buffer; chain‑of‑custody (agent/snap fingerprints).  
- **W5H & Beacons**: cosine alignment; named registry + loader; neg‑beacons penalize.  
- **Safe‑Cube DSL**: deny‑by‑default; file‑backed rules; decisions emit Trails + Policy Hash.  
- **Porter**: custody log, retries, DLQ, inspector; payload chain enriched with fingerprints.

---

## 6) Snaps, Archivist, E‑DBSU, Universes/Lattices
- **Snap v1**: ID, glyph (3‑word target at N=5), context, n‑level, shell, Weyl index/universe_ref (optional), trail_id, fingerprint.  
- **Archivist v1**: deterministic contract/expand; Safe‑Cube‑guarded.  
- **E‑DBSU**: NEW → TRIAGED → PROMOTED/DEMOTED → ARCHIVED; evidence signals (corroboration vs neg_conflict).  
- **Universe/Lattice**: hemisphere tags, `weyl_index` stubs; ThinkTank stamps universe_ref; Scout rendezvous uses hemispheres.

> **Proof obligations**: Snap algebra (compose/decompose laws), E‑DBSU promotion math, glyph collision policy and namespaces.

---

## 7) Interfaces (Minimum Contracts) — *frozen names*
- **MDHG**: `to_points`, `to_points_n`, `to_graph`, `promotion_breakdown`  
- **AGRM/TPG**: `TPGConfig`, surgery flags  
- **Trails**: `begin_trail`, `append_event`, `finalize`, `rendezvous`, validators  
- **W5H/Beacons**: `compute_w5h_alignment`, `BeaconsRegistry` (+loader)  
- **FS2**: `UniquenessRegistry`, `StepCache`, `Envelope.allows`  
- **Governance**: `POLICY_HASH`, `SafeCubeSentinel.allow(op, ctx)`, `Porter.deliver(payload, to=...)`  
- **Space**: `Universe`, `Lattice`, `tag_points`, `weyl_index`

---

## 8) Readiness Matrix
Legend: **Spec ✓**, **Contract ✓**, **Runtime seed △**, **Tests △**, **Integrated ★**

| Family | Spec | Contract | Runtime | Tests | Integrated |
|---|---:|---:|---:|---:|---:|
| Trails (schema/validator) | ✓ | ✓ | ✓ | ✓ | ★ |
| Policy/Safe‑Cube | ✓ | ✓ | ✓ | ✓ | ★ |
| Porter (custody/DLQ/inspect) | ✓ | ✓ | ✓ | ✓ | ★ |
| MDHG ops | ✓ | ✓ | ✓ | ✓ | ★ |
| W5H/Beacons | ✓ | ✓ | ✓ | ✓ | ★ |
| TPG (config/surgery seams) | ✓ | ✓ | △ | ✓ | △ |
| Snap/Archivist | ✓ | ✓ | ✓ | ✓ | ★ |
| E‑DBSU | ✓ | ✓ | ✓ | ✓ | ★ |
| ThinkTank/DTT/Assembly | ✓ | ✓ | ✓ | ✓ | ★ |
| Scout (rendezvous/advisory) | ✓ | ✓ | ✓ | ✓ | ★ |
| FS2 (canon/envelope) | ✓ | ✓ | △ | △ |  |
| Metrics (coverage/shelling) | ✓ |  | ✓ | ✓ | ★ |
| Space (Universe/Lattice) | ✓ | ✓ | ✓ | ✓ | ★ |

> **Blocking deltas**: FS2 wiring (canon/envelope), TPG surgery activation, scoring calibration.

---

## 9) Gap Review → Proof Obligations (actionable)

**P0**
1. **N/Bridge FSM** — state/transition spec → implement validator; demotion rules; Trail proofs.  
2. **MDHG Quotas & Budgets** — document O() + practical caps; streaming mode; degradation plan.  
3. **Scoring Panel** — config‑driven weights with units; monotonic property tests; fixtures + grid sweeps; bind in wrapper under flag.  
4. **Trail Enforcement** — turn tests‑only validator into enforced check in CI (not runtime‑hot); require begin→append→finalize in touched modules.  

**P1**
5. **FS2 Canonicalization** — invariants + rotation/reversal relabel; Envelope truth table; wire into MDHG expansion gate.  
6. **TPG Surgery** — entry/exit & guardrails; small synthetic improvement demos; Trails + revert on regress.  
7. **E‑DBSU Promotion Math** — confidence update (likelihood/Bayes or hash‑strength proxy); demotion on neg_conflict dominance.

**P2**
8. **Security profile** — minimal roles; secrets; audit alignment with Trails; multi‑tenant story sketch.  
9. **Performance** — cost models for scouts and edge quotas; MC advisory budgets.

---

## 10) Milestones & Gates

**M1 — Wrapper‑first Sandbox (now → short)**
- Finalize scoring config path (flagged) and calibrate on fixtures.  
- Activate **TPG** surgery with revert switch; add property metas.  
- Wire **FS2** canon/envelope into MDHG gate; kill dup expansions.  
- CI: enforce Trail schema on touched modules.

**Gate M1→M2**: P0 proofs green; coverage watermark/approach‑phase policies documented.

**M2 — Determinism & Budgets (short → mid)**
- N/Bridge FSM runtime + demotion tests; E‑DBSU promotion math bound.  
- Budgets/quotas pinned; streaming mode + graceful degradation; perf dashboards.  
- Expand metamorphic E8 harness; cross‑lattice invariants.

**Gate M2→M3**: No uncontrolled O(N²) hotspots in CI perf samples; replayable Trails for golden cases.

**M3 — Refactor & Interop (mid)**
- Snap algebra + glyph collision policy/namespace.  
- Superperm interop elevated (still non‑heavy by default).  
- Config injection replaces hardcoded thresholds; policy knobs surfaced in CLI; docs hardened.

---

## 11) Risk Register (top 8)
1. **Eight‑way assumption brittle** in certain domains → add counterexample harness; fail closed.  
2. **Scoring leakage** from config drift → bind weights to Policy Hash + Trails diff.  
3. **Dup/near‑dup** expansions without FS2 gate → prioritize Envelope wiring.  
4. **Trail drift** under new modules → CI guard now, not later.  
5. **Budget overruns** in dense graphs → streaming mode + quotas + degrade path.  
6. **Advisory hints mis‑adopted** → keep adoption opt‑in + strict tag requirement.  
7. **LSDT bleed‑through** into core → keep policy.blocked.lsdt + tests.  
8. **Security** gaps (authz/secrets) → add minimal role model + audit hooks aligned to Trails.

---

## 12) Approval Checklist (sign‑off to start refactor)
- [ ] §7 contracts stable (versioned) and referenced by all new code.
- [ ] Trail schema bound in CI; golden trail replay passes.
- [ ] Scoring panel config + fixtures committed; property tests green.
- [ ] FS2 canon/envelope wired to MDHG expansion gates.
- [ ] Quotas/budgets documented; streaming fallback demonstrated on toy universe.
- [ ] Safe‑Cube rules reviewed; deny‑by‑default paths covered by tests.

---

## 13) Open Questions (parked)
- Conditions to extend 8‑way Weyl mapping across multiple interior lattices without contradiction.  
- Confidence updates for bridges across independent lattices.  
- Defining N=4 saturation under observational incompleteness.  
- Glyph collisions across universes: namespace & backcompat.  
- Neg‑beacons vs Safe‑Cube denials (filter vs hard fail).

---

## 14) Glossary (working)
**AGRM** – region orchestrator.  
**MDHG** – multi‑dimensional hash/graph core.  
**TPG** – path generator (2‑opt seams).  
**Trails** – immutable lineage logs.  
**W5H** – who/what/when/where/why/how alignment vectors.  
**Beacons / Neg‑beacons** – anchors & penalties.  
**Safe‑Cube** – central policy sentinel & validation zone.  
**Porter / Mannequin** – custody & isolation surfaces.  
**E‑DBSU** – emergent DB for double‑unknown bridges.  
**MORSR / Wave Pool** – intake tuned to lattice semantics.  
**Snaps / Archivist** – compression + deterministic rehydration.  
**Approach‑phase** – near‑watermark consolidation path.

---

## 15) Appendix — Cross‑walk to Your Passes
- **Pass 9–16**: Contracts emitted; Trails schema; TPGConfig locked; scoring fixtures; W5H/Beacons wired; rendezvous events; beacons loader.  
- **Pass 17–21**: Safe‑Cube DSL; Porter custody; Archivist v1; E‑DBSU runtime; inclusiveness sweep.  
- **Pass 22–26**: ThinkTank, DTT, Assembly; Porter E2E orchestration + DLQ; superperm interop shim.  
- **Pass 27–32**: MDHG ops; Universe/Lattice tags; fingerprints & chain; Scout rendezvous; query helpers; inspector; scoring surface.  
- **Pass 33–38**: SnapOps orchestrator + CLI; policy knobs; coverage gating & approach‑phase; experiments + plots.  
- **Pass 39–42**: MC toggle; E8‑aware RW; MCMC/SMC diagnostics; advisory adoption (opt‑in); schema validators.  
- **Pass 43–48**: Coverage empirics harness; neg_conflict improvements; scout queue adoption; intake (authorship/segmenter/packager/lexicon/code); LSDT guardrails; mixed corpus triage.

---

### End of Consolidated Brief — ready for sign‑off and refactor planning.



## H) Policy Hash & Validators (executed now)
- **Policy Hash minted:** PH-acca5e4f34c68289 (written to `sandbox:/mnt/data/snaplat_policy_hash.txt`).
- **Header stub** created for quick paste into files → `sandbox:/mnt/data/header_stub.yaml`.
- **Header scan** ran across Pass41/42/44; full report → `sandbox:/mnt/data/header_scan_report.csv`.
- **Manus corpus** indexed; whitelist candidates prepared → `sandbox:/mnt/data/manus_index.csv`, `sandbox:/mnt/data/manus_whitelist_candidates.json`.

> Next: if you’d like, I can re‑package any of the zips with headers injected (classification + policy hash) and attach the repacked archives here.



## I) Repacked Archives & Whitelist (executed now)
- **Repacked with headers** (classification + policy hash inserted where missing):
  - `sandbox:/mnt/data/SnapLat_Pass41+headers.zip`
  - `sandbox:/mnt/data/SnapLat_Pass42+headers.zip`
  - `sandbox:/mnt/data/SnapLat_Pass44+headers.zip`
- **Summary table** available in-session: *Repack summary (headers injected)*.
- **Manus citation whitelist** adopted → `sandbox:/mnt/data/manus_whitelist.json` (policy‑bound to current hash). 
- **Browse allowed**: I will use web lookups for public facts/citations as needed and record CiteSnap URNs in outputs.



## J) P0 Package (delivered now)
- **SnapLat_P0_package.zip** → `sandbox:/mnt/data/SnapLat_P0_package.zip`
  - `src/plugins/edges/superperm/fs2.py` — FS2 runtime (UniquenessRegistry, StepCache, Envelope)
  - `examples/mdhg_envelope_gate.py` — example gating path
  - `tests/test_fs2_runtime.py` — uniqueness & envelope tests
  - `PLANS/MDHG_SCORING_UNITS.md` + `tests/test_scoring_units.py`
  - `PLANS/MDHG_BUDGETS_STREAMING.md` + `src/agrm/mdhg/streaming.py`
  - `PLANS/N_LEVEL_BRIDGE_FSM_v0.1.md` + `PLANS/E_DBSU_PROMOTION_MATH_v0.1.md`
  - `scripts/check_snaplat_header.py` + `.github/workflows/ci.yml`

> Next ready: wire the Envelope gate into your MDHG wrapper and toggle via env flag; add a tiny scoring weight file bound to the current Policy Hash.



## K) Envelope Gate & Scoring Config (executed now)
- **Envelope-gated wrapper** → `src/agrm/mdhg/wrapper_envelope.py` (enable via `SNAPLAT_USE_FS2_ENVELOPE=1`).
- **Scoring config** → `PLANS/mdhg_scoring.yaml` (bound to current Policy Hash) + loader at `src/agrm/mdhg/scoring_config.py` (logs diffs to Trails file if present).
- **Tests**: `tests/test_mdhg_envelope_integration.py`, `tests/test_scoring_config_binding.py`.
- **Download**: `sandbox:/mnt/data/SnapLat_P0_wiring.zip`.

> Adoption plan: wire wrapper_envelope into your MDHG adapter behind a feature flag; load weights via `scoring_config.load_weights(...)` and record an applied config Trail with the active Policy Hash.



## L) TPG Surgery + Streaming Harness (executed now)
- **TPG 2‑opt surgery** → `src/tpg/surgery.py` with `path_cost`, `two_opt`, and `apply_surgery` (revert‑on‑regress guard). Tests: `tests/test_tpg_surgery.py`.
- **Streaming harness** → `experiments/streaming_harness.py` with sweep results table attached in‑session (“Streaming harness results”). Plot saved to `sandbox:/mnt/data/streaming_edges_vs_n.png`.
- **Download**: `sandbox:/mnt/data/SnapLat_P0_crit2.zip`.

> Next: wire surgery behind a policy flag in SnapOps/TPGConfig and emit Trails for `surgery.begin` / `surgery.applied` / `surgery.reverted`. Add quota‑hit Trails to the streaming path.



## M) Design Idea Review — “Massive Nested DB as Single SSoT” (deep critique)
**Proposal (rephrased):** Persist *all* system data — Snaps, lattice node contents, etc. — into a deeply classified, nested database (system/group/set/family/generational/… facets), tailored schemas per system, and route **all reads** through it via agency‑based handoffs and chained, gated checkpoints.

### 1) What’s strong about this idea
- **Unified provenance:** Perfect fit for SnapLat’s Trails/Snaps; one logical SSoT improves replay, audit, and reproducibility.
- **Governance leverage:** Safe‑Cube + Policy Hash + Porter custody become first‑class: every pull crosses explicit gates with evidence in Trails.
- **Bridge/E‑DBSU fit:** Bridges (weights, N‑state) and double‑unknowns slot naturally as graph edges + lifecycle states inside the store.
- **CiteSnap alignment:** URNs key cleanly into the store; universes/lattices/Weyl can be indexed and enforced.

### 2) Principal risks (and mitigations)
1) **Monolith blast radius** — a single physical “massive” DB couples heterogeneous workloads, risking outages and slow queries.
   - *Mitigate:* Make it a **single logical SSoT**, not a single physical DB: event‑sourced core (append‑only Snaps/Trails, CAS by hash) + **derived indexes** (graph for MDHG, doc/relational for metadata, search for text). Keep them in sync via Trails/CDC.
2) **Schema explosion & drift** — “tailored per system” invites incompatible schemas.
   - *Mitigate:* A **Schema Registry** with versioned descriptors (JSON Schema/Protobuf), policy‑bound to the Policy Hash; evolution rules + migrations; per‑family invariants and property tests.
3) **Cross‑cutting queries over deep nesting** — joins and multi‑facet filters get slow/unreliable.
   - *Mitigate:* **Facet catalog** separated from payloads; poly‑hierarchical tags (system, group, set, family, generation, universe, lattice, Weyl). Precompute **materialized views** for hot paths; add Beacons indices.
4) **Write amplification & latency** — forcing every read through gated pipelines can stall work.
   - *Mitigate:* Split **control vs data** planes: gates/decisions live in Trails; reads hit read‑optimized **derived stores**. Enforce policy at the **service layer**, not via blocking DB triggers.
5) **Consistency / deadlocks** — chained checkpoints across agencies can deadlock.
   - *Mitigate:* Use **idempotent, append‑only events**; time‑boxed gates with failure paths (E‑DBSU triage, salvage‑retry); approach‑phase consolidation as a safe escape hatch.
6) **Security & multi‑tenancy** — per‑node classification + ACLs get complex; leakage via tags.
   - *Mitigate:* Row/edge‑level security keyed to **classification facets**; default **CONFIDENTIAL–PERSONAL**; audit via Trails; Secrets/roles minimal model.
7) **Cost & performance** — O(N²) risks and heavy graph queries.
   - *Mitigate:* **Streaming to_graph** with quotas; FS2 Envelope gate to kill dup/near‑dup expansions; quotas surfaced in Trails; coverage watermark & consolidation.

### 3) Recommended target architecture (SnapLat‑native)
- **Core (immutable):** Content‑addressed **Snaps** + **Trails** (append‑only), classification metadata as facets; Snap fingerprints & Policy Hash embedded.
- **Derived planes:**
  - **Graph (MDHG)** for nodes/edges/bridges with N‑state; RW/MCMC/SMC advisories; FS2 gate at ingest.
  - **Catalog/Registry** (W5H, Beacons/Neg‑beacons, Schema Registry, Universe/Lattice registry).
  - **Search/doc store** for text bodies and code snippets (MORSR/WavePool friendly).
- **Namespace:** **CiteSnap URNs** across all planes; Trails record cross‑plane projections.
- **Service layer:** **SnapOps/Porter/Safe‑Cube** enforce gates; policy knobs decide hard/soft gating; E‑DBSU lifecycle on ambiguous bridges.

### 4) What you must spec or prove before committing
- **Schema Registry contract** + evolution rules; versioned by Policy Hash; CI checks for breaking changes.
- **Facet catalog** (system/group/set/family/generation/universe/lattice/Weyl/bridges) with poly‑hierarchy and conflict rules.
- **Projection invariants:** Rebuild derived indexes from Trails (determinism tests, golden replays).
- **Performance budgets:** Quotas for to_points/to_graph; streaming thresholds; Trail signals for quota hits.
- **Security model:** Row/edge‑level ACLs using classification; deny‑by‑default; audit in Trails; export controls (no PUBLIC unless marked).

### 5) Do / Don’t (concise)
- **Do:** Keep the **immutable core** small and clean; project outward. Bind everything to Policy Hash; embed CiteSnap.
- **Don’t:** Put all workloads in one physical DB or enforce policy purely with DB triggers. Avoid per‑system ad‑hoc schemas without registry.

### 6) MVP slice to validate
1) Ingest → **Snaps+Trails** (CAS) with classification facets.
2) Project to **MDHG** (with Envelope gate) and **Catalog** (Beacons/W5H/Schema entries).
3) Run a query battery (8‑way Weyl eval, coverage gate, advisory adoption opt‑in) against the derived stores.
4) Rebuild from Trails to prove determinism. Measure latency, quotas, and gate efficacy.

### 7) Kill/Go criteria
- **Go:** Deterministic rebuilds; <X ms median query on hot views; gate/advisory events logged with policy‑bound diffs; no O(N²) regressions in CI.
- **Kill or Re‑scope:** Facet/tag drift beyond Y%; frequent deadlocks; coverage gate trips > Z% under normal loads; failed rebuilds.

**Verdict:** The idea is **directionally right** if implemented as a **single logical SSoT** with an **event‑sourced core** and **federated derived indexes**. SnapLat’s existing Safe‑Cube, SnapOps, FS2, E‑DBSU, and Trails make this feasible. The main failure mode is a physically monolithic DB with ad‑hoc per‑system schemas; avoid that by adopting the registry + projection plan above.



## N) Candidate Bake‑off (executed now)
**Protocol:** 3 initial architectures → pick winner → generate 2 variants → pick best → final refinement. Synthetic harness uses FS2 gating + streaming quotas + Trails completeness proxies. (Honest note: these are **proxies**, not real perf numbers.)

**Round 1 (3 candidates):** results table is in your workspace ("Round 1 — Candidate results").

**Round 2 (2 variants of the winner):** results table is attached ("Round 2 — Variants of winner").

**Round 3 (final refinement):** results table is attached ("Round 3 — Final refinement").

Artifacts:
- CSVs → `sandbox:/mnt/data/bakeoff_v1/round1_results.csv`, `sandbox:/mnt/data/bakeoff_v1/round2_results.csv`, `sandbox:/mnt/data/bakeoff_v1/round3_results.csv`
- Plots → included in `sandbox:/mnt/data/bakeoff_v1/snaplat_bakeoff_results.zip`
- Bundle → `sandbox:/mnt/data/bakeoff_v1/snaplat_bakeoff_results.zip`

> Next: If you want, I can convert the winning config into a concrete **Design Slice** (schemas + projections + gates) and generate stub code + tests to match, then run a tiny E2E on sample data.



## O) System‑level E2E Harness (executed now) — Findings & Needs
Artifacts:
- **Summary table** → `sandbox:/mnt/data/e2e_v1/e2e_summary.csv`
- **Trails log** → `sandbox:/mnt/data/e2e_v1/trails_log.jsonl`
- **Bundle** → `sandbox:/mnt/data/e2e_v1/e2e_artifacts.zip`

Snapshot highlights (synthetic but end‑to‑end stitched):
- FS2 gate rejected duplicates (OK).
- MDHG streaming produced edges with quotas; **coverage** fell below a strict watermark ⇒ coverage gate fired and routed to E‑DBSU (state TRIAGED/DEMOTED depending on signals). 
- TPG surgery improved or held path cost (no regress). 
- Scoring config used defaults (loader module not on path here) — treated as OK for this run.
- LSDT guard fired (OK: reference‑only content detected → advisory adoption blocked).

**What the system still needs (prioritized):**
1) **Schema Registry (P0):** Versioned, policy‑bound schemas + CI guard; per‑family invariants; migrations.
2) **Derived‑store projectors (P0):** Deterministic rebuild from Trails into Graph/Catalog/Search; golden replay scripts.
3) **SnapOps wiring (P0):** Policy flags for surgery/advisory/coverage; emit `surgery.begin/applied/reverted` and `quota_hit` Trails.
4) **Coverage config (P0):** Watermark + approach‑phase consolidation wired into this harness; salvage/triage loop (already specced; wire it here).
5) **Scoring loader path (P1):** Ensure runtime loader is importable; add applied‑diff Trails and mismatch tests.
6) **Security model (P1):** Row/edge ACLs bound to classification facets; deny‑by‑default checks.
7) **Beacons/W5H integration (P1):** Real vectors + named beacons into promotion_breakdown for scoring consistency with citations.
8) **ThinkTank/DTT/Assembly (P1):** Hook this harness to run the dry‑run and produce WorkOrders; DLQ on failure.
9) **Trail query/Viewer (P2):** Quick reconstruction by agent_fp/snap_fp; minimal web UI or report generator.
10) **Performance budgets (P2):** Bench actual N and quotas; add streaming backpressure and memory guard.

> I can convert items 1–4 into code/tests immediately (Schema Registry stub + projector skeleton + SnapOps policy flags in this harness + coverage consolidation), then re‑run this E2E and append results here.



## P) P1 Implementation — Registry, Projectors, Policy, Consolidation (executed now)
**What’s delivered**
- **Schema Registry (policy‑bound)** → `schemas/registry.json` (families: Snap, TrailEvent, GraphNode, GraphEdge) + `scripts/check_schemas.py`.
- **Deterministic projectors** from Trails → Graph/Catalog/Search:
  - Graph projection: nodes/edges derived from Trails (fs2 rejects, mdhg summary, coverage status).
  - Catalog projection: modules + (neg) beacons.
  - Search projection: toy event token index.
- **SnapOps policy wiring** (flags: enable_surgery, adopt_advisory, require_coverage, watermark, approach_margin, shell_checks) and **coverage+consolidation loop** (Archivist round‑trip check) with Trails `snapops.coverage_gate` / `snapops.consolidate_ok`.

**Artifacts**
- **Bundle:** `sandbox:/mnt/data/SnapLat_P1_core.zip`
- **E2E v2 Trails:** `sandbox:/mnt/data/e2e_v2/trails_v2.jsonl`
- **Projections:** `sandbox:/mnt/data/e2e_v2/graph_projection.json`, `sandbox:/mnt/data/e2e_v2/catalog_projection.json`, `sandbox:/mnt/data/e2e_v2/search_projection.json`
- **Dashboard table:** “Projections & Coverage” (in‑session)

**Notes**
- Graph/Catalog/Search projectors are deterministic for the same Trails input (golden replay ready). 
- Coverage is evaluated against a strict watermark (0.90) with approach‑phase margin and shell checks; results are logged in Trails.

> If you want, I can now wire the P1 projectors to consume **your real Trails** from earlier runs and produce a side‑by‑side projection report, plus a golden replay checksum to validate determinism.



## Q) Real Trails Projections — Side‑by‑Side (executed now)
- Ran projectors on **E2E_v1** and **E2E_v2** Trails.
- **Determinism:** projections are deterministic (hash‑stable reruns) for both logs.
- **Validation:** GraphNode/GraphEdge records conform to the Registry requirements.
- **Coverage:** extracted status and value where present (gate vs ok vs consolidate_ok).

**Artifacts**
- **Projection report (zip):** `sandbox:/mnt/data/SnapLat_P1_projections_report.zip`
- **Workspace table:** *Real Trails — Projection Report* (counts, validity, determinism, coverage status/value)

> Next: I can wire these projections into a simple rendered report (markdown/HTML) and add a checksum‑based golden replay test that fails if projections ever change for the same Trails input.



## R) P2 — Security, Scoring, Orchestration & Viewer (executed now)
**What’s in this pass**
- **Row/edge ACL (deny‑by‑default)** with Trails (`acl.allow/acl.deny`) keyed to classification facets (default: CONFIDENTIAL‑PERSONAL) and roles (viewer/analyst/ops).
- **Beacons + W5H scoring integration** (cosine alignment) feeding `promotion_breakdown`‑style output (w5h/neg/hot/bridge/score) with current Policy Hash.
- **ThinkTank → DTT → Assembly orchestrator** that yields a WorkOrder; DLQ hooks in place.
- **Trail Viewer** — generated HTML summary over **v1 + v2 + new** Trails.
- **Streaming tuner** — suggests edges_per_node_max and total_edges_max settings to hit coverage targets.

**Artifacts**
- **P2 bundle** → `sandbox:/mnt/data/SnapLat_P2_core.zip`
- **Trail Viewer** → `sandbox:/mnt/data/p2_outputs/trail_viewer.html`
- Workspace table: *Streaming tuner candidates* (top settings).

**Run outcome**
- Orchestration **OK**. WorkOrder modules: `mdhg`, `tpg`, `superperm`. 
- Scoring sample: w5h≈0.95, neg≈0.14, hot≈0.77 → **score≈0.726** (weights policy‑bound).

**Next critical items**
1) Enforce ACL checks in projectors + MDHG ingest (not just SnapOps).
2) Swap the toy hotness with real graph metrics (degree/edge heat), and wire named beacons from your `PLANS/beacons.json` if present.
3) Add DLQ test path through orchestrator and verify Assembly is skipped.
4) Optional: export Trail Viewer as a printable report (HTML/PDF) with op histograms and event timelines.



## S) P2 Follow‑ups — ACL Enforcement, Real Heat, Beacons Paths, DLQ, Printable Report (executed now)
**Hardened**
- **ACL at MDHG ingest** (`mdhg.ingest`) and **projector writes** (deny‑by‑default; Trails `acl.allow/acl.deny`).
- **Hotness (real)** derived from **degree heat** on the streaming graph (normalized average degree, capped by epn).
- **Beacons loader** searches multiple paths (`/mnt/data/PLANS/beacons.json`, pass zips, etc.); seeds defaults if none found; Trails `beacons.loaded`.
- **DLQ path** validated: DTT failure triggers `snapops.dlq`; Assembly is skipped.
- **Printable Trail report** with module histograms and Policy Hash footer.

**Artifacts**
- **Bundle:** `sandbox:/mnt/data/SnapLat_P2_followups.zip`
- **Printable report:** `sandbox:/mnt/data/p2_outputs_b/trail_report_printable.html`
- **Status table:** *P2 follow-ups — status* (in‑session)

**Next (queued, still critical)**
1) Apply ACL checks to **derived projections rebuild** job and CI (fail on deny). 
2) Backfill **Graph hotness** into `promotion_breakdown` in the MDHG wrapper, not just the harness. 
3) Add **role‑based views** in the Trail Viewer (filter by classification). 
4) Wire the **coverage watermark tuner** into SnapOps Policy so E2E runs can pick configs from the tuner table.



## T) P2c — CI ACL on Projections, Hotness-in-Scoring, Role Views, Tuner→Policy (executed now)
**CI & Enforcement**
- **CI rebuild job** for projections runs under ACL (role=analyst, classification=CONFIDENTIAL‑PERSONAL). Fails fast on deny; current run: **PASS**.

**Scoring**
- `promotion_breakdown` now **uses real hotness** (degree heat from graph) → score increases monotonically with connectivity (see workspace table "Promotion breakdown — hotness monotonicity").

**Viewer**
- Role‑filtered Trail Viewer generated: `trail_viewer_viewer.html`, `trail_viewer_analyst.html`, `trail_viewer_ops.html`.

**Policy automation**
- **Coverage tuner → policy** wiring picks `edges_per_node_max` and `total_edges_max` to hit target coverage at lowest cost. Selection saved to `applied_streaming_policy.json`. Top candidates table attached in session.

**Artifacts**
- Bundle: `sandbox:/mnt/data/SnapLat_P2c_bundle.zip` (CI projections, role viewers, applied policy).

> Ready to: (1) plug the applied policy into SnapOps at runtime, (2) export a golden‑replay test for projections that fails if hashes change for the same Trails, and (3) expose a small CLI to regenerate role viewers and policy selections on demand.



## U) P2d — Policy Applied, Golden Replay, and CLI (executed now)
**What I ran**
- **Applied tuner‑picked streaming policy** to a fresh E2E (v3) and logged Trails. Snapshot shows target coverage achieved with selected epn/total.
- **Golden replay for projections** on v1, v2, v3 Trails — recomputed twice and hashed. All three are **deterministic**.
- **Shipped a tiny CLI** (`snaplat_tools.py`) with two subcommands:
  - `project --trails <path> --outdir <dir>` → Graph/Catalog/Search JSON
  - `golden --trails <path>` → determinism + hashes

**Artifacts**
- **P2d runtime bundle:** `sandbox:/mnt/data/SnapLat_P2d_runtime.zip`
- **CLI:** `sandbox:/mnt/data/snaplat_tools/snaplat_tools.py`
- Workspace tables:
  - *E2E v3 — applied policy snapshot*
  - *Golden replay — projections*

> Ready to wire this CLI into CI and extend with `viewer --role …` or `select‑policy …` if you want a single tool entrypoint for day‑2 ops.



## V) P2e — CI Integration + CLI Extensions + Downstream Playbook (executed now)
**What’s added**
- **CLI extensions (`snaplat_tools.py`)**: `viewer` (role‑filtered HTML) and `select‑policy` (coverage tuner → policy JSON).
- **GitHub Actions CI** (`ci/snaplat_ci.yml`):
  - Locates a Trails file in the repo.
  - Runs **Golden Replay** (fails on non‑determinism).
  - Renders **Analyst Viewer** and uploads as artifact.
- **Downstream & Batch Playbook** — `docs/Downstream_and_Batch_Workflows.md` detailing step‑by‑step ops from intake → projections → policy → orchestration → gating → publishing → batch replays.

**Artifacts**
- **Bundle:** `sandbox:/mnt/data/SnapLat_CI_and_Workflows.zip`
- Files:
  - CLI: `sandbox:/mnt/data/snaplat_tools/snaplat_tools.py`
  - CI: `sandbox:/mnt/data/ci/snaplat_ci.yml`
  - Playbook: `sandbox:/mnt/data/docs/Downstream_and_Batch_Workflows.md`

> Ready to wire this into your repo as-is. If you want, I can also add a nightly batch workflow and a minimal HTML skin for the viewer.



## W) Master Roadmap & Plan (inclusive, system‑wide)

### North Star
Deterministic, explainable reasoning with bounded complexity (N‑levels, shells, 8‑way evaluation), full Trails lineage, and policy‑enforced execution. One append‑only source of truth (Trails/Snaps) → deterministic projections (Graph/Catalog/Search) → orchestrated action (ThinkTank → DTT → Assembly) under Safe‑Cube governance.

### Success Criteria
- **Determinism:** Golden replay stable for all projections and scoring; Trail replays produce identical outputs.
- **Explainability:** Every output links to Trails events + W5H alignment + Beacons; CiteGraph renders “why”.
- **Safety:** Deny‑by‑default across Safe‑Cube + ACL + Porter custody; auditable Policy Hash.
- **Performance:** Meets coverage watermark with quotas and backpressure; graceful degrade paths.
- **Data Ownership:** Proprietary © Nick Barker; default classification **CONFIDENTIAL‑PERSONAL**; export controls by role.

### Workstreams (WS)
- **WS‑A Foundations:** Contracts, Schema Registry, N‑level/Bridge FSM, Snap v1, E‑DBSU lifecycle.
- **WS‑B Graph Core & Scoring:** MDHG ops, promotion_breakdown (W5H/Beacons, hotness), quotas & streaming.
- **WS‑C Orchestration:** ThinkTank, DTT, Assembly, Scout, SnapOps policy gates (coverage, consolidation, advisory).
- **WS‑D Safety & Governance:** Safe‑Cube DSL, Policy Hash pinning, ACLs, Porter custody/DLQ, audit.
- **WS‑E Intake & RAG:** MORSR, Wave Pool, Idea Packaging, Code extract, Inclusiveness.
- **WS‑F Data Lifecycle:** Archivist expand/contract, Snap Ops Center, E‑DBSU runtime & rules.
- **WS‑G Observability:** Trails schema/validator, Viewer + role filters, query helpers, prints.
- **WS‑H Tooling & CI:** Golden replay, CLI, batch jobs, Makefile/actions, pre‑commit.
- **WS‑I Security & Compliance:** AuthZ roles, secrets, at‑rest/in‑flight encryption, multi‑tenant story.
- **WS‑J Productization:** Docs, playbooks, configs, packaging, release notes, DX polish.

### Phases & Milestones (acceptance tests abbreviated)
**M0 — Contracts Freeze (DONE partially)**
- .pyi surfaces (§11), Trail bindings, version markers.  ✅ Golden replays wired.
- **Accept:** Importable contracts; tests enforce required fields.

**M1 — Core Specs & Snap v1 (PARTIAL)**
- N_LEVEL_BRIDGE_FSM, E_DBSU_LIFECYCLE, SNAP_V1_SCHEMA, ARCHIVIST_API. 
- **Accept:** Round‑trip snap (idempotent), FSM unit tests; E‑DBSU NEW→TRIAGED→PROMOTED/DEMOTED→ARCHIVED.

**M2 — Deterministic Projections (DONE v1)**
- Projectors (Graph/Catalog/Search), Schema Registry, golden replay; Batch/CI stubs.
- **Accept:** Hash‑stable reruns; CI fails on drift.

**M3 — Orchestration Skeleton (DONE v1)**
- ThinkTank → DTT(dry) → Assembly; DLQ skip; Porter custody; role viewers.
- **Accept:** E2E transcript + WorkOrder; DLQ path proven; Trails coverage.

**M4 — Policy Gates & Coverage (DONE v2/v3)**
- Coverage watermark + approach‑phase consolidation; tuner→policy; applied in runtime.
- **Accept:** Gate triggers under watermark; consolidate_ok within margin.

**M5 — Scoring v2 (IN PROGRESS)**
- promotion_breakdown uses W5H/Beacons + **real hotness**; config‑driven weights; named beacons.
- **Accept:** Monotonicity properties; beacons loaded from registry; traces with component breakdown.

**M6 — Safety Hardening (IN PROGRESS)**
- Safe‑Cube rules expanded; ACL on ingest & projections; Porter DLQ + quarantine.
- **Accept:** Deny‑by‑default demonstrated; audit Trails for each decision.

**M7 — Intake & Wave Pool (QUEUED)**
- MORSR segmentation, Wave Pool ranking, SnapOps handoffs; inclusiveness sweep.
- **Accept:** Regions sum≈1; top‑k pooling; Trail joins from intake → orchestration.

**M8 — Advisory Heuristics (QUEUED)**
- RW/MCMC/SMC advisory behind policy toggles; diagnostics gates; optional adoption.
- **Accept:** Hints emitted & optionally adopted; Trails show adoption reasons.

**M9 — Security & Compliance (QUEUED)**
- Secrets, encryption, multi‑tenant; audit exports; data retention.
- **Accept:** Minimal role model validated; encrypted storage smoke.

**M10 — Alpha Cut (QUEUED)**
- End‑to‑end sandbox open; feature flags; SLOs & budgets pinned.
- **Accept:** Alpha SLOs met (determinism, explainability, safety, coverage).

### Dependencies (critical path)
- M1 (FSM, Snap) → M5 (Scoring) & M7 (Intake)
- M2 (Projections) → M4 (Coverage) → M10 (Alpha)
- M3 (Orchestration) → M4, M6 (Safety)

### Acceptance/DoD by subsystem
- **MDHG:** to_points/_n, to_graph (quota); promotion_breakdown with W5H/Beacons/hotness; property tests.
- **Trails:** schema validated; rendezvous & surgery events; replay harness.
- **Archivist:** Idempotent round‑trip; fingerprinting; Safe‑Cube deny hooks.
- **SnapOps:** Policy knobs; applied policy file; DLQ + coverage + consolidate_ok in Trails.
- **E‑DBSU:** Lifecycle transitions + property checks; evidence signals.
- **Safety:** Rules file tests; allow/deny with reason; custody & DLQ.
- **Beacons/W5H:** Registry load; cosine alignment; neg penalties; docs.
- **CI/CLI:** Golden replay; viewer & select‑policy; nightly batch job.

### Governance & Data Policy
- **License:** Proprietary — © Nick Barker. Inbound third‑party materials stored as **reference‑only** unless cleared.
- **Classification:** Default **CONFIDENTIAL‑PERSONAL**; roles: viewer/analyst/ops; deny‑by‑default.
- **Citations (“CiteGraph”):** URN scheme `snaplat://u/<universe>/l/<lattice>/n/<node>#snap:<id>`; edge weights carry force components (w5h, beacon, bridge, neg_penalty). Renders in Viewer with Trails back‑links.

### Risk Register → Mitigations
- **Spec drift:** lock via .pyi + CI golden replay → fail fast.
- **Coverage stalls:** tuner→policy + approach consolidation; E‑DBSU triage.
- **Non‑determinism:** CAS inputs; no implicit globals; hash seeds in tests.
- **Security gaps:** ACL everywhere; encrypted stores; audits; role‑filtered views.

### Backlog (Parked Questions → Experiments)
- 8‑way Weyl mapping proof across interior lattices.
- Bridge confidence update rule (Bayesian / hash strength).
- Glyph collisions: ledger + backpressure.
- N=4 saturation under observational incompleteness.

### Immediate Next Actions (this thread)
1) Finish Scoring v2 wiring into real wrapper; compute hotness from actual graphs; attach component deltas to Trails.
2) MORSR/Wave Pool integration path to SnapOps with Trails & tests.
3) Nightly CI workflow (batch projections + golden + policy select + viewer artifacts).
4) CiteGraph proto in Viewer: render URNs + component forces; tie to Beacons/W5H.

> This roadmap is live—each executed pass appends metrics and flips checklist items to DONE. I’ll keep updating it as we progress.



## X) P3 — Four Downstream Accelerators (executed now)
**1) MORSR + WavePool integrated into SnapOps**
- ThinkTank → DTT → **MORSR region map** → **WavePool ranking** → gate (`top≥0.34`).
- Trails: `morsr.regions`, `wavepool.rank`, `snapops.gate` (decision & top).

**2) Scoring v2 wrapper (Trails-attached)**
- Computes **hotness** (degree heat), **W5H/Beacons** alignment, **neg** penalties, **bridge**, and overall **score**.
- Emits `mdhg.promotion_breakdown.v2` with **component deltas** vs. baseline.

**3) Nightly CI workflow (scheduled)**
- `ci/snaplat_nightly.yml` runs on a cron & on demand: Golden replay, Analyst viewer, policy selection; uploads artifacts.
- Notes doc added.

**4) CiteGraph proto**
- URN scheme `snaplat://u/{universe}/l/{lattice}/n/{node}#snap:{id}` with a minimal HTML table showing forces per node.

**Artifacts**
- **Bundle:** `sandbox:/mnt/data/SnapLat_P3_build.zip`
- **CiteGraph HTML:** `sandbox:/mnt/data/p3/outputs/citegraph.html`
- **Nightly CI:** `sandbox:/mnt/data/p3/ci/snaplat_nightly.yml`
- **Trails:** `sandbox:/mnt/data/p3/outputs/trails_p3.jsonl`

**Snapshot (this run)**
- SnapOps+MORSR/WavePool: **OK** (gate passed).
- Scoring v2 components: w5h≈0.9810, hot≈0.7412, neg≈0.0613, bridge=0.5 ⇒ **score≈0.7702**.
- Deltas vs. baseline: Δscore≈+0.2202; Δhot≈+0.3412.

> Next candidates: (a) fold MORSR/WavePool outputs into SnapOps policy (auto‑branch to Assembly vs E‑DBSU), (b) enrich CiteGraph with edge‑level forces and Trails back‑links, (c) parameterize region gate with policy.



## Y) P3b — Policy Branching + Enriched CiteGraph (executed now)
**What changed**
- **SnapOps policy now auto‑branches** after MORSR/WavePool: if the top region score meets a threshold, we continue to **Assembly**; otherwise we route to **E‑DBSU** for triage. Threshold is configurable.
- **CiteGraph is richer:** it now shows **nodes and edges** with their force components (alignment, heat, negative penalties, bridge, total score) and includes a **Show sample Trails** button to preview related events from this run.

**Artifacts**
- Bundle: `sandbox:/mnt/data/SnapLat_P3b_branch_and_citegraph.zip`
- CiteGraph (enriched): `sandbox:/mnt/data/p3b/outputs/citegraph_enriched.html`
- Trails for this pass: `sandbox:/mnt/data/p3b/outputs/trails_p3b.jsonl`

**Run outcome**
- Branch taken: **assembly** (top region cleared the threshold).

**Next simple upgrades (if you want)**
- Make the threshold part of a policy file so you can tune it without code.
- Add per‑edge **back‑links** that jump directly to the exact Trail indices that created that edge.
- Show **why** a branch was taken (inline explanation based on the top region value vs. threshold).



## Z) P3c — Functional Hardening (policy file, strict gates, deterministic scoring, E‑DBSU) — executed
**Why this pass**
Prioritize function over UI: bake decisions into a policy file, make gating strict, prove determinism, and route failures cleanly to E‑DBSU.

**What’s in**
- **Policy file** `snapops_policy.json` (coverage watermark, approach margin, region threshold, score minimum, graph quotas, n_points). Loaded at runtime.
- **Strict gating logic** in SnapOps: Coverage→(consolidate margin)→Region & Score→Assembly or **E‑DBSU** (DLQ only on DTT fail). All decisions logged.
- **Deterministic scoring v2**: hotness from the actual graph, W5H/Beacons alignment, neg penalties, bridge. Golden check ensures identical components on re-run.
- **E‑DBSU triage**: deterministic promotion/demotion based on (score, mdhg region), archived with Trails.

**Artifacts**
- Bundle: `sandbox:/mnt/data/SnapLat_P3c_functional_hardening.zip`
- Trails: `sandbox:/mnt/data/p3c/outputs/trails_p3c.jsonl`

**Sanity scenarios**
- **A (default policy)** → branch: *assembly* if thresholds met; score & coverage logged.
- **B (strict policy)** → branch: *edbsu* when `score_min` is raised above observed score.
- **Determinism**: scoring golden check **PASS** (no randomness in components).

**Next functional steps**
1) Externalize **weights** (w_w5h, w_hot, w_neg, w_bridge) to policy; log breakdown and rationale.
2) Promote beacons to a repo file; add a **hash pin** of the registry in Trails.
3) Add **FSM unit tests** for the gating sequence (states: PLAN→DTT→GRAPH→COVERAGE→SCORE→REGION→{ASSEMBLY|E‑DBSU}).
4) Enforce policy in CI: fail if policy changes cause golden scoring to drift or gates flip unexpectedly on known fixtures.



## AA) P3d — Function‑First Hardening (executed now)
**What’s locked**
- **Weights in policy**: scoring weights (w_w5h, w_hot, w_neg, w_bridge) live in `snapops_policy.json`; loaded at runtime.
- **Beacons hash‑pinned**: registry is file‑backed and hashed (sha256) per run; Trails log `beacons.hash` with path+hash.
- **FSM test for gates**: validates the ordered sequence PLAN→DTT→GRAPH→COVERAGE→SCORE→REGION→{ASSEMBLY|E‑DBSU}; fails on any off‑path event.
- **Deterministic scoring**: re‑runs yield identical components (hot/w5h/neg/bridge/score) under policy and beacon set.
- **Functional CI**: workflow checks for Trails and runs the FSM order check; will fail a PR on violation.

**Artifacts**
- Bundle: `sandbox:/mnt/data/SnapLat_P3d_function_first_bundle.zip`
- Policy: `sandbox:/mnt/data/p3d/policy/snapops_policy.json`
- Verify script: `sandbox:/mnt/data/p3d/tools/verify_functional.py`
- CI workflow: `sandbox:/mnt/data/p3d/ci/snaplat_functional_ci.yml`
- Trails: `sandbox:/mnt/data/p3d/outputs/trails_p3d.jsonl`

**Status (this run)**
- Weights in policy ✅  | Beacons hash‑pinned ✅  | FSM test ✅  | Scoring deterministic ✅  | CI emitted ✅

**Next function‑first steps**
1) Add a **policy drift** check (hash the policy and fail CI if it changes without a version bump).
2) Emit **component deltas** vs. a pinned baseline in Trails; fail CI when the sign flips unexpectedly.
3) Promote the **beacons registry** into repo (with review) and forbid local overrides in CI.
4) Add a **fixture Trails** under `tests/fixtures/` so FSM + scoring checks run on stable data without needing a fresh E2E.



## AB) P3e — Policy/Beacons Guards, Component Baseline, Fixture Trails, CI (executed now)
**Guards added (function-first)**
- **Policy drift guard**: hashes the full policy; CI fails if it changes without a **version bump**. Pin file: `policy_hash.json`.
- **Beacons registry guard**: loads the beacons at the enforced path and checks against a pinned **sha256**; CI fails on mismatch or path change.
- **Component baseline guard**: recomputes scoring components on a pinned **fixture graph** and requires an exact match to the baseline JSON.
- **FSM order guard (fixture Trails)**: verifies the exact event order on a stable Trails file.

**Fixtures & pins**
- Policy (versioned): `p3e/policy/snapops_policy.json` (+ `policy_hash.json`).
- Beacons: `p3e/policy/beacons.json` (+ `beacons_hash.json`).
- Component baseline: `p3e/outputs/components_baseline.json`.
- Fixture Trails: `p3e/tests/fixtures/trails_fixture.jsonl`.

**CI workflow**
- `p3e/ci/snaplat_guards_ci.yml` runs all four guards on PR/push.

**Bundle**
- `sandbox:/mnt/data/SnapLat_P3e_guards_and_fixtures.zip`

> These checks make changes explicit and safe: if someone edits policy, beacons, scoring, or event order, CI will flag it unless the version and baselines are updated deliberately.

