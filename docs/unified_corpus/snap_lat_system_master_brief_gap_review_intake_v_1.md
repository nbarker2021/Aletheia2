# SnapLat — System Master Brief & Gap Review (Intake v1)

> Scope: A single, coherent description of **all systems and concepts** discovered so far from documents and code, with cross-links, operating model, and a critical gap analysis. This is *pre‑refactor*, *pre‑wrapper*—it is the knowledge pack you’d hand to a senior engineer before any changes.

---

## 1) Intent & First Principles
- **Goal:** Deterministic reasoning with controlled complexity growth via explicit context bounding, lattice/shell expansion, and traceable step chains. Output must be explainable and reproducible (Trails), and composable across domains via bridges.
- **Method:** Iterative passes that first **build perfect space** (N-level saturation, shells, universes, glyph compression) before performing the user task.
- **Safety & Governance:** All operations pass through **Safe Cube** checks, **Policy Hash** pinning, and **Porter/Mannequin** isolation where side effects or untrusted inputs exist.

**Design maxims**
- *Everything is a lattice.* Each datum can be expanded to a full lattice; lattices nest; bridges connect lattices at N≥5.
- *Trace then compress.* Work expands ephemeral structures to N=1..4, then compresses to glyphs/Snaps, preserving context.
- *Eight-way truth.* For any idea, evaluate all **8 Weyl points** (positive/inverse variants across interior lattices).

---

## 2) Core Formalism (N‑levels, Shells, E8, Bridges, Snaps)
- **N=0** — *Unchangeables*: natural laws, environmental constraints, fixed drivers; also contains bridge rules. Baseline stance/policy applied, but no data touched.
- **Bridges** — carry a base weight of **0.25** (N=.25) by convention. Two kinds:
  - **Double‑known bridges** → normal **N=0.5** nodes.
  - **Double‑unknown bridges** → routed to **E‑DBSU** (Emergent DB for Special Unknowns) for analysis; candidates for emergent meaning.
- **N=1..4** — increasing saturation of definition:
  - **N=1**: unambiguous per‑context meaning.
  - **N=2**: greedy safe zone (binary-determined info OK).
  - **N=3**: triggers **MDHG pre‑inform/JIT prep** (most N=3 lie on spectra).
  - **N=4**: exactly one perfect outcome.
- **N=5** — ambiguity emerges; exactly **8** outcomes exist (aligned with **E8** Weyl/Lie properties). Bridges form naturally at N=5.
- **E8 lattice / Weyl points** — The **safe cube** is the center; 8 Weyl points are the binary outcomes of the two true/false sets across all relevant interior lattices. Treat these 8 as the **starter safe cube** for cross-domain evaluation until full space is defined.
- **Shelling** — every datum has shells; expansion builds shells outward; compression collapses to glyphs; both kept as Snap lineage.
- **Snaps** — compressed records of reasoning/action. All interactions logged as Snap data; only active working sets are expanded via **Archivist + Snap Ops Center**; finished steps reconsolidate.

---

## 3) Orchestration Model (AGRM, Arms, Rendezvous, Ticks)
- **AGRM (Attention‑Guided Region Mapper)** defines *where* and *how* to work:
  - **Main arms**: primary workers following the planned path.
  - **Scout arms**: fast counter‑directional observers; spawn when:
    - task is N≥4 hard → greedy triggers **prep MDHG** and scouts preseed shells.
    - resources <50% used → opportunistic expansion.
  - **Partitions**: semantic clusters by default; geometric (hemispheres/quadrants) allowed; rendezvous points evolve with unlocks.
  - **Ticks**: execution proceeds in ticks; at **pause zones** main arm publishes findings to **ThinkTank**; ThinkTank returns guidance for the next tick.

**Greedy vs JIT**
- Greedy operates confidently at N=2; at N≥3 it **pre‑informs** MDHG, prebuilding structures for later iterations.

---

## 4) Reasoning & Always‑On Services
- **ThinkTank** — expert panel synthesis + critique; consumes tool outputs and proposes next steps.
- **DTT (Double‑Track Tester)** — sandbox test runner; validates alternative plans.
- **Assembly Line** — builds base modules and skeletons in parallel during early iterations.
- **Mannequin & Porter** — isolation & mediation for untrusted effects; **Safe‑Cube Sentinel** enforces guardrails.
- **Policy Hash / Run Manifest** — governance and config pinning for replay.

All three (ThinkTank, DTT, Assembly Line) are **always on**, **resource‑allocated** independently, and report to Trails.

---

## 5) Data Intake & Knowledge Growth (MORSR + Wave Pool RAG)
- **MORSR** + **Wave Pool** act as a RAG‑style parser tuned to **text bodies** with lattice/shell awareness:
  - Segments logs into **Idea Packages** (Human directive + AI work) using pivot detection (new artifacts, scope handoffs, approvals resets).
  - Expands lexicon continuously (keyword families, multi‑word anchors) to avoid search‑space narrowing.
  - Produces **probability maps** for where deep review should focus next; feeds MDHG tables.

---

## 6) Graph & Hash Core (MDHG) + Orchestrator (AGRM/TPG)
- **MDHG** (Multi‑Dimensional Hash Graph)
  - **Tables** as buildings/floors/rooms/elevators; **hot/edge maps** capture conflict and adjacency.
  - **Materializers**: `to_points(universe) → list`, `to_graph(points, quotas=None) → (nodes, edges)`.
  - **Promotion/Scoring**: `promotion_breakdown(record) → dict` with W5H alignment when present.
  - **Caches/Uniqueness**: step cache, neg/pos beacon cache; boundary checks.
- **AGRM Orchestrator + TPG**
  - **TPGConfig** (enabled, max_nodes, beam, imperf_threshold, two_opt).
  - **Surgery / 2‑opt** for path refinement.
  - **Negative beacons** penalize undesired regions.

---

## 7) Trails / W5H / Beacons (Observability)
- **Trails** = begin/append/finalize chain with event taxonomy and lineage pointers; Trail Viewer hooks into reports.
- **W5H** vectors represent question framing and alignment; **BeaconsRegistry** holds domain anchors.
- **Neg‑beacons**: counter‑examples or forbidden zones (penalty/filters).
- Logging schema includes: policy hash, safe‑cube decision, porter custody, lattice shell/level, Weyl point index, MDHG location (building/room/elevator), W5H payload.

---

## 8) FS2 / Superperm Plugin (Uniqueness & Canonicalization)
- **Purpose**: enforce uniqueness constraints across combinatorial expansions (e.g., superpermutation search), reconcile duplicates via canonicalization.
- **Surfaces**: `UniquenessRegistry`, `StepCache`, `Envelope.allows(step)`; canonical relabel/rotation/reversal; integration with MDHG/TPG.
- **Known pre‑alpha flags from code**: missing import `hashlib`; reference to undefined `AntiLaminate`.

---

## 9) Compression & Archivist / Snap Ops Center
- **Glyph Compression**: target **3‑word glyph** for N=5; ASCII first, broaden as glyph space matures; if 3 words fails, treat as **insufficient N‑saturation**, not a rule exception.
- **Archivist** moves active working sets from compressed Snaps to expanded form and back; **Porter** carries payloads between zones; **Snap Ops Center** orchestrates expand/contract.

---

## 10) Execution Lifecycle
1. **Initialize** N=0 constraints, policy hash, safe‑cube guard.
2. **Scout Bake‑in**: spawn scouts; preseed MDHG when any path hits N≥3.
3. **Iterate** in ticks: expand shells, evaluate all 8 Weyl points per interior lattice; log Trails.
4. **Consolidate**: compress to glyphs; store Snaps; promote/demote bridges (E‑DBSU for double‑unknowns).
5. **Task Execution**: only after full N=4 saturation of the relevant space.

---

## 11) Interfaces (Desired Minimum)
- **MDHG**: `to_points`, `to_graph`, `promotion_breakdown`, caches.
- **AGRM/TPG**: `TPGConfig`, surgery 2‑opt flags; neg‑beacon access.
- **Trails**: `begin_trail`, `append_event`, `finalize`.
- **W5H/Beacons**: `compute_w5h_alignment`, `BeaconsRegistry.get(name)`.
- **FS2**: `UniquenessRegistry.check(seq)`, `StepCache.get/set`, `Envelope.allows(step)`.
- **Governance**: `POLICY_HASH`, `SafeCubeSentinel.allow(op, context)`, `Porter.deliver(payload, to=...)`.

---

## 12) Critical Review — Hindrances & Design Choices
**Where choices are needed or risky:**
- **Partitions**: semantic vs geometric; we adopt **semantic‑first**, with geometric overlays when needed.
- **Greedy vs JIT**: Greedy kept to N≤2; at N≥3 we *prepare* MDHG and keep scouts active.
- **Glyph rule**: hard **3‑word** target for N=5; failure indicates insufficient N, not larger glyphs.
- **Character set**: ASCII first; expand as the glyph library grows.
- **Bridge semantics**: honor **.25 weight**; strict routing of double‑unknown to **E‑DBSU**.
- **Full eight‑way evaluation**: mandatory per idea and per interior lattice.

**Observed hindrances (from code & intake):**
- Mixed concerns in single modules; missing imports (`hashlib`), undefined symbols (`AntiLaminate`).
- Hardcoded thresholds; lack of config injection.
- Sparse/absent telemetry hooks; no standardized Trail events.
- Lack of deterministic seams and tests; some implicit globals likely.

---

## 13) Gap Review (from Interface Fit + Code Intake)
**Interfaces missing or partial**
- `policy_hash`, `trails_hooks` (begin/append/finalize), `to_points`, `to_graph`, `promotion_breakdown`.
- `TPGConfig`, `neg_beacon_support`.
- `w5h_alignment`, `BeaconsRegistry`.
- FS2 envelope: `UniquenessRegistry`, `StepCache`, `Envelope.allows`.
- Governance: `SafeCubeSentinel`, `Porter` custody.

**Observability gaps**
- Trail schema not wired; no taxonomy-enforced logging.

**Determinism & Config gaps**
- Hardcoded values; no `run_manifest` pattern visible.

**Testing gaps**
- No harnesses; import‑time side‑effects likely.

**Performance risks**
- Potential O(N^2) hotspots in naive graph/uniqueness checks; need caches/quotas.

**Data lifecycle gaps**
- Snap/Archivist interfaces unspecified; E‑DBSU lifecycle not codified.

---

## 14) Lexicon (Working Glossary)
- **AGRM** — orchestrator of work regions.
- **MDHG** — multi‑dimensional hash/graph core.
- **TPG** — path generator + 2‑opt surgery.
- **Trails** — immutable event lineage.
- **W5H** — who/what/when/where/why/how vectors.
- **Beacons / Neg‑beacons** — positive/negative anchors.
- **Safe Cube** — central validation zone.
- **Porter / Mannequin** — custody/isolation.
- **E‑DBSU** — emergent database for double‑unknown bridges.
- **MORSR / Wave Pool** — RAG‑style intake tuned to text with lattice semantics.

---

## 15) Synthesis — What “Mastery” Looks Like Here
A senior engineer must be fluent in:
- Running the **8‑way evaluation** loop for any idea across interior lattices.
- Deciding **when to expand** shells/universes and when to compress (Snaps).
- Operating **scout/main arms**; reading ThinkTank/DTT outputs and gating ticks.
- Wiring Trails (begin→append→finalize) and **pinning policy hash**.
- Converting greedy progress at N=2 into JIT prep for N≥3 (MDHG pre‑inform, MORSR focus).
- Enforcing uniqueness with **FS2** while preserving bridges at N=5.

---

## 16) Immediate Recommendations (Pre‑Refactor)
1) Lock the **interface surfaces** in §11 as contracts (names/signatures).
2) Adopt the **Trail schema** & templates globally.
3) Stand up **run_manifest** and configuration injection; remove hardcoded thresholds.
4) Decompose mixed‑concern modules along the family plans; add docstrings/types.
5) Formalize **Archivist ↔ Snap Ops** API; define E‑DBSU lifecycle.

*End of Intake v1 — ready for milestone planning & refactor execution once approved.*


---

# 17) Holistic “Incomplete in Form or Function” Review
> A cross‑system critique of everything gathered (docs, code, canvases). The aim is to surface where the idea is **underspecified, unsafe, unverifiable, or not yet executable**—and to define **proof obligations** to convert concept → working system.

## 17.1 Theory & Formalism (E8 / N‑levels / Bridges / Glyphs)
**What’s incomplete**
- **N‑level calculus**: We reference N=0/.25/.5/1..5 semantics, but lack a **formal state machine** for legal transitions, merge rules, and downgrade rules (e.g., when new evidence demotes an N=1 to N=.5).  
- **Bridge taxonomy & weights**: `.25` baseline is a convention; we lack a **calibration scheme** and **confidence update rule** (Bayesian/likelihood or hashing strength) for turning bridge evidence into N=.5, N=1.  
- **E‑DBSU lifecycle** (double‑unknowns): missing intake schema, triage policy, promotion/demotion rules, and retention/archival policy.  
- **E8/Weyl mapping**: Claim: at N=5 there are exactly 8 outcomes; we treat Weyl points as emergent bridges. We need: (1) a **constructive mapping** from idea→interior lattices→8 points; (2) **invariants** that hold across domains; (3) **counterexample harness** to detect when 8‑way assumption fails.  
- **Glyph compression (3‑word rule)**: No **collision policy**, **namespace/versioning**, or **ambiguity backpressure** (what happens when 3 words are insufficient but N isn’t saturated).  
- **Snap algebra**: No formal **compose/decompose** laws for Snaps (associativity/commutativity constraints, Snap‑of‑Snap recursion, idempotent replays).

**Proof obligations**
- A **typed spec** for N‑transitions + bridge updates.  
- A **Weyl test battery**: given any idea, produce 8 evaluants + metamorphic relations; fail closed if fewer/greater are observed.  
- A **Glyph ledger**: collision tests, reversible mapping (glyph ⇄ meaning), and backward‑compatible upgrades.

---

## 17.2 Orchestration (AGRM, Arms, Rendezvous, Ticks)
**What’s incomplete**
- **Partitioning policy**: “semantic first, geometric optional” is agreed, but no **selection heuristic** (e.g., entropy/novelty/edge density) nor **repartitioning trigger** (when/why to re‑tile).  
- **Scout arms**: Spawn/TTL/backoff rules absent. Risk of **deadlock** at pause zones if main arm blocks awaiting scouts that are starved of compute. No **conflict aggregation** policy when scouts contradict main arm (ThinkTank arbitration protocol missing).  
- **Tick scheduler**: No **fairness** or **priority** model; no **pause handshake** or **checkpoint format** (partial work rehydration).  
- **Resource governor**: “<50% free → add scouts” is crude; needs **budgeting** (CPU/mem/time) and **preemption** rules.

**Proof obligations**
- A **tick contract** (state diagram + checkpoint schema).  
- A **scout SLA**: spawn limits, rendezvous cadence, conflict resolution via ThinkTank quorum + Trails evidence.  
- **Partition heuristic** with measurable objectives (coverage gain vs. cost).

---

## 17.3 Graph Core (MDHG) & Orchestrator (AGRM/TPG)
**What’s incomplete**
- **Data structure choices**: Buildings/floors/rooms are metaphors; we need concrete **types** (hash maps, adjacency lists, sparse matrices) + complexity budgets.  
- **Materializers** (`to_points`, `to_graph`): signatures are named, but **quotas**, **edge filtering**, **uniqueness policy**, and **points provenance** are unspecified.  
- **Promotion/scoring**: `promotion_breakdown` exists as a name only; no **formula** integrating W5H alignment, neg‑beacons, and MDHG hot‑edge maps.  
- **Adaptation/conflicts**: no algorithm for **elevator choice**, **conflict backoff**, or **threshold removal** (config injection required).  
- **TPG**: `TPGConfig` not defined; **2‑opt surgery** entry/exit conditions undefined; **negative beacon** penalties missing.

**Proof obligations**
- A **MDHG API spec** with types + complexity 
- A **scoring panel** (formula + units + calibration steps).  
- **TPGConfig** dataclass + surgery protocol.

---

## 17.4 Observability (Trails / W5H / Beacons)
**What’s incomplete**
- **Event taxonomy**: begin/append/finalize named, but no **required fields** per event, **PII policy**, or **retention**.  
- **Trail Viewer**: no interface or anchors into reports.  
- **W5H vector** generation is unspecified (source, normalization, weighting).  
- **BeaconsRegistry** lacks persistence model, versioning, and conflict handling.

**Proof obligations**
- **Trail schema** contract (already drafted) must be bound per event type.  
- **W5H construction** doc + tests (consistency under paraphrase, noise).  
- **BeaconsRegistry** CRUD with provenance.

---

## 17.5 Governance & Safety (Policy / Mannequin / Porter)
**What’s incomplete**
- **Safe‑Cube Sentinel** lacks a policy language and denial audit trail.  
- **Porter** has no **custody log**, **retry/poison queue**, or **backpressure** plan.  
- **Policy Hash** pinning not connected to runtime decisions (hash present, but not enforced).

**Proof obligations**
- **Safe‑Cube DSL** (allow/deny with reasons) + Trails hooks.  
- **Porter contract** (at‑least‑once semantics + dead‑letter handling).  
- **Policy enforcement points** mapped across modules.

---

## 17.6 FS2 / Superperm Plugin
**What’s incomplete**
- **Canonicalization** (relabel/rotation/reversal) is referenced, not implemented; **cache invalidation** rules absent.  
- **Envelope/Uniqueness** not wired to MDHG expansion gates.  
- Code issues: missing `hashlib`, undefined `AntiLaminate`, mixed concerns.

**Proof obligations**
- **Invariant tests** for canonicalization; **Envelope** truth table; **UniquenessRegistry** contract.

---

## 17.7 Data Lifecycle (Snaps / Archivist / E‑DBSU)
**What’s incomplete**
- **Snap schema** (IDs, parentage, context windows, glyph set, compression codec) not pinned.  
- **Archivist** lacks **rehydration rules**, **merge semantics**, and **conflict resolution**.  
- **E‑DBSU** lacks lifecycle states (new → triaged → promoted/demoted → archived).

**Proof obligations**
- **Snap v1 schema** + idempotent replay demo.  
- **Archivist spec** (expand/contract API) with determinism tests.  
- **E‑DBSU state machine** and promotion criteria.

---

## 17.8 Testing & Verification
**What’s incomplete**
- No **metamorphic tests** across the 8 Weyl points; no **property‑based** tests for lattice operations; no **golden trail replays**.  
- Test data strategy is undefined (fixtures, anonymization, coverage targets).

**Proof obligations**
- **Metamorphic suite** for 8‑way consistency; **property tests** for shells/bridges; **Trail replay** harness with hash pinning.

---

## 17.9 Performance & Resource Model
**What’s incomplete**
- No **cost model** for MDHG building or scout proliferation; missing **quotas** for points/edges; no **streaming** mode.  
- No **graceful degradation** path when budget is exceeded.

**Proof obligations**
- **Complexity budgets** (O‑notation + practical limits); **quota policy**; **degradation playbook**.

---

## 17.10 Security & Compliance
**What’s incomplete**
- **AuthZ** for operations; **secrets** handling; **encryption** at rest/in flight; **audit** trails alignment with Trails.  
- No **multi‑tenant** story.

**Proof obligations**
- **Security profile** + audit checklist; minimal **role model** (ops vs. analyst vs. system).

---

## 17.11 Repo/Build/Tooling
**What’s incomplete**
- **Type checking** (mypy/pyright) and **linting** policies not set.  
- No **CI harness** yet; tests are placeholders.

**Proof obligations**
- Add **pre‑commit** hooks; enable **CI** with importability + stub tests.

---

## 17.12 Top Critical Gaps (Actionable)
| Pri | Family | Gap | Proof Obligation | First Experiment |
|---|---|---|---|---|
| P0 | MDHG | `to_points/to_graph` undefined | Type‑precise API + quota/uniqueness policy | Build a toy universe → points/graph; measure quotas |
| P0 | Trails | No enforced schema | Bind schema to events in harness | Emit begin/append/finalize for a dry‑run |
| P0 | Policy | Hash not enforced | Map enforcement points & fail closed | Inject hash into a failing op |
| P0 | AGRM/TPG | `TPGConfig` undefined | Dataclass + surgery protocol | Run 2‑opt on synthetic path |
| P1 | Scoring | `promotion_breakdown` empty | Calibrated formula w/ W5H + neg‑beacons | Grid‑search weights on fixtures |
| P1 | Orchestration | Scout SLA missing | Spawn/TTL/backoff & ThinkTank quorum | Simulate contradictory scouts |
| P1 | FS2 | Canonicalization & envelope | Invariant suite + envelope truth table | Randomized canon. tests |
| P1 | Data | Snap schema missing | v1 schema + replay idempotence | Round‑trip a toy Snap |
| P2 | Safety | Safe‑Cube DSL absent | Deny/allow grammar + audit | Force a denial path |
| P2 | Porter | Custody semantics | At‑least‑once + dead‑letter | Inject a poison payload |

---

## 17.13 Build‑Direction Gating (Do Not Build Until)
- **Contracts frozen**: Interfaces in §11 emitted as `.pyi` stubs; Trail schema bound.  
- **Spec complete**: N‑transition FSM; E‑DBSU lifecycle; Snap v1 schema.  
- **Harness ready**: Weyl metamorphic tests; Trail replay harness; toy universes/fixtures.  
- **Budgets pinned**: MDHG/TPG complexity & quotas; scout governor.  
- **Security minimum**: policy enforcement points wired; deny‑by‑default works.

---

## 17.14 Parked Questions (for Final Review Only — no answers now)
- Under what conditions can the 8‑way Weyl mapping be **provably** extended across multiple interior lattices without contradiction?  
- What’s the **confidence update rule** for bridges that cross independent lattices?  
- What constitutes **N=4 saturation** in probabilistic domains (observational incompleteness)?  
- What is the **collision response** when two distinct meanings compress to the same 3‑word glyph in different universes?  
- How do **neg‑beacons** interact with Safe‑Cube denials (filter vs. hard fail)?

---

## 17.15 Immediate Next Actions (No Code Changes)
1) Emit `.pyi` contracts for the interfaces in §11 and pin Trail schema → **Contract Freeze**.  
2) Author **N‑level + bridge FSM** + E‑DBSU lifecycle → spec review.  
3) Define **Snap v1** schema + Archivist expand/contract API.  
4) Stand up **Weyl metamorphic** test harness + toy universes.  
5) Draft **MDHG scoring panel** (with W5H & neg‑beacons) and run fixture calibrations.


---

## Pass 9 — Contracts & Core Specs (frozen, no logic changes)
**Interface contracts emitted as `.pyi` (under `src/`)**
- `agrm/mdhg/api.pyi` → `to_points`, `to_graph`, `promotion_breakdown`  
- `tpg/config.pyi` → `TPGConfig` dataclass  
- `trails/api.pyi` → `begin_trail`, `append_event`, `finalize`  
- `agrm/snap/w5h.pyi` → `compute_w5h_alignment`  
- `agrm/snap/beacons.pyi` → `BeaconsRegistry`  
- `plugins/edges/superperm/fs2.pyi` → `UniquenessRegistry`, `StepCache`, `Envelope`  
- `ops/policy/api.pyi` → `POLICY_HASH: str`  
- `mannequin/api.pyi` → `SafeCubeSentinel.allow`, `Porter.deliver`

**Core specs & schemas (under `PLANS/`)**
- `TRAIL_EVENT_BINDINGS.md` — required fields per Trail event type.  
- `N_LEVEL_BRIDGE_FSM.md` — states/transitions/merge & demotion rules.  
- `E_DBSU_LIFECYCLE.md` — double‑unknown bridge lifecycle.  
- `SNAP_V1_SCHEMA.json` — Snap schema (IDs, glyph, context, n-level, shell, Weyl index).  
- `ARCHIVIST_API.md` — expand/contract API & determinism rules.  
- `MDHG_SCORING.md` + `mdhg_scoring.yaml` — scoring panel scaffold & default weights.

**Test scaffolding**
- `tests/metamorphic/test_weyl_metamorphic.py` — 8‑way outcome & metamorphic checks (placeholder until implemented).

**Downloads**
- Contracts & specs bundle: `/mnt/data/snaplat_contracts_specs_bundle.zip`

**Build Gate Status:** Contracts & core specs are **frozen** for review. Next, we can enable the M1 sandbox with wrappers that call these surfaces while preserving legacy logic.


---

## Contract Evolution & Seed Wiring
**Policy added:** `PLANS/CONTRACT_EVOLUTION.md` — contracts are **doc‑ & test‑driven**, mutable until final sign‑off; versioning and change process defined.

**Seed wiring (single next‑step action):** runtime **Trails** API implemented minimally at `src/trails/api.py` with an in‑memory sink to satisfy `trails/api.pyi`. A smoke test `tests/test_trails_bindings_smoke.py` asserts required fields/order (`begin → append → finalize`).

**Contracts version marker:** `src/contracts/version.json` (`0.1.0-experimental`, mutable=true).

> Rationale: This provides one carefully chosen “first wire” that unblocks future wrappers, while keeping the rest of the system unchanged. Contracts remain explicitly provisional and will evolve with further doc/test findings.


---

## Pass 10 — M1 Sandbox Seed (keeps legacy code untouched)
**What I added**
- `src/agrm/mdhg/wrapper.py` — adapter exposing `to_points`, `to_graph`, `promotion_breakdown`. Emits Trails. Uses a toy deterministic fallback; can optionally delegate to a legacy winner (no auto‑import to avoid side effects).
- `src/ops/policy/api.py` — minimal policy hash seed (`SNAPLAT_POLICY_HASH` env with `dev` default).
- `tests/test_mdhg_wrapper_smoke.py` — smoke: points→graph→score path works; Trails calls are made.

**Why this matches your rule**
- It’s a **single step of wiring** for the next action (M1 wrappers) while contracts remain mutable. No edits to legacy winners.

**Next candidates**
- Fold the wrapper through TPG config once we finalize `TPGConfig` shape in tests.  
- Add a tiny `promotion_breakdown` calibration fixture to start scoring panel tuning.


---

## Pass 11 — Trail Schema Enforcement (tests-only)
**Added**
- `src/trails/validate.py` — tiny validator (`validate_event`) enforcing required fields per event type (kept in sync with `TRAIL_EVENT_BINDINGS.md`).
- `tests/test_trails_schema_enforcement.py` — runs MDHG wrapper flow and validates *every* emitted event.
- `PLANS/TRAIL_SCHEMA_ENFORCEMENT.md` — policy note: enforcement in tests keeps runtime loose while catching drift early.

**Why this first**
- Trails are a P0 surface. Enforcing schema now prevents silent drift as we wire more adapters and keeps observability stable.


---

## Pass 12 — TPGConfig Locked (tests + minimal runtime)
**What’s added**
- `src/tpg/config.py` — `TPGConfig` dataclass (enabled, max_nodes, beam, imperf_threshold, use_two_opt) + sandbox no‑op `two_opt(...)` and `apply_surgery(...)` to pin orchestration seams.
- `tests/test_tpg_config_contract.py` — asserts field presence, defaults, and no‑op surgery call path.
- `PLANS/TPG_CONFIG_NOTES.md` — brief notes on locked fields and future wiring under feature flags.

**Why next**
- AGRM/TPG orchestration is a P0 touchpoint; locking `TPGConfig` gives us stable seams to integrate without touching legacy code.

**Next suggestion**
- Add a **tiny scoring fixture** to begin calibrating `promotion_breakdown` (M1): we can generate a few toy records and assert the component breakdown totals and monotonicity under weight sweeps (still no changes to legacy modules).


---

## Pass 13 — M1 Scoring Fixtures & Property Tests
**Fixtures**
- `tests/fixtures/mdhg_scoring_fixtures.json` — curated toy records spanning `w5h`, `hot`, `neg_penalty`, `bridge_conf` axes.

**Tests (property-based)**
- `tests/test_mdhg_scoring_fixture.py` — asserts component presence, [0,1] bounds, and monotonicity (↑`w5h`/`bridge_conf`/`hot` ⇒ ↑score; ↑`neg_penalty` ⇒ ↓score).

**Review artifact**
- Grid sweep logged at `PLANS/sweeps/m1_scoring_sweep.csv` and previewed in-session for inspection.

**Why properties vs. hard numbers**
- We will iterate weights (`PLANS/mdhg_scoring.yaml`) as understanding matures. Property tests keep the suite stable while allowing numeric calibration to change.

**Next (suggested)**
- Wire W5H alignment stub (`agrm/snap/w5h.py`) to feed the scoring panel; add a tiny BeaconsRegistry stub to start real alignment tests.


---

## Pass 14 — W5H & Beacons wired to Scoring (sandbox)
**Runtime stubs**
- `agrm/snap/w5h.py` — `compute_w5h_alignment(beacon_vec, room_vec)` via cosine similarity → [0,1].
- `agrm/snap/beacons.py` — minimal `BeaconsRegistry` (in‑memory).

**MDHG wrapper update**
- `promotion_breakdown` now **derives `w5h`** from vectors when present and **derives `neg_penalty`** from `neg_beacon_vec` vs `room_vec` (capped at 0.2). Numeric fields still honored if vectors are absent.

**Tests**
- `tests/test_w5h_beacons_integration.py` — (1) alignment ↑ ⇒ score ↑, (2) neg‑beacon alignment ⇒ score ↓ relative to neutral.

**Notes**
- `PLANS/W5H_BEACONS_NOTES.md` — documents the sandbox semantics; coefficients will later move to config.

**Next (M1.2 suggestion)**
- Make scoring **config‑driven** by reading `PLANS/mdhg_scoring.yaml` in the wrapper (under a feature flag) and extend the property tests to assert monotonic response to config changes.


---

## Pass 15 — Config‑Driven Scoring & Named Beacons
**What changed**
- **Config‑driven weights** in `promotion_breakdown` (flagged): when `SNAPLAT_USE_SCORING_CONFIG=1`, the wrapper reads weights from `PLANS/mdhg_scoring.yaml` (or `SNAPLAT_SCORING_CONFIG` path) via a minimal YAML‑like parser. Fallback remains the prior toy weights.
- **Named beacons** support: `beacon_name` and `neg_beacon_name` are resolved via a default in‑memory `BeaconsRegistry`; vectors remain accepted directly.

**Code touched**
- `agrm/snap/beacons.py` — added default registry + `set_registry_data()`/`get_registry()` helpers.
- `agrm/mdhg/wrapper.py` — added config loader; extended scoring to use named beacons; switched to weighted sum with clamping.

**Tests**
- `tests/test_scoring_config_and_named_beacons.py` — verifies (1) changing `w_w5h` in config increases score monotonically and (2) named beacons behave equivalently to explicit vectors; neg‑beacon reduces score vs. neutral.

**Demo artifact**
- “Config‑driven Scoring — Demo” table compares scores under low vs. high `w_w5h` configs.

**Next (suggested)**
- **M1.3 — Scout Rendezvous Event Schema (tests‑only)**: add a `rendezvous` Trail event with required fields (`arm`, `checkpoint`, `quorum`, `payload`) to lock ThinkTank/DTT handoff shape; update validator and bindings doc.  
- **M1.4 — Named Beacons registry load**: allow optional load from `PLANS/beacons.json` and add a test that swaps registries on the fly.


---

## Pass 16 — Scout Rendezvous Event (tests‑only) + Prep Next (Beacons load)
**Trail schema extended**
- `TRAIL_EVENT_BINDINGS.md` now includes **`rendezvous`** with required fields: `trail_id, ts, event, module, arm, checkpoint, quorum, payload`.
- `src/trails/validate.py` updated to enforce `rendezvous` schema.

**Runtime seed**
- `src/trails/api.py` gained `rendezvous(trail_id, *, arm, checkpoint, quorum, payload)` which appends a `rendezvous` event to the in‑memory sink (used in tests/harnesses).

**Tests**
- `tests/test_trail_rendezvous_schema.py` emits a rendezvous event and validates it with the schema validator.

**Prep for next step (Beacons Registry Loading)**
- `PLANS/BEACONS_REGISTRY_PLAN.md` — design note for loading `PLANS/beacons.json` into the default registry.
- `PLANS/beacons.example.json` — example file shape to seed named beacons.
- `tests/test_beacons_registry_loading.py` — **skipped** placeholder; will activate once the loader is implemented next turn.

**Next up (queued)**
- Implement `load_registry_from_file(path)` in `agrm.snap.beacons` and activate the skipped test, keeping runtime isolated and deterministic.


---

## Pass 17 — Beacon Registry Loader (runtime) + Prep Next (Safe‑Cube DSL)
**Beacons runtime**
- `agrm/snap/beacons.py` now supports `load_registry_from_file(path)` to seed the default registry from `PLANS/beacons.json` (or any JSON path). Shape: `{ name: {vec: [...], neg: bool}, ... }`.
- Existing helpers remain: `set_registry_data`, `get_registry`.

**Tests**
- `tests/test_beacons_registry_loading.py` — loads from a temp JSON, verifies lookups, and hot-swaps registry contents.

**Project beacons file**
- `PLANS/beacons.json` initialized with `topicA`, `topicB` (neg), `topicC` examples.

**Prep for next step: Safe‑Cube DSL**
- `PLANS/SAFE_CUBE_DSL_PLAN.md` — design for a minimal, auditable allow/deny DSL bound to Trails.
- `tests/test_safe_cube_dsl_skeleton.py` — skipped placeholder to be enabled when the sentinel runtime lands.

**Next (queued)**
- Implement `SafeCubeSentinel` runtime per plan (file‑backed rules, deterministic decisions, emit Trails), then enable the test.


---

## Pass 18 — Safe‑Cube DSL Runtime + Prep Next (Porter custody/DLQ)
**Runtime**
- `src/mannequin/api.py` implements **SafeCubeSentinel** with `load_rules(path)` and `allow(op, context)`.
  - Decisions are **deny-by-default**; matching rules allow/deny with optional reason.
  - Every decision emits Trails (`begin` → `append` decision → `finalize`).
  - `ops.policy.api.POLICY_HASH` is attached to the Trail.
- Module helpers: `load_rules(path)`, `get_sentinel()`; a tiny **Porter** seed remains (custody semantics planned next).

**Rules file**
- `PLANS/safe_cube_rules.json` seeded with sample allow/deny entries.

**Tests**
- `tests/test_safe_cube_dsl_runtime.py` — loads a temp ruleset, asserts allow/deny outcomes, validates that all Trail events conform to schema.

**Prep Next**
- `PLANS/PORTER_CUSTODY_PLAN.md` — design for at‑least‑once, custody log, retries, dead‑letter queue.
- `tests/test_porter_custody_skeleton.py` — skipped placeholder to enable when custody semantics land.


---

## Pass 19 — Porter Custody & Dead‑Letter (runtime) + Prep Next (Archivist v1)
**Runtime**
- `src/mannequin/api.py` — **Porter** now provides in‑memory delivery with:
  - `register_sink(name, fn)` to attach consumers.
  - `deliver(payload, *, to, retries=0)` with custody log, retry, and **dead‑letter queue** on persistent error.
  - Trails: `begin`/`append porter.attempt` and `append porter.error` (on failure) → `finalize`.
  - Helpers: `drain(to)` (custody view), `dead_letters(to)`.

**Tests**
- `tests/test_porter_success.py` — successful delivery path; Trails validated.  
- `tests/test_porter_dead_letter.py` — failing sink moves item to DLQ; error Trail appended.

**Prep Next (queued)**
- `PLANS/ARCHIVIST_V1_PLAN.md` — plan to implement deterministic expand/contract around `SNAP_V1_SCHEMA.json` with idempotent round‑trip and Safe‑Cube guardrails.
- `tests/test_archivist_roundtrip_skeleton.py` — skipped placeholder; will enable when Archivist lands next.


---

## Pass 20 — Archivist v1 (expand/contract) + Prep Next (E‑DBSU runtime)
**Runtime**
- `src/agrm/snap/archivist.py` implements **deterministic** `contract(expanded)` and `expand(snap_id)`:
  - **Contract**: computes `payload_hash` from canonical JSON of the expanded context; writes a Snap (id, parent_ids, glyph, context, n_level, shell, optional weyl_index/universe_ref, trail_id, created_at). Emits Trails.
  - **Expand**: rehydrates from an in‑memory registry; validates minimal Snap shape; emits Trails.
  - **Safe‑Cube guard**: invalid/unknown Snap routes op as `archivist.expand.invalid_schema`; decision is enforced by the sentinel (deny‑by‑default).

**Tests**
- `tests/test_archivist_roundtrip_runtime.py` — round‑trip idempotence (expanded → contract → expand) and validation of Trails. Also injects a malformed Snap to confirm Safe‑Cube denial path.

**Prep — E‑DBSU**
- `PLANS/E_DBSU_RUNTIME_PLAN.md` — lifecycle API plan and signal sources.
- `tests/test_edbsu_runtime_skeleton.py` — skipped placeholder to activate on next pass.

**Next up (queued)**
- Implement **E‑DBSU runtime** with Trails & Safe‑Cube integration (NEW→TRIAGED→PROMOTED/DEMOTED→ARCHIVED), along with property‑style tests for lifecycle transitions and evidence signals.


---

## Pass 21 — E‑DBSU Runtime + Inclusiveness Review (inverse search & expanded keywords)
**Runtime**
- `src/agrm/snap/edbsu.py` — E‑DBSU lifecycle with Trails & Safe‑Cube:
  - `submit(candidate)→id` → `NEW`
  - `triage(id, evidence)` → `TRIAGED` (records signals like `corroboration`, `neg_conflict`)
  - `promote(id)` → `PROMOTED` iff `TRIAGED` and `corroboration > neg_conflict`, else `DEMOTED` with reason
  - `demote(id, reason)` → `DEMOTED`; `archive(id)` → `ARCHIVED`
  - Helper: `get_state(id)`

**Tests**
- `tests/test_edbsu_runtime.py` — happy path `NEW→TRIAGED→PROMOTED→ARCHIVED` with Trails validation; conflict path demotes.

**Inclusiveness Review (over provided docs)**
- Extracted a **lexicon** and **multi‑word phrases**; collected **negation/inverse** lines for inspection.
- Mapped **system terms to repo coverage** to spot gaps.
- Generated **expanded keyword variants** (hyphen/underscore/case/no‑sep) to keep future scans broad.

**Artifacts** (`PLANS/INCLUSIVENESS/`)
- `lexicon_top_200.csv` — most frequent tokens.
- `phrases_top_200.csv` — most frequent 2–4‑grams (biased to domain anchors).
- `negation_lines.json` — lines containing `not/no/never/inverse/neg/deny/unsafe/unknown/double unknown/ambiguous`.
- `system_term_coverage.csv` — term→files coverage map.
- `expanded_keyword_queries.csv` — broadened variants for future search passes.

**Immediate observations (coverage)**
- Covered in repo: `MDHG`, `Trails`, `Beacons`, `W5H`, `Porter`, `Archivist`, `SafeCube/Mannequin`, `TPG` (config), `E‑DBSU`.
- Likely **gaps to fill** soon: `ThinkTank`, `DTT`, `Assembly Line`, `MORSR`, `Wave Pool`, `Snap Ops Center` (explicit module), `Scout Arms` harness.

**Next up (queued)**
- Implement **ThinkTank** minimal runtime with Trails + a tests‑only harness for panel critiques (then DTT & Assembly Line seeds).  
- Continue inclusiveness sweep using the expanded queries to backfill remaining gaps.


---

## Pass 22 — ThinkTank Runtime + Prep Next (DTT & Assembly Line)
**Runtime**
- `src/thinktank/api.py` — `ThinkTank.critique(findings, evidence)`
  - Deterministic heuristic yields a score and a step list (seed → graph → score → E‑DBSU triage or Archivist contract → Porter handoff to DTT harness).
  - Safe‑Cube gated; Trails for begin/proposed/finalize.

**Tests**
- `tests/test_thinktank_runtime.py` — basic critique path validates Trails; explicit denial path when Safe‑Cube rules disallow.

**Plans & Skipped Tests (queued)**
- `PLANS/DTT_PLAN.md` & `tests/test_dtt_skeleton.py` — Deterministic Test Tooling dry‑run harness.
- `PLANS/ASSEMBLY_LINE_PLAN.md` & `tests/test_assembly_line_skeleton.py` — Stage work orders from DTT transcripts.

**Next up (queued)**
- Implement `dtt.api.run_steps(steps, mode="dry")` and `assembly.api.stage(transcript)` with Trails + property tests, then wire ThinkTank → DTT → Assembly happy path via Porter.


---

## Pass 23 — DTT (dry‑run) & Assembly Line runtimes + E2E
**Runtimes**
- `src/dtt/api.py` — `run_steps(steps, mode="dry")` returns a deterministic **Transcript**; Trails: `dtt.run` + `dtt.step` + `dtt.summary` → `dtt.done`.
- `src/assembly/api.py` — `stage(transcript)` returns a **WorkOrder** with `modules` (from op heads), `ops`, and `counts`; Trails covered.

**Tests**
- `tests/test_dtt_runtime.py` — determinism and Trails validation.
- `tests/test_assembly_runtime.py` — stages modules from a transcript; Trails validation.
- `tests/test_thinktank_dtt_assembly_e2e.py` — end‑to‑end: ThinkTank → DTT (dry) → Assembly; Trails validation.

**Next (suggested)**
- Wire Porter sink `dtt.harness` in a tests‑only integration to accept `{steps:[...]}` payload and feed `run_steps`, then deliver the transcript to an `assembly.harness` sink — closing the loop via Porter for a full orchestration path.


---

## Pass 24 — Porter‑wired Orchestration (ThinkTank → DTT → Assembly)
**What’s added**
- **E2E via Porter sinks** (tests‑only): `tests/test_porter_orchestration_e2e.py`
  - Registers `dtt.harness` sink → runs `dtt.run_steps(steps, mode="dry")` and then hands the **Transcript** to `assembly.harness`.
  - `assembly.harness` sink stages a **WorkOrder** via `assembly.stage(transcript)`.
  - End‑to‑end asserts: Transcript is produced, WorkOrder includes expected modules (e.g., `mdhg`, `archivist`), and all Trails validate.

**Why this matters**
- Exercises the orchestration loop using the **same Porter semantics** we’ll use in the larger system: ThinkTank proposals → dry‑run (DTT) → staging (Assembly) with custody + Trail coverage.

**Next (suggested)**
- Add a **failure‑injection** variant (include a `{"op":"fail.synthetic"}` step) to prove DLQ behavior in orchestration context.
- Begin stubs for **MORSR** and **Wave Pool** plus a **Snap Ops Center** entry, covering the gaps surfaced in the Inclusiveness review.


---

## Pass 25 — Failure‑Injection Orchestration + MORSR/WavePool/SnapOps stubs + Stage Gap Review
**DLQ orchestration test**
- `tests/test_porter_orchestration_failure_injection.py` — injects `{"op":"fail.synthetic"}` into steps; DTT sink raises on non‑OK transcript ⇒ **Porter DLQ** captures payload; Trail `porter.error` asserted.

**New runtime stubs (Trails‑covered)**
- `src/morsr/api.py` — `analyze(transcript) -> RegionMap` (probability mass over op‑heads derived from transcript counts).
- `src/wavepool/api.py` — `pool_candidates({id: score}) -> Ranked` (descending).
- `src/snapops/api.py` — `orchestrate(request) -> OrchestrationPlan` (ThinkTank‑like step plan, tagged `(soc)`).

**Tests**
- `tests/test_morsr_runtime.py` — regions sum to ~1, Trails valid.
- `tests/test_wavepool_runtime.py` — ranking order, Trails valid.
- `tests/test_snapops_runtime.py` — orchestrate outputs plausible steps, Trails valid.

**Remaining gap review (this stage)**
- **Coverage table updated** (see workspace: *Pass 25 — Updated System Term Coverage*). Newly covered: **MORSR**, **Wave Pool**, **Snap Ops Center**.  
- Still **light/ambiguous**: domain nouns like `superpermutation`, `universes`, fine‑grained `to_graph`/`to_points` op symbols (appear mostly in tests/spec text rather than code), and naming variants (`SafeCube`, hyphen/underscore drift for E‑DBSU). These are lexical gaps, not runtime blockers; we’ll pick them up as we encode those features.

**Next (queued)**
- Add a **Porter failure‑chain E2E** variant that proceeds to Assembly **only** on success; assert absence of Assembly Trails when DLQ triggers.
- Begin **Superpermutation** interop shim (read‑only) to register solver hooks without executing search; add a smoke test.


---

## Pass 26 — No‑Assembly‑On‑Fail Orchestration + Superpermutation Interop (read‑only)
**E2E orchestration (strict)**
- `tests/test_porter_no_assembly_on_fail.py` — registers Porter sinks and **does not forward** to Assembly when DTT transcript reports `ok=False`.
  - Asserts **no Assembly Trails** are present and the Assembly sink is never invoked.

**Superpermutation interop shim**
- `src/superperm/interop.py` — `summarize(config) -> Summary` (read‑only). Extracts `n`, `alphabet_size`, `lower_bound`, `upper_bound`, plus `hints`; emits `superperm.summarize → superperm.summary → superperm.done` Trails.
- `tests/test_superperm_interop.py` — smoke test for the summarizer with Trail validation.

**Why this matters**
- Locks a guardrail: orchestration **must halt before Assembly** when upstream validation fails.
- Adds a safe on‑ramp for superpermutation artifacts without triggering heavy compute.

**Next (suggested)**
- Optional: add a ThinkTank hook to emit a `superperm.summarize` step when `n≥5` signals appear in findings, keeping it non‑executing (DTT‑only) unless elevated.


---

## Pass 27 — MDHG Ops (`to_points`, `to_graph`) + ThinkTank Superperm Hook + Prep (Universes/Lattices)
**Runtime — MDHG ops**
- `src/agrm/mdhg/ops.py`:
  - `to_points(universe=None, seed=None) -> Points` — deterministic point generation (Trails: `mdhg.to_points` → `mdhg.points` → `mdhg.to_points.done`).
  - `to_graph(points, quotas={edges:8}) -> Graph` — deterministic sequential‑neighbor edges up to quota (Trails: `mdhg.to_graph` → `mdhg.graph` → `mdhg.to_graph.done`).

**Tests**
- `tests/test_mdhg_ops_runtime.py` — determinism of `to_points` and monotonicity of `to_graph` edges; Trail validation.

**ThinkTank enhancement**
- `thinktank.api` now emits a **non‑executing** `{"op":"superperm.summarize"}` step when findings imply superpermutation context (`"superperm"` present or parsed `n>=5`).
- Test: `tests/test_thinktank_superperm_hook.py` asserts presence of the step when signaled.

**Prep Next**
- `PLANS/UNIVERSES_LATTICES_PLAN.md` — skeleton API for Universe/Lattice + Weyl indexing; registry helpers; test guidance.
- `tests/test_universes_lattices_skeleton.py` — skipped placeholder to enable next.

**Next (queued)**
- Implement `agrm.space` skeleton per plan (Universe, Lattice, `weyl_index` stub, default registry) and integrate tags into Archivist + MDHG seeds.


---

## Pass 28 — Universes/Lattices (`agrm.space`) + MDHG Integration
**Runtime**
- `src/agrm/space/api.py` — new **Universe** (name, hemispheres) and **Lattice** (name, dim, `weyl_index(point)` stub) with helpers:
  - `get_default_universe()` / `set_default_universe(u)`
  - `tag_points(universe, coords)` (NW/NE/SW/SE by quadrant)
  - `log_universe(u)` / `log_lattice(l)` (Trail stubs)

**MDHG integration**
- `agrm.mdhg.ops.Points` gains optional `hemi: List[str]`.
- `to_points(...)` now tags hemispheres via `agrm.space.api.tag_points(...)` when possible.

**Tests**
- `tests/test_space_runtime.py` —
  - `Lattice.weyl_index` determinism
  - `Universe` hemisphere tagging + MDHG `to_points` attachment
  - Archivist passthrough of `universe_ref` and `weyl_index`

**Next (suggested)**
- Add ThinkTank/DTT step to stamp `universe_ref` on proposals when default universe is set; optional Trail assertions.
- Introduce a minimal **Scout Arms** harness that uses hemisphere tags to schedule rendezvous checkpoints.


---

## Pass 29 — Agent & Snap Fingerprints (Chain‑of‑Custody) + Universe Stamping + Scout Arms
**New infra**
- `src/ops/agents/api.py` — `agent_fingerprint(agent)` (deterministic under current `POLICY_HASH`).
- `src/ops/chain/api.py` — `make_chain(agent, snap_id=None, snap_fp=None)`; lazy lookup of Snap fingerprint via Archivist.

**Archivist updates**
- `contract(...)` now stamps `snap["fingerprint"] = sha256(payload_hash|POLICY_HASH)`.
- Exposes `get_fingerprint_by_id(snap_id)`.
- Trails `begin` payloads now include `chain = {agent_fp:..., snap_fp:...}` for `expand()` (auto‑looked up) and for `contract()` (agent fp).

**Porter updates**
- Trails carry `chain.agent_fp` for `porter.deliver`.
- `porter.attempt` appends include `snap_fp` when present in payload.

**Agent chain on begin (added where relevant)**
- DTT (`dtt.run`), Assembly (`assembly.stage`), ThinkTank (`thinktank.critique`), E‑DBSU ops — all begin Trails now include `chain.agent_fp`.

**Universe stamping**
- ThinkTank stamps `universe_ref` into proposed steps when a default Universe is set; DTT echoes it in transcript meta.

**Scout Arms (harness)**
- `src/scout/api.py` — `sweep(points_hemi)` emits `rendezvous` Trail events per unique hemisphere; returns `RendezvousReport`.

**Tests**
- `tests/test_fingerprints_chain_runtime.py` — Snap fingerprint exists and appears in `expand(...)` begin Trail; Porter attaches `snap_fp`.
- `tests/test_universe_stamping_runtime.py` — ThinkTank stamps `universe_ref` across steps.
- `tests/test_scout_runtime.py` — Scout emits `rendezvous` events by hemisphere.

**Next (queued)**
- Add optional `snap_id` propagation through steps/transcripts to make chain queries across DTT/Assembly trivial when a specific Snap is the subject.
- Expose a `trails.query.by_fingerprint(agent_fp, snap_fp)` helper (read‑only) to reconstruct decision chains rapidly.


---

## Pass 30 — Snap‑ID Propagation + Trail Query Helper
**Chain‑of‑custody propagation**
- **ThinkTank**: `critique(..., subject_snap_id=None)` (backward‑compatible). When provided, stamps `snap_id` into every proposed step.
- **DTT**: TranscriptEntry `meta` now preserves `snap_id` (and `universe_ref`) from steps.
- **Assembly**: `WorkOrder` now includes `snap_ids` (unique set derived from transcript entries).
- **Porter**: if a payload has only `snap_id`, auto‑enriches `porter.attempt` append with `snap_fp` by resolving via Archivist.

**Trail query helpers**
- `src/trails/query/api.py`:
  - `filter_events(events, agent_fp=None, snap_fp=None)` — filters Trail events matching agent and/or snap fingerprints (checks `payload.chain` and `payload.snap_fp`).
  - `summarize_ops(events)` — quick op histogram for the filtered slice.

**Tests**
- `tests/test_chain_propagation_runtime.py` — creates a Snap → ThinkTank with `subject_snap_id` → DTT → Assembly; asserts `snap_id` present through to `WorkOrder.snap_ids` and Trails validate.
- `tests/test_trails_query_runtime.py` — emits events and validates agent‑fp filtering and op summarization.

**Next (suggested)**
- Add a `porter.inspect(chain: {agent_fp?, snap_fp?})` debug helper that uses the query API to return a short chain summary (ops, first/last timestamps, modules touched).
- Consider indexing Trails to a light store (in‑memory ring buffer) for O(1) queries during tests.


---

## Pass 31 — Porter.inspect + Trails Ring Buffer (opt‑in) + Tests
**New**
- `src/trails/query/buffer.py` — lightweight in‑memory ring buffer for Trail events: `configure(maxlen)`, `ingest(events)`, `snapshot()`, `clear()`.
- `mannequin.api.Porter.inspect(agent_fp=None, snap_fp=None, drain=True)` — convenience inspector that drains Trails (optional), ingests into the ring buffer, filters via `trails.query.api.filter_events`, and returns `{n_events, counts}`.

**Tests**
- `tests/test_porter_inspect_runtime.py` — verifies `inspect()` returns a count and op histogram after a ThinkTank→DTT sequence.

**Notes**
- This keeps the core Trails API untouched while providing a fast, test‑friendly query surface. We can later swap `buffer.py` for a pluggable store if needed.


---

## Pass 32 — Inspector Enrichment + MDHG Scoring + Shipping
**Inspector enrichment**
- `Porter.inspect` now returns: `modules`, `first_idx`, `last_idx`, and (if available) `first_ts`/`last_ts`, in addition to `{n_events, counts}`.

**MDHG scoring surface**
- `agrm.mdhg.ops.promotion_breakdown(Graph) -> PromotionBreakdown` (degree‑based score + coord tie‑breaker), Trails‑covered.
- Tests extended in `tests/test_mdhg_ops_runtime.py`.

**Release packaging**
- Created `PLANS/RELEASE_NOTES_PASS32.md` (file manifest).
- **Repo ZIP** ready: `SnapLat_Pass32.zip` (root of your workspace).

**Next (optional)**
- If desired, I can add a minimal CLI entry point to exercise ThinkTank → DTT → Assembly end‑to‑end and print a tiny chain report using `Porter.inspect`.


---

## Pass 33 — Snap Ops Center (policy‑aware) + CLI Demo + Shipping
**Snap Ops Center (policy‑aware orchestrator)**
- `snapops.api.run(request) -> OrchestrationResult`:
  1) ThinkTank critique → step plan (optional `subject_snap_id` propagation).
  2) DTT dry‑run; Trail `snapops.transcript` with `ok` + counts.
  3) **Branch:**
     - **Success:** `assembly.stage(transcript)` → Trail `snapops.assembly` → `ok=True`.
     - **Failure:** submit candidate to **E‑DBSU** and `triage(...)` → Trail `snapops.edbsu` → `ok=False`.
  - Trails are SnapOps‑fingerprinted via `chain.agent_fp`.

**CLI demo**
- `python -m snapops.demo`:
  - Flags: `--inject-fail`, `--snap-id`.
  - Auto‑loads rules to allow ThinkTank.
  - Prints result summary and a SnapOps‑filtered Trail op histogram using the query buffer.

**Tests**
- `tests/test_snapops_policy_runtime.py` — asserts both branches (success → WorkOrder; failure → E‑DBSU triage) behave as designed.

**Shipping**
- **Repo ZIP updated:** `SnapLat_Pass33.zip` in your workspace root.


---

## Pass 34 — Staged: Makefile/CLI → SnapOps Policy Knobs → Contracts Doc + Shipping
**A) Makefile & CLI wrappers**
- `Makefile` at repo root with targets: `demo-success`, `demo-fail`, `inspect`, `package`.

**B) SnapOps policy knobs & flows**
- `snapops.api.Policy` with `triage_threshold`, `retry_on_fail`, `max_retries`, `archive_on_persist`.
- `snapops.api.run(request)` now honors `policy`: on failure it **submits & triages** to E‑DBSU, optionally **salvage‑retries** by removing `fail.*` steps, then **promotes** on recovery or **demotes/archives** on persistent failure. Trails: `snapops.salvage`, `snapops.promoted`, `snapops.persist`.
- Tests: `tests/test_snapops_policy_knobs.py`.

**C) Code‑review artifacts**
- `PLANS/CODE_REVIEW_PASS34.md` — naming, contracts, Trails conventions, and non‑breaking migration plan.

**Shipping**
- **Repo ZIP:** `SnapLat_Pass34.zip` available in your workspace.


---

## Pass 35 — README + CI + Naming Constants (start) + Taxicab Research Doc + Shipping
**Docs & DX**
- `README.md` (quickstart, demos, tests, packaging, structure & contracts snapshot).
- GitHub Actions: `.github/workflows/ci.yml` (Python 3.11 + pytest).

**Non‑breaking naming migration**
- `ops/contracts/naming.py` constants for common op strings; **MDHG ops** updated to use constants internally (values unchanged).

**Research**
- `PLANS/TAXICAB_EVAL.md` — experimental plan to evaluate **1729** as a coverage/completeness watermark.

**Shipping**
- **Repo ZIP:** `SnapLat_Pass35.zip` now available in your workspace.


---

## Pass 36 — Taxicab Threshold Harness + Coverage Gating + MDHG `to_points_n`
**MDHG**
- Added `to_points_n(n:int, seed:Optional[int]) -> Points` to generate synthetic universes of arbitrary size.

**Metrics**
- New `agrm.metrics.coverage` with `degree_distribution`, `redundancy_ratio`, `neg_conflict_proxy`.

**SnapOps Policy**
- Extended `Policy` with `coverage_watermark=1729` and `require_coverage=True`.
- `snapops.run(...)` now honors `coverage_k` in the request; if `transcript.ok` but `coverage_k < watermark`, it emits `snapops.coverage_gate` and routes a provisional candidate to E‑DBSU (triage), returning `ok=False`.

**Experiment harness**
- `experiments/taxicab_threshold.py` sweeps k∈{256,512,1024,1729,2048,4096}; computes redundancy/neg_conflict proxies; drives SnapOps with coverage gating; prints JSON metrics.
- Auto‑loads a rule to allow `thinktank.critique` for the run.

**README update**
- Added **Experiments** section with one‑liner to run the sweep and interpret metrics.

**Shipping**
- **Repo ZIP:** `SnapLat_Pass36.zip` added to workspace.


---

## Pass 37 — Approaching‑Gate Consolidation + Plots + Shipping
**Approaching‑Gate phase**
- **Policy (SnapOps):** `approach_margin` (default 0.15), `shell_checks` (default 2), `allow_override_on_consolidate` (default True).
- **Behavior:** If `transcript.ok` but coverage is within margin below the watermark, run a **consolidation loop** (Archivist round‑trip shelling checks) and re‑review. On success, allow proceed below watermark and emit `snapops.consolidate_ok`; otherwise gate to E‑DBSU with `phase=approach`.

**Shelling checks**
- `agrm.metrics.shelling.check_shelling(payload)` — contracts → expands and verifies invariants.

**Experiment & plotting**
- `experiments/taxicab_threshold.py` rewritten for clarity: `run_sweep(..., integrated=False)` by default.
- `experiments/plot_taxicab.py` — generates two matplotlib plots (redundancy vs k; coverage gate trigger vs k).
- Inline plots attached in this turn from a non‑integrated sweep.

**Shipping**
- **Repo ZIP:** `SnapLat_Pass37.zip` is prepared in your workspace.


---

## Pass 38 — Approach‑Phase Transcript Meta + CLI Policy Knobs + Plot Export
**DTT transcript meta**
- Transcript now includes a `meta` dict; SnapOps marks `transcript.meta["phase"] = "approach"` when the **Approaching‑Gate** path allows proceed via consolidation.

**CLI (snapops.demo)**
- Added flags: `--coverage-k`, `--watermark`, `--approach-margin`, `--shell-checks`, `--no-override`, `--no-require-coverage`, `--retry`, `--max-retries`.
- Policy is assembled from flags and passed into `snapops.run(...)`.

**Plot helper**
- `experiments/plot_taxicab.py` now accepts `--out-dir` and saves `redundancy_vs_k.png` and `coverage_gate_vs_k.png` in that folder (in addition to showing the figures).

**Shipping**
- **Repo ZIP:** `SnapLat_Pass38.zip` added to workspace.


---

## Pass 39 — Monte Carlo / Markov **Toggle Class** (auto‑enable when space struggles)
**New**
- `ops/heuristics/toggle.py`
  - `MCPolicy` (enable_auto, triggers, modes, budget)
  - `should_enable(...)` — decides if heuristics should turn **on** based on `neg_conflict`, failure counts, and coverage status.
  - `plan_modes(...)` — sanitizes requested modes (random_walk, mcmc, smc, anneal).
- `agrm/mdhg/randomwalk.py` — stub `rank_by_random_walk(...)` (degree‑biased stationary estimate placeholder).
- `ops/mc/{mcmc,smc,anneal}.py` — lightweight stubs exposing `suggest(...)` (for later expansion).

**SnapOps integration**
- **Policy fields:** `mc_enable_auto`, `mc_trigger_neg_conflict`, `mc_trigger_min_failures`, `mc_modes`, `mc_budget`.
- **Runtime:** when transcript fails (or space is otherwise struggling) and policy enables auto, SnapOps emits `snapops.mc.enabled` with selected modes/budget (advisory heuristic layer; does not alter steps yet).

**Tests**
- `tests/test_snapops_mc_toggle.py` — asserts the `snapops.mc.enabled` event appears under high `neg_conflict` with an injected failure.

**Docs**
- README section **“Monte Carlo / Markov toggle”** with a policy JSON example and behavior description.

**Shipping**
- **Repo ZIP:** `SnapLat_Pass39.zip` now available in the workspace.


---

## Pass 40 — E8‑aware RW kernel + Scout Hints + MCMC Diagnostics (advisory, policy‑gated)
**Random‑walk kernel (E8‑aware)**
- Power‑iteration with teleportation; optional degree/Weyl/Hemisphere biases.
- Location: `agrm/mdhg/randomwalk.py` → `power_iter_rw`, `rank_by_random_walk`.

**Scout integration**
- `scout/hints.py` → `rw_hints(...)` computes top‑k nodes and emits `scout.rw.hints` Trail.
- SnapOps: when MC toggle enables and includes `random_walk`, it computes advisory hints (no plan mutation yet).

**MCMC diagnostics scaffold**
- `ops/mc/mcmc.py` → `run(...)` and `diagnostics(...)` (R‑hat, ESS approximations) to gate future advisory hints.

**Docs & Tests**
- README: **E8‑aware stochastic advisory** section with policy knobs.
- Tests: `tests/test_scout_rw_hints.py` smoke test for hint emission.

**Shipping**
- **Repo ZIP:** `SnapLat_Pass40.zip` added to workspace.


---

## Pass 41 — E8 tags → RW kernel; diagnostics‑gated MCMC hints; SMC scaffold
**RW kernel**
- `agrm/mdhg/randomwalk.py` now derives **Weyl/Hemisphere tags** from Universe when available (fallback: cyclic tags), and offers `rw_scores_with_e8(...)`.

**ThinkTank advisory**
- Diagnostics‑gated **MCMC hints**: ThinkTank emits `thinktank.mcmc.diag` always; emits `thinktank.mcmc.hints` only when **ESS ≥ min** and **R‑hat ≤ max**.

**SMC scaffold**
- `ops/mc/smc.py` provides `suggest_stream(...)`; SnapOps emits `scout.smc.hints` under **sub‑coverage** when mode includes `"smc"`.

**Policy knobs**
- Added: `mcmc_ess_min`, `mcmc_rhat_max`, `smc_particles_max`, `smc_resample_threshold`, `rw_alpha`, `rw_lambda_w`, `rw_lambda_h`.

**Tests & Docs**
- Tests for RW hints, MCMC hinting, and SMC hint emission.
- README updated with thresholds and RW knobs.

**Shipping**
- **Repo ZIP:** `SnapLat_Pass41.zip` is available in your workspace.


---

## Pass 42 — Schemas, Advisory Adoption (opt‑in), Strict RW Tags
**Contracts/Schemas**
- `ops/contracts/schemas.py`: lightweight validators for `Step`, `Transcript`, `WorkOrder` (pydantic‑style without external dep).

**Advisory adoption (opt‑in)**
- Policy knobs: `adopt_advisory` (default False), `rw_require_true_tags` (default False).
- If enabled and RW hints exist, SnapOps attempts **stable reordering** of `request.findings` when entries match `node:<id>`/`#<id>`; logs `snapops.advisory.adopted`.

**Strict RW tag requirement**
- `randomwalk.derive_tags(...)` now returns `(weyl, hem, true_tags)`. If `rw_require_true_tags=True` and tags are synthetic, RW hints are skipped.

**Docs/Tests**
- README updated with sections for Schemas & Advisory adoption.
- Tests: schema validators + adoption smoke test.

**Shipping**
- **Repo ZIP:** `SnapLat_Pass42.zip` available in workspace.


---

## Pass 43 — Stage Step 4 (Coverage Empirics harness) + Post‑mortem prep
**Harness**
- `experiments/coverage_empirics.py`: runs redundancy/neg‑conflict over a directory of `.txt/.md`, flags `coverage_gate_triggered`, outputs CSV/JSON & plots.
- `ingest/text.py`: simple co‑occurrence graph builder (windowed). Uses existing coverage metrics.

**Sample run (toy)**
- `sample_docs/` + `sample_plots/` created; verified end‑to‑end entrypoint.

**Post‑mortem index**
- Added tabular summary of Pass 36–43 to support detailed write‑up.

**Ship**
- `SnapLat_Pass43.zip` includes the harness and sample outputs.


---

## Pass 44 — Schema Guards, Improved `neg_conflict`, Scout Queue Advisory Adoption
**Schema validators (wired)**
- SnapOps/ThinkTank/DTT now invoke lightweight validators from `ops/contracts/schemas.py` (non‑breaking try/except guards) to catch payload drift at boundaries.

**Improved `neg_conflict`**
- `agrm/metrics/coverage.neg_conflict(...)` uses **negative triangle rate** when edge signs are present; otherwise falls back to a clustering×redundancy complement proxy. Optional `bridges` upweights conflicts at bridge edges.

**Scout queue advisory adoption**
- `scout/queue.py` provides `reorder_by_hints(...)` + `extract_node_id(...)`. When policy opts in, Scout queues can be re‑ordered by RW rankings; Trails emit `scout.queue.reordered`.

**Docs & Tests**
- README expanded (neg_conflict and Scout queue adoption).
- Tests for `neg_conflict` and queue reordering added.

**Shipping**
- **Repo ZIP:** `SnapLat_Pass44.zip` available in workspace.


---

## Pass 45 — Doc‑Intake Prep (Authorship, Segmenter, Idea Packager, Lexicon, Code Extractor)
**Authorship rules**
- `ingest/log_authorship.py`: headers (`You said:` → human; `ChatGPT said:` / `thought for …` → AI) with fallback style cues.

**Segmenter**
- `ingest/log_segmenter.py`: produces role‑labeled segments with char ranges + mini‑hash.

**Idea Packager**
- `ingest/idea_packager.py`: groups segments into Human‑led idea packages; pivots on artifact drops / scope changes.

**Lexicon & Code**
- `ingest/lexicon.py`: grows uni/bi/tri‑gram lexicon (never narrowing).
- `ingest/code_extract.py`: pulls fenced code blocks (lang + body) per package.

**Indexer & CLI**
- `ingest/indexer.py` + `experiments/ingest_run.py`: build JSON/CSV indices per file.
- Ran against uploaded logs; indexes saved alongside inputs.

**Shipping**
- **Repo ZIP:** `SnapLat_Pass45.zip` added to workspace.


---

## Pass 46 — LSDT Guardrails + Tagging + Intake Hooks
**Intent**: Treat **LSDT (Layered Simulation Dimensional Theory)** as **reference‑only**: useful datapoints for experiments, **not** part of core SnapLat design.

**What’s new**
- `ingest/theories.py` — LSDT tagger (regex spans) wired into `ingest/indexer.py` output as `theory` entries per package.
- SnapOps **policy knob**: `lsdt_reference_only=True` (default). If LSDT is present in `findings`, advisory **adoption is blocked**, and we log `policy.blocked.lsdt`.
- Test `test_policy_lsdt_block.py` ensures blocking behavior or absence of `adopted` event when LSDT appears.

**Optional pre‑index**
- If present, zipped **Manus AI work on E8** is listed; any `.txt/.md` got indexed with LSDT spans summarized (for datapoint use only).

**Shipping**
- **Repo ZIP:** `SnapLat_Pass46.zip` is available.


---

## Pass 47 — Manus Zip Intake Triage
**What I did**
- Extracted the provided zip, built per‑file indices for all `.txt/.md/.rst` using the intake pipeline (authorship → segments → idea packages → lexicon → code → LSDT spans).
- Scored each file for **direct relevance** to SnapLat (E8/Weyl/MDHG/AGRM/SnapOps/Scout/ThinkTank/DTT/Assembly/Archivist/Porter/Mannequin/fingerprints/Trails/superpermutation/glyph/shelling/Universe/lattice/bridges/N=4/5/coverage/watermark/1729/random walk/MCMC/SMC/anneal/approach‑phase/E‑DBSU/morsr/wave pool/RAG).
- Flagged **LSDT mentions** (reference‑only) and **likely‑personal** items (generic heuristics; no content quoted).

**Artifacts**
- Workspace tables: Inventory, Intake Summary (scored), Top Direct Candidates, LSDT Mentions, Likely Personal.
- Provenance snapshot: `manus_zip_intake_snapshot.json` in the workspace root.

**Policy**
- LSDT remains **reference‑only**: any advisory adoption is blocked when LSDT content would centralize ordering (Trail: `policy.blocked.lsdt`).


---

## Pass 48 — Mixed Corpus Build + Coverage & Policy Triage
**Goal**: Exercise intake/gating with **direct**, **connected**, **reference‑only (LSDT)**, and **likely‑personal** documents in one run.

**What I did**
- Selected a balanced set from the Manus zip (using prior scoring) and copied into `_mixed_corpus/` with subfolders: `direct/`, `connected/`, `lsdt_reference_only/`, `likely_personal/`.
- Ran the **coverage empirics harness** per subfolder (demo watermark = **200**) and produced plots + metrics.
- Generated a **triage policy** per doc (JSON) indicating include/exclude, approach‑phase, and MC advisory modes; LSDT marked **reference‑only**; personal flagged **exclude**.

**Artifacts**
- Summary tables in workspace: file lists & coverage+triage.
- Triage JSON: `_mixed_corpus/_triage.json`.
- **ZIP**: `mixed_corpus_pack.zip` containing the mixed set and plots.

**Next**
- When you hand off the test docs, we can raise the watermark back to **1729** and re‑run this mixed‑corpus trial to observe real gating/adoption behavior under the full threshold.

