# Running To‑Do Log

**Scope:** Track *improvements/changes needed* as they arise while you describe the system. I’ll keep adding items; no implementation or re-architecture until you say you’re done explaining.

**Statuses:** P1 (must-have), P2 (should-have), P3 (nice-to-have).  
**Sections:** Open Items → (later) Resolved / Deferred.

---

## Open Items

### T-001 — Add hysteresis & migration thresholds to MDHG
- **Category:** MDHG stability
- **Why:** Prevent thrashing when hot maps fluctuate.
- **Action:** Require Δ-score > θ across N consecutive checks before reassigning buckets; add cooldown windows.
- **Priority:** P1

### T-002 — Rate limits & cooldowns for topology changes
- **Category:** Ops controls
- **Why:** Avoid churn from frequent room moves and elevator rewiring.
- **Action:** Per-building quotas, per-room move cooldowns, batch structural updates.
- **Priority:** P1

### T-003 — Canonical IDs with uncertainty & soft memberships
- **Category:** Identity resolution
- **Why:** E8/hash alone won’t solve synonymy/polysemy.
- **Action:** Maintain canonical IDs with confidence; allow rooms to belong to multiple buildings with weights.
- **Priority:** P1

### T-004 — Keep continuous embeddings alongside E8 codes
- **Category:** Representation
- **Why:** Avoid semantic loss from quantization.
- **Action:** Store (embedding, E8 code); use E8 as codebook/quantizer only.
- **Priority:** P1

### T-005 — Depth budgets & structural interning
- **Category:** Scale control
- **Why:** Recursive decomposition can explode size.
- **Action:** Cap decomposition depth; content-hash identical substructures; dedupe rooms.
- **Priority:** P1

### T-006 — Elevator lifecycle policy (creation/decay/caps)
- **Category:** Cross-links
- **Why:** Prevent superlinear edge growth; keep links meaningful.
- **Action:** Create via PMI/co-use > θ over window W; decay with half-life H; cap per-room top-k by utility.
- **Priority:** P1

### T-007 — Complexity spike handler in MDHG
- **Category:** Surge control
- **Why:** Handle collision/fan-out bursts without full rehash.
- **Action:** Temporarily expand aux-hash fanout, raise elevator budget locally, schedule background rebalancing.
- **Priority:** P1

### T-008 — Golden-angle rotations & Hamiltonian bucket ordering
- **Category:** Hash design
- **Why:** Improve locality; reduce resonance/collisions.
- **Action:** Define φ-rotated projections; traverse buckets via Hamiltonian-like path; record primary + neighbor buckets.
- **Priority:** P2

### T-009 — Sheaf/factor consistency checks across floors
- **Category:** Reasoning integrity
- **Why:** Ensure higher-level claims are supported by lower floors.
- **Action:** Compute tension scores on overlaps; flag offenders for AGRM.
- **Priority:** P2

### T-010 — MVCC snapshots + provenance traces
- **Category:** Explainability/repro
- **Why:** Answer “why this retrieval/answer?” and enable replay.
- **Action:** Append-only event log; periodic snapshots; expose trace (hash path, thresholds, elevators used).
- **Priority:** P1

### T-011 — Bootstrap strategy for cold start
- **Category:** Initialization
- **Why:** No hot maps initially.
- **Action:** Offline clustering; synthetic elevator seeding via co-occurrence/citation; warm-up queries.
- **Priority:** P2

### T-012 — Concept drift handling
- **Category:** Evolution
- **Why:** Topic shifts break buildings over time.
- **Action:** Sliding windows; shadow clusters; switch after stability tests.
- **Priority:** P2

### T-013 — Latency protections under surge
- **Category:** SRE/perf
- **Why:** Elevator creation and rebalancing can spike tail latency.
- **Action:** Batch proposals; rate-limit per building; cache negative results briefly.
- **Priority:** P2

### T-014 — Acceptance metrics & dashboards
- **Category:** Evaluation
- **Why:** Validate approach against baselines.
- **Action:** Track recall/MRR, grounding rate, p50/p95 latency, migration rate, elevator lifetime, trace render time, storage/compute cost.
- **Priority:** P1

### T-015 — Interfaces for MDHG/AGRM/elevators/planner
- **Category:** API design
- **Why:** Decouple components; enable testing.
- **Action:** Define methods: mdhg.assign/update_feedback; elevator.propose; planner.plan; graph.retrieve; sheaf.check.
- **Priority:** P1

### T-016 — Health metrics for structures
- **Category:** Observability
- **Why:** Early warning on instability.
- **Action:** Edge tension distributions, bucket collision rate, move rates, hotness volatility.
- **Priority:** P2

---

## Reserved for new items (append-only during your brief)

### Newly added
- T-017 — Corpus manifest & fingerprinting
  - **Why:** We have ~350 items (156 .py, 158 .txt, etc.). Need a definitive inventory for reproducibility.
  - **Action:** Generate tree with sizes, SHA-256, mtimes; tag by family (mdhg/agrm/ConStruct/legacy/etc.).
  - **Priority:** P1

- T-018 — Canonicalization & dedup strategy
  - **Why:** Multiple versions/forks and duplicate documents exist.
  - **Action:** Choose canonical per family via recency + completeness; record redirects from aliases.
  - **Priority:** P1

- T-019 — Dependency map & missing references
  - **Why:** AGRM imports `superbuilder.*`; need a full codegraph and stub policy.
  - **Action:** Build import graph; flag unresolved/missing; propose stubs.
  - **Priority:** P1

- T-020 — Static quality scan & guardrails
  - **Why:** Legacy code spans styles; need linting, types, complexity caps.
  - **Action:** Run lint/type checks; set targets for cyclomatic complexity; note high-risk modules.
  - **Priority:** P2

- T-021 — Execution harness & smoke tests (n≤5)
  - **Why:** Validate legacy superpermutation generators and analysis scripts.
  - **Action:** Deterministic seeds; capture outputs as fixtures; compare across versions.
  - **Priority:** P1

- T-022 — Architecture crosswalk (code ↔ concepts)
  - **Why:** Map MDHG/AGRM/Buildings-Floors-Rooms-Elevators to concrete modules/files.
  - **Action:** Produce a matrix linking primitives → modules → tests.
  - **Priority:** P1

- T-023 — Provenance & version labeling
  - **Why:** Many files lack headers; future confusion risk.
  - **Action:** Add file headers or external manifest with origin/date/owner.
  - **Priority:** P2

- T-024 — Policy spec for dynamic topology changes
  - **Why:** Needs explicit thresholds for hashing/elevator churn.
  - **Action:** Draft YAML schema for thresholds, hysteresis, cooldowns.
  - **Priority:** P1

### Newly added (from file reviews)
- T-025 — Merge duplicate docs
  - **Why:** `The Complete Shelling Guide.md` and `complete_shelling_guide_master_reference.md` are identical; same for the two Glyph Systems analyses.
  - **Action:** Keep one canonical filename per pair; add alias pointers; update references.
  - **Priority:** P1

- T-026 — Formalize Shelling as an algorithm
  - **Why:** Current guides are descriptive; lack a precise state machine.
  - **Action:** Define inputs/outputs, states, transitions, invariants, and termination; provide a minimal reference implementation with examples.
  - **Priority:** P1

- T-027 — Specify "Safe Cube" architecture
  - **Why:** Mentioned as stability framework but undefined.
  - **Action:** Formal spec (dimensions, constraints, invariants), mapping to E8 coordinates, and validation tests.
  - **Priority:** P1

- T-028 — Tool registry & capability matrix for advanced shelling
  - **Why:** Tool selection criteria are listed, but no concrete mapping.
  - **Action:** Build a registry (tools → capabilities → costs), plus a selection heuristic and evaluation metrics.
  - **Priority:** P2

- T-029 — Define and operationalize "n=5 level analysis"
  - **Why:** Referenced as target milestone without a measurable definition.
  - **Action:** Specify objective criteria and tests; tie to floors/rooms.
  - **Priority:** P1

- T-030 — Glyph Systems formal grammar & compiler
  - **Why:** The Glyph docs outline structure but not a concrete syntax/semantics.
  - **Action:** Draft BNF, token set, typing rules, n=-1 index semantics, interpreter/compiler plan, and E8 mapping.
  - **Priority:** P1

- T-031 — Superpermutation ↔ Glyph integration proof obligations
  - **Why:** Integration claims are high-level.
  - **Action:** Define properties to prove (closure, compositionality, invertibility, complexity bounds) and provide small constructive examples.
  - **Priority:** P2

- T-032 — AGRM/MDHG code hardening
  - **Why:** `agrm_selector.py` and `mdhg_hash.py` are minimal/naive.
  - **Action:** Add tests, replace naive frequency-only hotness with multi-signal MDHG; implement golden-angle projections and Hamiltonian bucket walk; benchmark.
  - **Priority:** P1

- T-033 — Buildings/Floors/Rooms/Elevators schema
  - **Why:** Conceptual model is clear; needs concrete data model.
  - **Action:** Define tables/graphs, IDs, constraints, and lifecycle policies; map to existing modules.
  - **Priority:** P1

### Newly added (n-levels & superpermutation framing)
- T-034 — Formal N-level semantics (Shelling & E8)
  - **Why:** You clarified N=1 as base nodes; N=1 may be a glyph up to n=5; practical limit ≈ n=4.
  - **Action:** Define allowed transformation types, distance metric, and invariants; map N to Floors; encode in metadata.
  - **Priority:** P1

- T-035 — Superpermutation-first intake policy
  - **Why:** Treat every input as a superpermutation of choice resolutions.
  - **Action:** Define the choice alphabet per datum, coverage objective (exact/approx), and fallback when n>4.
  - **Priority:** P1

- T-036 — AGRM planning with N-budgets
  - **Why:** Prevent explosion by bounding reasoning/search by N.
  - **Action:** Planner enforces N caps, allocates elevator budget per N, triggers approximate modes when near limits.
  - **Priority:** P1

- T-037 — MDHG hash features for transform distance
  - **Why:** Bucket selection should respect how far a room is from base (n-distance).
  - **Action:** Include N as feature; modulate golden-angle projections; keep separate hot maps per N-band.
  - **Priority:** P2

- T-038 — Glyph compiler guards (n≤5)
  - **Why:** Ensure glyph programs don’t exceed safe transform depth.
  - **Action:** Static checks; annotate paths with cumulative N; degrade gracefully beyond budget.
  - **Priority:** P1

- T-039 — E8 shell indexing
  - **Why:** Relate transformation depth to geometric shells.
  - **Action:** Reserve shell index for N; define permissible moves and symmetries per shell; add validation.
  - **Priority:** P2

- T-040 — Superpermutation evaluation metrics
  - **Why:** Need measurable success when using superpermutation framing.
  - **Action:** Coverage ratio, minimality approx, redundancy, and answer faithfulness vs. baseline.
  - **Priority:** P2

- T-041 — Guide updates to unify N semantics
  - **Why:** Current guides hint but aren’t explicit.
  - **Action:** Update Shelling & Glyph docs with formal N definitions, examples, and guardrails.
  - **Priority:** P2

- T-042 — Room metadata: transform path
  - **Why:** Traceability of how a room’s content was derived.
  - **Action:** Store per-room path of glyph transforms, with cumulative N and sources.
  - **Priority:** P1

- T-043 — Proof/derivation plan for n≈4 limit
  - **Why:** Ground the practical limit claim.
  - **Action:** Sketch combinatorial argument; align with superpermutation growth; define empirical tests.
  - **Priority:** P3

### Newly added (from N=0, shells, and glyph compression)
- T-044 — N=0 specification (drivers, restraints, bridge nodes)
  - **Why:** You defined N=0 as pre-determination + global constraints + loop-escape bridges.
  - **Action:** Formalize constraint headers, validation rules, and a bridge-node schema; integrate with planner safeguards.
  - **Priority:** P1

- T-045 — Recursive shelling algorithm spec
  - **Why:** Need an exact procedure for decomposition to N≤5 with early glyphing at N=5.
  - **Action:** Write step-by-step algorithm and pseudo-code; define stopping conditions and error handling.
  - **Priority:** P1

- T-046 — Glyph compression protocol (3‑word tokens)
  - **Why:** At N=5, compress to a reversible token that preserves up/down shell traversal.
  - **Action:** Specify token formation (semantic trigram + stable hash suffix), metadata (transform_path, provenance), and round‑trip rules.
  - **Priority:** P1

- T-047 — Reversibility & audit trail for glyphs
  - **Why:** Compressed tokens must expand deterministically.
  - **Action:** Store transform log, source pointers, and E8 shell coordinates; provide verify/expand functions.
  - **Priority:** P1

- T-048 — Loop-escape & traversal-loss prevention
  - **Why:** N=0→1 bridges are used to prevent cycles and dead-ends.
  - **Action:** Define detection of loops/stalls and bridge selection rules with budgets; add to AGRM planner.
  - **Priority:** P2

- T-049 — Multimodal intake mapping to E8
  - **Why:** “Truly universal” means text/code/figures/audio/etc.
  - **Action:** Outline per-modality embeddings → E8 quantization; unify into common shell indices with calibration tests.
  - **Priority:** P2

- T-050 — Transformer analogy: mapping & deltas
  - **Why:** Align conceptual model to attention mechanics across high-D spaces.
  - **Action:** Write a crosswalk (shelling ↔ attention blocks ↔ message passing), note where explicit E8 and discrete transforms differ.
  - **Priority:** P3

- T-051 — Complexity detectors for early glyphing
  - **Why:** Trigger glyph compression when growth is about to explode.
  - **Action:** Define features (branching factor, entropy, coverage slope) and thresholds; integrate with MDHG hot maps.
  - **Priority:** P1

- T-052 — Tests for 3‑word token uniqueness & collision risk
  - **Why:** Human-friendly tokens can collide.
  - **Action:** Add hash suffix policy, namespace rules, and collision detection tests; ensure reproducibility across runs.
  - **Priority:** P2

### Newly added (AGRM — hemispheres/quadrants, single‑tick, gating)
- T-053 — Spatial partitioning spec (hemispheres/quadrants)
  - **Why:** AGRM divides the map to manage search; needs principled partitioning in E8.
  - **Action:** Define partition method (spectral bisection/k‑means/kd‑tree on embeddings with E8 codes); document boundary rules and rebalancing.
  - **Priority:** P1

- T-054 — Entrypoint policy (hottest N=1 shell)
  - **Why:** Start from the most relevant grounded node.
  - **Action:** Formalize scoring (hotness × relevance × confidence × proximity); tie‑breakers and stability/hysteresis.
  - **Priority:** P1

- T-055 — Single‑tick traversal semantics
  - **Why:** “One step across all actions” needs exact scheduling.
  - **Action:** Define synchronous wavefront/agenda model, step budget, concurrency, and rollback on constraint violations.
  - **Priority:** P1

- T-056 — Branch control (beam/agenda management)
  - **Why:** Each tick spawns many valid paths.
  - **Action:** Implement beam search with Pareto scoring (coverage, tension, ΔN cost, novelty); pruning and checkpointing.
  - **Priority:** P1

- T-057 — N‑level locks for unclear regions
  - **Why:** Gate hemispheres/quadrants until clarity is sufficient.
  - **Action:** Define unlock criteria (evidence threshold, tension < θ, minimal coverage) and escalation via N=0 bridges.
  - **Priority:** P1

- T-058 — Quarantine & validation gate (pre‑output)
  - **Why:** Final answers must pass content validity & rule compliance.
  - **Action:** Build checks: provenance grounding, sheaf tension, constraint compliance, and “best‑path” justification; log trace.
  - **Priority:** P1

- T-059 — Multi‑objective cost function (AGRMs core)
  - **Why:** TSP analogy implies path optimization.
  - **Action:** Define weighted objectives: distance (semantic), ΔN, hotness utility, coverage gain, tension penalty, compute cost; support scalarization/Pareto.
  - **Priority:** P1

- T-060 — SNAP integration points
  - **Why:** AGRM “uses all tools,” SNAP is next.
  - **Action:** Specify SNAP hooks for candidate generation, pruning, and verification; define I/O contracts.
  - **Priority:** P2

- T-061 — Loop/stall detection & recovery
  - **Why:** Messy logs create reasoning loops.
  - **Action:** Detect repeating frontier states; invoke N=0 bridge nodes; enforce time/N budgets; snapshot & rewind.
  - **Priority:** P1

### Newly added (State Save / Checkpointing)
- T-062 — Checkpoint schema (state save)
  - **Why:** You want “save this as a state save” semantics.
  - **Action:** Define a YAML/JSON schema covering AGRM params, MDHG digests, E8 shell counts, glyph registry, N=0 constraints, file manifest (paths, sizes, SHA‑256), and links to the To‑Do doc.
  - **Priority:** P1

- T-063 — Save/restore procedures
  - **Why:** Need exact steps to freeze and rehydrate.
  - **Action:** Specify pre‑save quiesce rules (flush logs, close ticks), snapshot order, and restore validation (hashes, version checks, tension baseline).
  - **Priority:** P1

- T-064 — Naming & rotation policy
  - **Why:** Avoid clutter and ensure traceability.
  - **Action:** Timestamped IDs with labels; keep last K full manifests + deltas; GC policy.
  - **Priority:** P2

- T-065 — Minimal vs. full checkpoint
  - **Why:** Tradeoffs between speed and fidelity.
  - **Action:** Define two modes: logical (light) vs. reproducible (heavy with file hashes), and when to use each.
  - **Priority:** P1

- T-066 — Provenance & diff tooling
  - **Why:** Understand what changed between checkpoints.
  - **Action:** Implement diff reports (files, AGRM/MDHG params, hot regions, N-locks) with human summary.
  - **Priority:** P2

- T-067 — Security & privacy policy
  - **Why:** Checkpoints may include sensitive docs.
  - **Action:** Redaction rules, encrypted archives option, and audit list of included items.
  - **Priority:** P1

### Newly added (SNAP / SNAPDNA / SAP)
- T-068 — SNAP schema & ontology
  - **Why:** Treat every saved state as a portable unit.
  - **Action:** Define required fields (purpose, scope, N-limits, capabilities/tools, constraints_N0, provenance, E8 shells, glyph tokens, AGRM/MDHG params, inputs manifest, hashes, license).
  - **Priority:** P1

- T-069 — SNAP lifecycle & state machine
  - **Why:** Create, validate, freeze, compose, deploy, retire.
  - **Action:** Specify transitions, gates (tests, reviews), and rollback.
  - **Priority:** P1

- T-070 — SNAPDNA ingestion pipeline
  - **Why:** Build expertise from web/training material.
  - **Action:** Retrieval → digest → N-level shelling → glyphing at N=5 → validation → save; include license/PII checks.
  - **Priority:** P1

- T-071 — Expertise agent instantiation from SNAPs
  - **Why:** On-demand “perfect for the job” agents.
  - **Action:** Define ephemeral agent spec (capabilities, tool sandboxes, budgets), how many SNAPs can be loaded, and conflict resolution.
  - **Priority:** P1

- T-072 — SNAP composition/stitched DNA
  - **Why:** Hybrid operations across domains.
  - **Action:** Compatibility rules, precedence, sheaf consistency across SNAPs, E8 alignment, and merge policy.
  - **Priority:** P1

- T-073 — Versioning & immutability
  - **Why:** Reproducibility and audit.
  - **Action:** Content-addressed IDs, semver for interfaces, signed manifests.
  - **Priority:** P1

- T-074 — Evaluation & certification
  - **Why:** Quality control before “perfect agent” claims.
  - **Action:** Define test suites, acceptance thresholds, drift monitors, and recert intervals.
  - **Priority:** P1

- T-075 — Security, trust & taint
  - **Why:** Prevent data leakage and untrusted sources.
  - **Action:** Trust scores, source signatures, taint tracking, redaction.
  - **Priority:** P1

- T-076 — Registry & discovery
  - **Why:** Find and reuse SNAPs efficiently.
  - **Action:** Vector/E8 index, tags, capabilities search, dependency graph.
  - **Priority:** P2

- T-077 — AGRM/MDHG integration
  - **Why:** SNAPs should steer entrypoints and partitions.
  - **Action:** How SNAP metadata influences hot maps, beam width, N-budgets.
  - **Priority:** P1

- T-078 — SNAP staleness & refresh
  - **Why:** Expertise goes stale.
  - **Action:** Drift detection, refresh policies, and deprecation.
  - **Priority:** P2

- T-079 — SAP governance hooks
  - **Why:** Sentinel/Arbiter/Porter oversee SNAP creation/use.
  - **Action:** Define roles, approvals, and enforcement APIs.
  - **Priority:** P1

### Newly added (Domain Packs, SNAPOperations, SAP governance)
- T-080 — Domain SNAP Packs (Finance/Biology/Data Mgmt)
  - **Why:** Top-level classes tailor constraints, tools, and ontologies.
  - **Action:** Define per-domain N=0 rules, capability sets, evaluation suites, and example SNAP templates.
  - **Priority:** P1

- T-081 — SNAPOperations (meta-SNAP) spec
  - **Why:** System-wide mapping & control plane.
  - **Action:** Schema for tool/version registry, instant code-drop bundles, kernel-state checkpoints, token→E8 lattice maps, policy toggles.
  - **Priority:** P1

- T-082 — Code Drop Bundles & Safety
  - **Why:** Instant deployment can be risky.
  - **Action:** Signing, sandboxing, dry-run & canary modes, rollback, and compatibility checks before activation.
  - **Priority:** P1

- T-083 — Kernel/Runtime Snapshot Boundaries
  - **Why:** Define what “kernel-level data” includes.
  - **Action:** Formalize boundaries (env vars, configs, caches, model knobs), exclusion lists (secrets), and restore semantics.
  - **Priority:** P1

- T-084 — Token-Output Lattice Mapper
  - **Why:** Compare answers across time/SNAPs.
  - **Action:** Map outputs to E8 shells; compute drift/tension; store exemplars for regression tests.
  - **Priority:** P2

- T-085 — Custom SNAP Definition Engine
  - **Why:** Create any SNAP type on demand, audited.
  - **Action:** Templating/DSL for schemas, auto-generated tests/gates, and SAP approval flow.
  - **Priority:** P1

- T-086 — SAP Role & Escalation Spec (High/Mid/Base)
  - **Why:** You defined Sentinel/Arbiter/Porter stratification.
  - **Action:** RACI, authorities (e.g., “stop & defer all”), escalation ladder, quorum rules, and audit trails.
  - **Priority:** P1

- T-087 — Porter Zero-Trust Data Plane
  - **Why:** Porter is the only data mover.
  - **Action:** Signed payloads, allowlists, rate limits, DLP/PII scans, and deterministic routing policies.
  - **Priority:** P1

- T-088 — Arbiter Quarantine Mode
  - **Why:** System-wide freeze for incident triage.
  - **Action:** Snapshot triggers, scope of freeze, rollback points, and comms templates.
  - **Priority:** P1

- T-089 — Sentinel Policy-as-Code
  - **Why:** Enforce governance consistently.
  - **Action:** Declarative rules (YAML), test harness, and continuous compliance checks.
  - **Priority:** P2

- T-090 — Base-Level Heartbeat & Readiness Toggle
  - **Why:** Every system self-checks before ops and reports upward.
  - **Action:** Health probes, readiness gates, and status propagation to Mid/High levels.
  - **Priority:** P1

### Newly added (Think Tank, DTT, Assembly Line, Sandbox, MORSR, Wave Pool)
- T-091 — Global decay policy for SNAP sprawl
  - **Why:** You prefer decay to contain growth.
  - **Action:** Define decay/refresh rules per SNAP class; reinforce via usage; retire via staleness and tension.
  - **Priority:** P1

- T-092 — Think Tank (MoE via SNAPs) spec
  - **Why:** Central council for determinations.
  - **Action:** Member selection from registry, voting/aggregation methods, N-budget discipline, quorum/abstain rules.
  - **Priority:** P1

- T-093 — DTT (Deploy-to-Test) harness
  - **Why:** Test many variants during decision-making.
  - **Action:** Variant generator, sandbox runners, metrics (coverage, ΔN, tension, latency), promote/demote policy, feedback to Think Tank.
  - **Priority:** P1

- T-094 — Assembly Line microfunctionization
  - **Why:** Decompose decisions into atomic tests and integrate future builds.
  - **Action:** DAG of microfunctions, artifact cache, regression suites, canary integration.
  - **Priority:** P1

- T-095 — Isolated sandbox architecture (glass wall SAP)
  - **Why:** Always-allocated, independent, audited.
  - **Action:** Namespaces/tenants, read-only mirrors, sidecar Porter with allowlists, mannequin agent state-swap protocol.
  - **Priority:** P1

- T-096 — Mannequin agent pattern
  - **Why:** Lightweight state carrier visualized as a “screen”.
  - **Action:** State descriptor schema, swap/rollback semantics, heartbeat, failure domains.
  - **Priority:** P2

- T-097 — MORSR algorithm specification
  - **Why:** Middle→Ripple→SubRipple over E8 (240 neighbors).
  - **Action:** Precise neighbor scoring (like/unlike), shell-limited ripples, middle selection, stop criteria, complexity bounds.
  - **Priority:** P1

- T-098 — MORSR acceptance tests & metrics
  - **Why:** Validate behavior.
  - **Action:** Recovery from “missing middle,” coverage vs. baseline, time/space cost, stability under noise.
  - **Priority:** P2

- T-099 — Wave Pool formalization
  - **Why:** Multi-SNAP counter-waves to derive a contextual intersection.
  - **Action:** Define wave = SNAP context stream; velocities = traversal rates; meeting = alignment op (cross-corr/DTW/sheaf meet); hibernation protocol.
  - **Priority:** P1

- T-100 — Wave Bank (read-only partitions)
  - **Why:** Store dormant waves for later excitation.
  - **Action:** Storage format, versioning, renewal triggers, access controls.
  - **Priority:** P2

- T-101 — Safety gates for Think Tank/DTT/Assembly Line
  - **Why:** Prevent runaway variant storms.
  - **Action:** Budget caps, rate limits, escalation to SAP, audit sampling.
  - **Priority:** P1

- T-102 — E8 240-neighbor op in MDHG
  - **Why:** First-ring exploration.
  - **Action:** Fast nearest-neighbor over 240 root directions; cache; integrate into MDHG bucket neighbor expansion.
  - **Priority:** P2

- T-103 — State-swap semantics & failover
  - **Why:** Live backup and quick recovery.
  - **Action:** Dual-running snapshots, switchover SLOs, consistency checks, anti-split-brain rules.
  - **Priority:** P2

### Newly added (User model, Agent Center, Archivist/Repository, Projector, Doc Parser, Atomic APIs)
- T-104 — Dynamic User Profile (context-aware)
  - **Why:** As context shifts, the lattice topology changes; N1 must be re-determined.
  - **Action:** Define user profile view with privacy boundaries, retopology triggers, and evaluation rules; no PII beyond explicit.
  - **Priority:** P1

- T-105 — Agent Center (permanent agents)
  - **Why:** House system agents that serve standing roles.
  - **Action:** Roles, lifecycles, capabilities, health checks, budgets, and reporting lines to Archivist/SAP.
  - **Priority:** P1

- T-106 — Archivist (sole data receiver) spec
  - **Why:** Single bot allowed to accept data; long-term storage curator.
  - **Action:** Intake schema, content addressing, dedupe, taint/license tracking, append-only logs; SAP hooks.
  - **Priority:** P1

- T-107 — The Repository (hard I/O gated store)
  - **Why:** System backup & canonical knowledge.
  - **Action:** Read-mostly design, write via Archivist+Porter only, MVCC snapshots, encrypted archives, retention.
  - **Priority:** P1

- T-108 — Projector (ephemeral data egress)
  - **Why:** Repository sends read-only, TTL-bound data on call.
  - **Action:** One-way channels, TTL/lease semantics, signed manifests, Porter mediation.
  - **Priority:** P1

- T-109 — Document Parser (RAG-tailored to this stack)
  - **Why:** Align parsing with floors/rooms/E8/MORSR.
  - **Action:** Segmentation rules, entity/linking, elevator seeding, grounding checks, glyph triggers.
  - **Priority:** P1

- T-110 — Atomic Tool/API standard
  - **Why:** "Everything must be callable by everything else."
  - **Action:** Interface contracts, dependency injection, capability registry, discovery, and versioning.
  - **Priority:** P1

- T-111 — Full-build scope & acceptance
  - **Why:** Build must include all described components (or superior designs) while preserving intent.
  - **Action:** System acceptance criteria, invariants, and parity checklist.
  - **Priority:** P1

- T-112 — Lattice retopology policy
  - **Why:** Recompute N1 when context shifts.
  - **Action:** Triggers, costs, and stability guards; snapshot before/after.
  - **Priority:** P2

- T-113 — User privacy & ethics policy
  - **Why:** Profile without intrusive data.
  - **Action:** Allowed fields, opt-in, redaction, explainability to user.
  - **Priority:** P1

- T-114 — Agent Center ↔ Archivist data routes
  - **Why:** Ensure the Archivist remains the only data ingress; define reporting cadence.
  - **Action:** Routing rules, audit logs, and SAP enforcement.
  - **Priority:** P1

- T-115 — SNAP-first tooling pattern
  - **Why:** Make each tool a SNAP/class for universal invocation.
  - **Action:** SNAP wrappers for tools; factory/DI for agent composition.
  - **Priority:** P1

- T-116 — Single-file full SNAP backup format
  - **Why:** Compact, portable repository snapshot.
  - **Action:** Container format (TAR/CAR) with manifest, checksums, and signatures; restore procedure.
  - **Priority:** P2

- T-117 — Observability for Agent Center
  - **Why:** Fleet health and performance visibility.
  - **Action:** Dashboards for tension, coverage, latency, error rates, and decay effectiveness.
  - **Priority:** P2

- T-118 — Access policies for Repository/Projector/Porter
  - **Why:** Maintain hard I/O gates.
  - **Action:** Roles, scopes, allowlists, and incident responses.
  - **Priority:** P1

- T-119 — Superpermutation integration (pending deep dive)
  - **Why:** You’ll cover it next.
  - **Action:** Reserve space for traversal schedules, partial coverage modes, and proofs/metrics.
  - **Priority:** P1

### Newly added (Superpermutation / Haruhi integration)
- T-120 — Golden‑ratio bridge locator
  - **Why:** φ-driven heuristics can reveal naturally poor overlaps and mandatory bridges.
  - **Action:** Define φ‑scaled scoring on overlap/tension; add tensor/anti‑tensor terms; benchmark vs. known weak spots.
  - **Priority:** P1

- T-121 — Mandatory bridge points catalog
  - **Why:** There are specific spots where a bridge must occur.
  - **Action:** Detect and record canonical breakpoint patterns; expose to AGRM as hard hints.
  - **Priority:** P1

- T-122 — SNAP seeding with random seeds & prodigals
  - **Why:** Explore solution space systematically.
  - **Action:** Run seeded builds; extract prodigals/megawinners; persist as SNAPs with provenance.
  - **Priority:** P1

- T-123 — N=5 enforcement across digits
  - **Why:** Each digit above must be solved at N=5 repeatedly; all 8 outputs must feed next‑n generation.
  - **Action:** Enforce policy in planner; schedule multi‑build fanout; track usage of all eight outputs.
  - **Priority:** P1

- T-124 — Seed‑map registry of solves
  - **Why:** “Map superpermutation solving by seed.”
  - **Action:** Register (seed → outputs, length, bridges, prodigals used); enable replay/compare.
  - **Priority:** P1

- T-125 — Lattice/shell/glyph invariants (non‑optional)
  - **Why:** These structures are mandatory for consistency.
  - **Action:** Validation gates to reject artifacts lacking E8 shells, glyph compression at N=5, and proper room metadata.
  - **Priority:** P1

- T-126 — Parametric TestSuite class
  - **Why:** “Tailor any test for any data.”
  - **Action:** DSL or config‑driven test cases (coverage, bridges, φ‑scores, tension), with pluggable metrics and fixtures.
  - **Priority:** P1

- T-127 — Prodigal/extensibility scoring refresh
  - **Why:** Ensure next‑n utility is ranked correctly.
  - **Action:** Add φ‑aware terms, bridge adjacency penalties, and De Bruijn imbalance features.
  - **Priority:** P2

- T-128 — Haruhi‑mode in AGRM/MDHG
  - **Why:** Treat tasks explicitly as superpermutation traversals when flagged.
  - **Action:** Switch scoring to coverage/overlap metrics; prefer φ‑aligned moves; stricter N caps.
  - **Priority:** P2

### Newly added (AGRM Sweep Mechanics)
- T-129 — Fan-sweep algorithm spec (arms × N)
  - **Why:** You sweep x arms per N in a fan around space.
  - **Action:** Define inputs (N-budget, arms/hemisphere), step semantics, conflict resolution, and stop rules.
  - **Priority:** P1

- T-130 — Quadrant/hemisphere start-point policy
  - **Why:** Start on most packed, closest, non-shared quadrants (x4, one per hemisphere).
  - **Action:** Packedness metric (density/entropy), proximity scoring, non-share constraint; fallback when empty.
  - **Priority:** P1

- T-131 — Route-efficiency gating & thresholds
  - **Why:** Gates open based on route efficiency.
  - **Action:** Define efficiency function (distance, ΔN, coverage gain, tension, compute cost) and open/close thresholds with hysteresis.
  - **Priority:** P1

- T-132 — Iterative feedback loops (MDHG/SNAP injection)
  - **Why:** Each sweep informs hot/edge maps and SNAP seeds.
  - **Action:** Per-tick updates to hot maps, arm priorities, and seed records; learning hooks.
  - **Priority:** P1

- T-133 — Greedy fallback policy
  - **Why:** Let greedy handle easy subproblems.
  - **Action:** Detect low-entropy regions; hand off to greedy; merge results; log criteria.
  - **Priority:** P2

- T-134 — N=5 & session-bloat triggers
  - **Why:** Heavy-mode activation when any frontier hits N=5 or session context balloons.
  - **Action:** Define thresholds; actions: early glyphing, beam shrink, tighter gating, MORSR assist.
  - **Priority:** P1

### Placeholders for upcoming items
- T-135 — (placeholder)
- T-136 — (placeholder)
- T-137 — (placeholder)

