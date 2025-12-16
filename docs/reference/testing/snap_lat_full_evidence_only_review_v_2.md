# SnapLat — Full Evidence-Only Review (v2)

> Evidence-only re-review of D1/D2/D4 with expanded context and cleaner structure. No invention. Code remains incidental. This canvas supersedes the prior working canvas for organization; we will still reference it for migration.

---

## Session ground rules
- **Scope:** Only this session’s text docs and their embedded code (D1, D2, D4). D3 archive remains deferred unless explicitly unlocked.
- **Method:** Vacuum-first sweeps, orthogonal re-sweeps, negative/contrast sweeps, keyword-lock detector, coverage heatmap, cross-doc context chaining.
- **Governance:** Safe-Cube, PolicyBus, SAP (Sentinel/Arbiter/Porter) enforced at every boundary.
- **Code handling:** Code is an incidental signal; heavy-code (HC) / parameterized-process (PP) zones are tagged and scheduled for context expansion.

---

## Locked author policies (current)
- **N=0, Bridges:** N=0 = unchangeable constraints; Bridges = 0.25. Known **double-bridge** = N=0.5; **unknown** double-bridge → E‑DBSU (special distinction).
- **Partitions:** Geometric and/or semantic; refinement allowed if methodology stays true.
- **Glyph rule:** Strict 3-word compression; only when N=1 completeness is satisfied. Tokens: **ASCII-only** initially; expansion later as abstraction aids.
- **n-level invariants:** At **n=4** there is **one** perfect outcome; at **n=5** there are **exactly 8** perfect outcomes (ties to E8/Weyl/Lie properties). At n=5, bridges self-create and ambiguity enters; complexity increases.
- **Weyl completeness & inverse-variant parity:** Every idea **must be examined across all 8 Weyl points** (and recursively across interior lattices). **Inverse variants (false side)** receive the **same treatment** as positive results. The combined 8‑way evaluation can serve as a **starter Safe‑Cube** until the target space is fully defined.
- **Reason codes:** Format to be chosen empirically (toggle between snake/kebab/namespaced) and logged at boundaries.
- **Scouts:** Spawn on **geo↔sem conflict** and on **high tension spikes**; when task judged **near N≥3**, greedy flips **prep_mdhg** to pre-seed MDHG via scouts while continuing greedy work, with a **full data handoff** before full ops.
- **Think Tank/DTT/Assembly Line:** Always-on, sandboxed via Porter/Mannequin; used at every opportunity; dedicated resources separate from main arms.
- **Resource-aware scaling:** Main arms always run; when system utilization <50% (with smoothing), permissive to add scouts (details to be tuned by testing).

---



## Working protocol (v2)
1) **Intake & catalog:** Register D1, D2, D4 (titles, types, dates if present). Mark sections, code blocks, figures, checkpoints.
2) **Deep parse:**
   - Pass A (Structure): outline flow & sections
   - Pass B (Content): extract definitions/algorithms/claims/decisions with line anchors
   - Pass C (Linkage): cross-link across docs
3) **Expert lenses:** coder, mathematics, system design, critical review (evidence only)
4) **Gap analysis:** log ambiguities/missing pieces with severity & next steps
5) **Interaction graph (v2):** AGRM ↔ MDHG ↔ SNAP ↔ SAP ↔ Safe-Cube ↔ Universe ↔ Thought tools ↔ Archivist/Projector (with directionality & boundary reasons)
6) **Scouts integration:** document lifecycle, spawn triggers, rendezvous plans, `prep_mdhg` behavior, provisional-edge rules
7) **Validation plan (v2):** shell↔glyph invariants, AGRM sweeps, MDHG hot/edge map effects, Safe‑Cube gates, reproducibility
8) **Repo crosswalk (v2):** D1 work plan ↔ D4 structure (present/partial/missing)
9) **Iterative expansion/compression lifecycle:** (see below)
10) **TODO matrix (v2):** P1/P2/P3 with dependencies; migrate relevant TODOs from prior canvas
11) **Open questions:** keep tight list; resolve and lock via Decision log

---


## Intake log
| Doc ID | File | Type | Notes |
|---|---|---|---|
| D1 | full session SnapLat build.txt | Log / session build | Authoritative architecture & work order, policies, examples, manifests, validation plan
| D2 | compressed full conversation.txt | Condensed intent | Cross-check for definitions, families, ops, governance
| D4 | possible file structure.txt | Layout draft | Package/dirs for agrm/mdhg/sap/archivist/snap/universe/mannequin/plugins/cli/utils/docs/tests

---

## Interaction graph (v2)
*Edges will be annotated with boundary reasons and policy events as we lift more text.*

**Current stitched picture (from evidence across D1/D2/D4):**
- **Universe ⇄ Mannequin:** universe policies (profiles) feed **Sentinel** checks → mannequin **activate_stream** (slice gating), writes **slice SNAPs**; materializers **to_points/to_graph** feed MDHG/AGRM.
- **AGRM (controller)**: uses PolicyBus precheck, optional sweeps, records heat; emits **controller run manifest**; produces **mdhg_event::elevator** candidates.
- **MDHG:** adjacency backends (grid/faiss/hnsw), samplers (grid/lhs/poisson/random), hot/edge maps; snapshots attach as **universe overlays**; elevator candidates promoted via **SnapOpsCenter/PromotionManager**.
- **Promotion pipeline:** FastLane decisions (staged/promoted), Archivist journals, overlays updated; **promotion manifests** emitted.
- **Governance (SAP/PolicyBus):** PolicyBus as central gate; Safe‑Cube checks; Porter custody for data; Arbiter quarantine.
- **Think tools:** Think Tank/DTT/Assembly Line always-on sandbox; receive slice/results and produce plans/hypotheses back to arms.

---

## E8/Weyl operational interpretation (from user reasoning)
- **Safe-Cube = origin:** Treat Safe-Cube as the **center** of the local lattice.
- **n=4 → 1 perfect outcome:** Binary evaluation over **(context × node)** with variant/invariant axes collapses to a single globally consistent solution at n=4.
- **n=5 → 8 bridge outcomes (Weyl points):** Introducing bridges creates exactly **8** canonical outcomes that correspond to **emergent bridge classes** (operationally, “Weyl points”). These are the safe expansion directions.
- **Lattices of lattices:** Each datum can be (a) a full lattice, (b) a node in another lattice, (c) contain lattices of its parts; **Snap Ops Center** + **glyph compression** provide compact representation and traceable bridging.
- **Operational rule-of-eight:** When n=5 is reached, outcomes must be classed into **eight** canonical bridge directions; overflow routes to **E‑DBSU** or remains provisional until folded into a class.

**Implications for runtime:**
- **Scouts priority:** Prefer rendezvous along the 8 canonical directions from Safe‑Cube; use them as seed orientations for pre-seeding MDHG.
- **MDHG ingestion:** Tag scout edges as **weyl_class ∈ {1..8}** when aligned; else `weyl_class=unknown` → provisional/EDBSU.
- **Policy gating:** Add reasons like `weyl_class_assigned`, `weyl_overflow`, `weyl_unclassifiable` at boundaries.

---


*To be populated as we re-sweep: edges with boundary reasons and policy events.*

---

## Scouts integration — proposal (kept as proposal until confirmed)
- **Lifecycle FSM:** Spawned → Traveling → Rendezvous → Report → Merge/Retire (+Denied, Timeout)
- **Spawn policy:** geo↔sem conflict ≥ θ and/or high tension spike; **N≥3 proximity** triggers **prep_mdhg** (see below).
- **Rendezvous:** hemisphere hubs first; narrow to floor/room as unlocks occur
- **Greedy coupling:** greedy continues; scouts pre-seed MDHG; final data handoff precedes full ops
- **Provisional edges:** lower weight; require confirmation/decay
- **Think Tank loop:** main arm reports on pause; tools feed Think Tank; DTT/Assembly Line test; recommendations propagate to arms (neighbor hints included)
- **Weyl parity spawning:** Ensure **at least one scout per Weyl class** (true/false) is attempted at shallow depth during early passes; record `weyl_class`, `weyl_glyph`, and `variant` in scout records.

**Manifest/Checkpoint fields (draft):**
- `prep_mdhg {enabled, trigger: 'n3_proximity'|'geo_sem_conflict'|'tension_spike', proximity_score, at_tick}`
- `scouts [{id, conflict_score, tick_scale, hemisphere, rendezvous_id, ttl, spawned_at_tick, outcome, reason, weyl_class?, weyl_glyph?, variant?}]`
- `scout_edges [{edge, weight, provisional, confirmed_by, decay_ttl, weyl_class?, weyl_glyph?, variant?}]`
- `sanity_checks [{zone_id, predicate, result, evidence_hash}]`
- `think_tank [{pause_tick, panel, hypotheses, recommendations, tested, propagation}]`
- `weyl_coverage: [{class, glyph?, variant, depth, evidence_score}]`
- `variant_parity: {true_chain_score, false_chain_score, threshold}`

---



## Iterative expansion/compression lifecycle (from user clarification)
- **Prime directive:** Iteration‑first. Early iterations build **perfect space/bounds**; execution starts only after **n=4 completeness** for touched atoms and links.
- **Ephemeral universes/lattices:** Create/shell **on demand** per active scope; keep n=1 glyphs + bridges to n≥2; persist shells per datum.
- **Archivist + Snap Ops Center via Porter:** Expand only active parts, then consolidate/compress back; maintain lineage.
- **N‑level gate:** After each tick, run **n‑level check**; if <n=4, loop; if =n=4, unlock execution.
- **Think tools always‑on:** Think Tank/DTT/Assembly Line synthesize plans/modules continuously in sandbox.
- **Play‑by‑play index:** SNAP/seed log supports continuation and perfect replay.

---


*To be lifted verbatim from docs and structured into measurable checks.*

---

## Repo crosswalk (v2) — placeholder
*To be re-built against D4 with present/partial/missing marks.*

---

## TODO matrix (v2) — seed entries
- Migrate open P1 items from prior canvas (PolicyBus reason format; Safe‑Cube preflight; Slice SNAP schema; Promotion state machine; Universe policy profiles; E‑DBSU thresholds/TTL) and scouts-related TODOs (work-order schema; rendezvous catalog; provisional edges; sanity checks; autoscaler; Think Tank handshake).

### New TODOs (E8/Weyl operationalization)
| ID | Description | Priority | Status |
|---|---|---|---|
| V2‑E8‑001 | Define **variant/invariant axes** formally and how `(context × node)` truth evaluations map to **n=4 = 1 perfect outcome**. | P1 | Open |
| V2‑E8‑002 | Specify the **eight bridge classes** at n=5 and their labeling scheme (`weyl_class 1..8` or named). | P1 | Open |
| V2‑E8‑003 | Write the **classification test** that folds arbitrary n=5 outcomes into the 8 classes; log overflow to E‑DBSU. | P1 | Open |
| V2‑E8‑004 | Instrument **scout→MDHG** ingestion to carry `weyl_class`, `weyl_glyph` (strict 3‑word glyph), `provisional`, `confirmed_by`, `decay_ttl`, and **`variant`** flag. | P1 | Open |
| V2‑E8‑005 | Add **reason codes**: `weyl_class_assigned`, `weyl_glyph_assigned`, `weyl_overflow`, `weyl_unclassifiable`, `glyph_promoted`, `glyph_demoted_edbsu`, `weyl_missing_class`, `inverse_variant_missing`. | P1 | Open |
| V2‑E8‑006 | Create **adversarial tests**: cases with >8 candidates at n=5 to verify folding/overflow rules. | P2 | Open |
| V2‑E8‑007 | Design **Weyl glyph registry**: 3‑word ASCII glyphs → structured metadata (source family, target family, relation, context hash). Collision & ambiguity handling. | P1 | Open |
| V2‑E8‑008 | Define **pre‑expansion via glyph anchors**: when a glyph matches N=1 in another universe, auto‑seed candidate shells and record lineage. | P1 | Open |
| V2‑E8‑009 | Define **variant parity metrics** (true vs false evidence chains) and thresholds for promotion gates. | P1 | Open |
| V2‑E8‑010 | Add **8‑way shallow pass** requirement in early iterations (minimum depth D) with budget guardrails. | P1 | Open |

### New TODOs (N3 proximity & JIT MDHG)
| ID | Description | Priority | Status |
|---|---|---|---|
| V2‑N3‑001 | Compute **n3_proximity_score** (shelling progress + E8 position + superpermutation signal); set hysteresis & smoothing. | P1 | Open |
| V2‑N3‑002 | Specify **MDHG JIT prewarm**: which components to stage (index shards, adjacency caches, samplers), TTL for warm state, and teardown policy. | P1 | Open |
| V2‑N3‑003 | Define **Greedy safety envelope**: N≤2 unrestricted; (2,3) with watchers; ≥3 flips **prep_mdhg** and limits greedy to binary facts. | P1 | Open |
| V2‑N3‑004 | Extend manifests with fields: `n3_proximity_score`, `mdhg_jit_state {prepared|active|expired}`, `prep_mdhg_reason`. | P1 | Open |
| V2‑N3‑005 | Add reason codes: `prep_mdhg_n3`, `jit_prepared`, `jit_expired`. | P2 | Open |

### New TODOs (RAG‑TE)
| ID | Description | Priority | Status |
|---|---|---|---|
| V2‑RAG‑001 | Implement **shell‑aware chunking** (glyph/shell/universe tags) for text bodies. | P1 | Open |
| V2‑RAG‑002 | Build **hybrid retrieval** (dense+sparse) with **glyph/`weyl_class` anchors** and scout‑hint routing. | P1 | Open |
| V2‑RAG‑003 | Add **graph index** for query‑focused summarization (Graph‑RAG‑style neighborhoods). | P1 | Open |
| V2‑RAG‑004 | Add **hierarchical summaries** (RAPTOR‑style recursive trees) for long bodies. | P1 | Open |
| V2‑RAG‑005 | Add **self‑reflection** controller (Self‑RAG‑style) to decide if/when to retrieve and to critique drafts. | P1 | Open |
| V2‑RAG‑006 | Add **HyDE** synthetic passages for uncertain queries; verify against real text before export. | P2 | Open |
| V2‑RAG‑007 | Define **probability matrix** schema (retriever, reranker, graph consistency, self‑critique, scout evidence, `weyl` alignment, **contradiction mining** for inverse variants). | P1 | Open |
| V2‑RAG‑008 | Add **RAG‑TE manifest** fields: toggles, cache TTLs, component stats, and boundary reasons. | P1 | Open |

---

## Invariants — additions (E8 & bidirectional coverage)
- **8‑point bidirectional sweep (REQUIRED):** Every idea evaluation and every active lattice must process **all eight Weyl directions** including their **inverse variants** (true/false sides). A pass is **incomplete** unless all 8 are examined **for the primary lattice and each interior lattice** within scope. A **temporary Safe‑Cube** may be formed from the combined 8‑point understanding (record `temp_safe_cube=true`, with TTL) until the full space is defined.

## Validation plan — additions
### 8‑point E8 completeness
- **Idea‑level check:** Verify the ledger records **8/8 Weyl sweeps** before declaring the pass complete; else flag `weyl_sweep_incomplete` and auto‑queue.
- **Interior‑lattice check:** For each touched node’s interior lattice, verify **independent** 8‑point coverage (no inheritance without evidence).
- **Temp Safe‑Cube:** Permit a temporary Safe‑Cube only when all 8 directions are evidenced; attach TTL and lineage.

### Compression/Decompression via Snap Ops Center
- **Decompression trigger:** Porter‑mediated request expands **only** the active scope from SNAPs/glyphs; failures log `snap_decompress_failed`.
- **Consolidation:** After the tick, consolidate back into SNAPs/glyphs, preserving lineage and weyl coverage marks.

## TODO matrix — additions
| ID | Description | Priority | Status |
|---|---|---|---|
| V2‑E8‑009 | Build **E8 lattice toolkit**: create/shell/manage lattices; map glyphs↔shells↔E8; union/intersection; interior lattice handlers. | P1 | Open |
| V2‑E8‑010 | Implement **Shell Manager**: per‑datum shells with persistence format, TTL, and promotion rules. | P1 | Open |
| V2‑E8‑011 | Implement **Compression/Decompression** via Snap Ops Center: ephemeral expansion, Porter routes, consolidation back to SNAPs/glyphs. | P1 | Open |
| V2‑E8‑012 | Add **Weyl pass ledger**: per‑idea and per‑interior‑lattice 8‑direction coverage with timestamps, evidence hashes, variant/invariant flags. | P1 | Open |
| V2‑E8‑013 | Extend **reason taxonomy** with: `weyl_sweep_incomplete`, `temp_safe_cube_started/expired`, `e8_shell_incomplete`, `snap_decompress_failed`. | P2 | Open |

## Holistic coverage tracker — addendum
| System | What it is | Coverage now | Last note |
|---|---|---|---|
| **E8 lattice tools** | Creation/shelling/management + glyph mapping + interior handlers | **Proposal + TODOs** | Toolkit + Shell Manager + SnapOps decompression queued |

## Multi‑system slice plan — update
### Slice MS‑02 (complete)
**Universe / Materializers**
- **Materializers (LOCK):** `to_points(universe, criteria, wave_size, max_nodes) → coords_8d` and `to_graph(universe, criteria, adj_k, adj_buckets, adj_backend, elevators_quota) → {neighbors, elevators}`; both **stream via mannequin** so only touched regions materialize. fileciteturn13file1L6-L10
- **Policy profiles (LOCK):** YAML profiles + loader; mannequin **Sentinel** reads `family_allow/deny`, `type_allow/deny`, and `safe_cube` (requires tag criteria) and enforces budgets. fileciteturn13file1L12-L18
- **Overlays (LOCK):** `attach_mdhg_overlay(...)` + `write_universe_diff(...)` to journal overlay mutations. fileciteturn13file0L21-L26

**RAG‑TE**
- **Adapter hook (LOCK intent):** RAG doc parser adapter surface into Universe/Materialize to SNAP doc chunks with glyph/shell tags; wire to RAG‑TE (ephemeral) for probability matrix. fileciteturn13file5L80-L82

**Scouts**
- **Manifests (TO SPEC):** `{id, spawn_tick, spawn_seed, rendezvous{hemisphere, region}, ttl, outcome, reason}`, `provisional_edges[]` with `weyl_class?/glyph?`.

**E8/Weyl**
- **Class labels (DRAFT):** 8 bridge classes labeled by **3‑word glyphs**; registry schema `{glyph, source_fam, target_fam, relation, context_hash}`.
- **Weyl pass ledger (LOCK intent):** Per‑idea and per‑interior‑lattice coverage: timestamps, evidence hashes, variant/invariant flags.

---

## Validation plan — MS‑02 additions
- **Universe/materializers:** Mannequin streaming honored (no full lattice activation); deny under `safe_cube:true` without tags; allow path returns `sentinel_pass` with budgets. fileciteturn13file4L15-L19 fileciteturn13file5L47-L51
- **Overlays/diffs:** Attaching MDHG snapshots writes `universe_diff` SNAPs with `{before_overlay, after_overlay}`. fileciteturn13file0L23-L26
- **RAG‑TE:** Adapter produces SNAPs tagged with `{glyphs, shells, universe}`; retrieval decisions logged; probability matrix exported to AGRM as weights. (Adapter surface present.) fileciteturn13file5L80-L82

---

## TODO matrix — MS‑02 adds
| ID | Description | Priority | Status |
|---|---|---|---|
| V2‑UNI‑001 | Lock materializer schemas in code/docs and add Validation tests for mannequin‑streaming behavior. | P1 | Open |
| V2‑UNI‑002 | Add overlay/diff invariants to Promotion summary and Ops Center dashboards. | P1 | Open |
| V2‑RAG‑009 | Finalize RAG‑TE cache TTL defaults and probability‑matrix column list. | P1 | Open |
| V2‑SCT‑001 | Define **scout manifest** schema and integrate reason codes into PolicyBus taxonomy. | P1 | Open |
| V2‑E8‑014 | Publish initial **Weyl class label list** with glyph examples; bind to registry. | P1 | Open |
| V2‑E8‑015 | Implement **Weyl pass ledger** persistence and attach to run manifests. | P1 | Open |

---

### Slice MS‑03 (complete)
**Promotion pipeline & DNA overrides**
- **PromotionManager wiring (LOCK):** uses FastLane decisions, records via Archivist, creates result SNAPs, **overlay‑adds** to target universe. fileciteturn13file9L36-L44
- **Elevator FastLane (LOCK):** `ElevatorFastLane(threshold, compat_policy)` → `ElevatorDecision{staged, promoted, reason}`; enriched candidates include `{roomA{family,type,glyph}, roomB{…}, inverse_glyphs[], score, source_mdhg_snapshot}`. fileciteturn13file0L31-L35 fileciteturn13file0L45-L53
- **Staging worker (LOCK intent):** scans family="staging" SNAPs; re‑audits with `dna_overrides`; promotes, expires, or keeps staged; emits `staging_promote|staging_expired|staging_still`. fileciteturn13file14L27-L39
- **Per‑scenario DNA overrides (LOCK intent):** manifests support `dna_overrides` (e.g., `{min_lineage_depth, min_w5h_total}`) with precedence over global DNA policy. fileciteturn13file14L21-L26

**Mannequin v2 — SAP‑gated streaming (LOCK):**
- Porter custody for every slice wave; budgets with **125% guard**; wave‑by‑wave `activate_stream(...)` yielding SliceHandles; per‑wave **save_slice_snap()** with lineage. fileciteturn13file11L39-L44 fileciteturn13file5L31-L36

**Ops Center & CLI surfaces (LOCK):**
- `assess_slice_against_universe(slice_snap_id, universe_snap_ids)` and `label_preference(snap_id, label, weight)` for compatibility scoring + preferences. fileciteturn13file6L40-L46
- CLI verbs: `universe-apply-policy`, `snap-ops-score`, `promote-stage`/`promote finalize`; DuckDB index is **optional** (JSON remains source of truth). fileciteturn13file6L52-L61 fileciteturn13file5L5-L11

**MDHG persistence & rehydrate gap (LOCK intent):**
- Snapshot rehydrate **loses index→room membership**; fix plan: persist compact membership + `rehydrate_index_map(N)` fallback (k‑means per floor). fileciteturn13file12L27-L38

---

## Validation plan — MS‑03 additions
- **Promotion:**
  - FastLane decisions journaled; result SNAPs written; overlays updated; reasons preserved. fileciteturn13file0L31-L35 fileciteturn13file0L49-L53
  - Staging worker decisions `staging_promote|staging_expired|staging_still` emitted and TTL respected. fileciteturn13file14L27-L39
- **Mannequin streaming:**
  - Gate shows `{allow, reason, predicted_nodes, budget_nodes}`; budget guard enforced at **125%**; per‑wave slice SNAPs written. fileciteturn13file5L47-L51 fileciteturn13file5L31-L36
- **MDHG rehydrate:**
  - New snapshot schema carries membership; `rehydrate_index_map(...)` reconstructs room membership; regression tests for elevator detection post‑rehydrate. fileciteturn13file12L27-L38

---

## TODO matrix — MS‑03 adds
| ID | Description | Priority | Status |
|---|---|---|---|
| V2‑PRM‑001 | Validate FastLane journaling + decision reasons; assert result SNAP overlay updates. | P1 | Open |
| V2‑PRM‑002 | Implement **staging worker** with `dna_overrides` support; surface counts in reports. | P1 | Open |
| V2‑PRM‑003 | Add **staging TTL** and decay policy to DNA policy & manifest. | P1 | Open |
| V2‑MNQ‑001 | Enforce Porter custody + 125% budget guard in mannequin; add gate telemetry to slice SNAPs. | P1 | Open |
| V2‑MDHG‑001 | Persist compact index→room membership; add `rehydrate_index_map(N)` with k‑means fallback. | P1 | Open |
| V2‑OPS‑001 | Wire Ops Center score + preference labeling into Promotion prechecks. | P2 | Open |

---

## Opportunities & refinements (Batch C)
- **Decision reason taxonomy**: unify codes across FastLane, Promotion, Staging worker (e.g., `fastlane_threshold_fail`, `family_denied`, `staging_ttl_expired`).
- **Universe diff audit**: attach **before/after overlay hashes** to result SNAPs for tamper‑evident auditing. fileciteturn13file0L25-L26
- **Budget-aware slice‑fan prefetch**: keep a warm ring per AGRM arm with TTL (MORSR/WavePool signals feed candidates). fileciteturn13file11L6-L11
- **Ops preference feedback loop**: have Promotion outcomes feed `label_preference(...)` for future compatibility scoring. fileciteturn13file6L40-L46
- **DuckDB optional dashboards**: promotion & staging summaries via the meta index while JSON remains the source of truth. fileciteturn13file5L5-L11

---

### Slice MS‑04 (complete)
**Scouts — manifest & reasons (LOCK)**
- **Manifest schema:**
  ```
  scout: {
    id, arm_id, spawn_tick, spawn_seq, spawn_seed,
    trigger: {type: geo_sem_conflict|tension_spike|n3_proximity, n3_proximity_score?},
    rendezvous: {hemisphere, region, floor?, room?, semantic_hub?},
    ttl_ticks,
    outcome: rendezvous|merged|timeout|denied,
    reason,
    evidence: [{note, hash, at_tick}],
    provisional_edges: [{edge_id, from, to, weight, provisional:true, decay_ttl,
                         weyl_class?, weyl_glyph?, confirmed_by?}],
    metrics: {latency_ticks, travel_units, coverage_gain, disambiguation_delta}
  }
  ```
- **Reason codes:** `scout_spawned`, `scout_rendezvous`, `scout_merged`, `scout_timeout`, `scout_denied_policy`, `scout_edge_provisional`, `scout_edge_confirmed`, `scout_edge_decayed`, `scout_prep_mdhg_n3`.

**RAG‑TE — defaults & probability matrix (LOCK intent)**
- **Cache TTLs:** `index_warm_ttl=30m`, `evidence_ttl=2h`; hysteresis on/off threshold to avoid thrash; `state ∈ {cold, prepared, active, expired}`.
- **Probability matrix columns (initial):** `retriever_score, rerank_score, graph_consistency, hierarchy_consistency, self_critique_pass, hyde_delta, scout_evidence_weight, weyl_alignment, policy_gate_ready, safe_cube_risk, coverage_gain, novelty, contradiction_risk`.
- **Export:** matrix rows tagged to slices, fed to AGRM as weights; SNAP’d with lineage.

**E8/Weyl — class labels, registry & ledger (LOCK intent)**
- **Class labels:** eight classes identified by **3‑word glyphs** (primary labels) with a stable `class_id ∈ {1..8}` mapping.
- **Glyph registry (schema):** `{glyph, class_id, source_family, target_family, relation, context_hash, created_at_tick, confirmed:bool, confirmed_by:[...], collisions:[...], lineage}`.
- **Weyl pass ledger (record):** `{idea_id, lattice_id, direction_id(1..8), variant:positive|inverse, sweep_id, evidence_hash, edges_confirmed, scouts_involved, mdhg_snapshot_id, status:complete|provisional, at_tick}`.
- **Binding:** run manifests reference a `weyl_coverage_snap` (aggregated view) and per‑lattice ledgers.

---

## Validation plan — MS‑04 additions
- **Scouts:**
  - Manifest presence and required fields asserted at spawn; rendezvous/outcome transitions logged; provisional edges carry `weyl_*` labels when known.
  - Decay on timeout; merged scouts must transfer evidence and close with `scout_merged`.
- **RAG‑TE:**
  - JIT transitions respect TTL & hysteresis; export matrix exists with required columns; AGRM consumes weights deterministically.
- **E8/Weyl:**
  - Glyph registry enforces 3‑word ASCII; collisions require dual confirmation before promotion.
  - Weyl pass ledger shows **8×2 (positive+inverse)** coverage per idea and per interior lattice before “complete”.

---

## TODO matrix — MS‑04 adds
| ID | Description | Priority | Status |
|---|---|---|---|
| V2‑SCT‑002 | Implement scout manifest writer/reader; add causal ordering check in run replay. | P1 | Open |
| V2‑SCT‑003 | Add scout KPI dashboard (latency, rendezvous rate, disambiguation delta, timeouts). | P2 | Open |
| V2‑RAG‑010 | Wire TTL/hysteresis state machine; expose counters in RAG‑TE SNAPs. | P1 | Open |
| V2‑RAG‑011 | Define AGRM weight ingestion contract for probability matrix. | P1 | Open |
| V2‑E8‑016 | Build glyph registry persistence with collision handling and promotion/demotion workflow. | P1 | Open |
| V2‑E8‑017 | Implement Weyl pass ledger storage and attach to manifests; add replay tests. | P1 | Open |

---

## Opportunities & refinements (Batch D)
- **Scout budget governor:** tie `max_scouts_per_conflict` to `n3_proximity_score` bands; raise caps only when RAG‑TE matrix shows high `weyl_alignment` & low `contradiction_risk`.
- **Matrix‑aware sampling:** MDHG samplers bias selection by probability matrix weights for faster convergence on contentious regions.
- **Glyph collision signals:** apply MinHash on glyph token trigrams to pre‑flag likely collisions before registry insert; human‑readable nudge in Ops Center.
- **Deterministic decay curve:** make decay a piecewise schedule (grace → linear → exponential) controlled by evidence arrival.

---

## GS‑05 — Doc‑grounded synthesis (core components & invariants)
**Scope:** D1 “full session SnapLat build”, D2 “compressed full conversation”, D4 “possible file structure”; integrated with current canvas (v2).

### Policy & Mannequin
- PolicyBus threaded through **Mannequin → Controller → Promotion**; gates default **allow unless policy says otherwise**; Safe‑Cube requires tag criteria.
- **Mannequin v2**: SAP‑gated, Porter custody each wave, budgets with 125% guard, **slice** streaming, per‑wave **slice SNAP** with lineage.

### Universe & Materializers
- `to_points(...)` → 8‑D coords; `to_graph(...)` → neighbor graph + elevator candidates; **only touched** regions materialize via mannequin.
- Overlays/diffs: attach MDHG snapshots as universe overlays; write **universe_diff** SNAPs (before/after overlay hashes).

### MDHG
- Persistent hierarchy with decay and policy‑aware room splits; room **tags** (family/type/**3‑word glyph**); **HotMap** heat persisted; elevator candidates emitted.

### Promotion & Ops
- **Elevator FastLane** → `ElevatorDecision{staged,promoted,reason}`; PromotionManager writes result SNAPs and overlay‑adds; **staging worker** (intent) with DNA overrides.
- **Ops Center**: score `assess_slice_against_universe(...)`; `label_preference(...)`; optional DuckDB meta index (JSON is canonical).

### Shelling & Glyphs (first pass)
- `shell(meta,max_n=5)`; `compress`/`inflate`/`invert`; **3‑word base** + inverse hooks; tests verifying round‑trip and inverse marking.

### Invariants staged into Validation
- Safe‑Cube deny without tags; allow path returns sentinel pass with budgets.
- **8‑point Weyl completeness** (primary + interior lattices; positive & inverse variants) required before “complete”; temp Safe‑Cube rule recorded with TTL.
- Manifests everywhere (run/slice/promotion); overlays & diffs journaled; MDHG rehydrate gap captured with fix plan.

### Noted gaps now tracked
- E‑DBSU triggers/TTL; scout manifest/metrics; RAG‑TE TTLs + probability matrix columns; Weyl glyph registry + pass ledger.

> See chat for line‑anchored citations. This section is citation‑free by design (canvas constraint) and mirrors the evidence-backed points we reference in-line.

---

## Review protocol v2.2 — backtracking & foresight (standing rule)
- **Backtracking:** Each slice re-reads prior canvas sections + doc anchors to **re-verify earlier conclusions** and surface deltas. Any mismatch is logged to Validation & TODOs with citations.
- **Foresight:** Each slice proposes **forward linkages** strictly derived from doc text (no invention), adding orchestration hooks and cross-system bindings that the docs already imply.
- **Context expansion:** Maintain a rolling ledger of assumptions, reason codes, and invariants; attach evidence SNAP ids where available.

## GS‑06 — Backtracking confirmations & forward bindings (doc‑grounded)
**Confirmed (doc → canvas parity)**
- **Mannequin v2 gates & budgets:** SAP/Sentinel gating, Porter custody, per‑wave SNAPs, **125% guard** on predicted nodes. (See Validation plan entries.)
- **Materializers:** `to_points`/`to_graph` fed by mannequin streaming only; outputs shape matches MDHG/AGRM seeding.
- **Overlays/diffs:** `attach_mdhg_overlay(...)` and `write_universe_diff(...)` journal overlay mutations.
- **MDHG:** persistence, decay, policy‑aware splits; HotMap persistence; elevator candidate emission.
- **Promotion path:** Elevator FastLane → PromotionManager → result SNAPs + overlay add; Ops Center scoring & CLI verbs.
- **Run manifests:** present for mannequin/controller/promotion.

**Forward bindings (already implied by docs)**
- **Policy profiles loader → Sentinel:** Universe YAML profiles flow through `apply_policy(...)` and are enforced at the mannequin gate; keep Safe‑Cube tag requirement synced.
- **Ops Center feedback loop:** preferences via `label_preference(...)` guide future compatibility scoring.
- **Iterative orchestration:** FreshStart + Iterative Orchestrator provide multi‑sweep, sanity‑gated runs; bind outputs to Promotion & Ops reports.

## Validation plan — GS‑06 additions
- **FreshStart:** Verify repo/UM wipes expected families; presence of `freshstart::…` SNAP; overlays reset.
- **Iterative Orchestrator:** Require min promotions and stability epsilon; fail‑fast on `persist=False` + elevators>0 + events==0; emit final `iter_report::…` SNAP.
- **Policy profiles:** Loading + application reflected at mannequin gate (deny without tags under `safe_cube:true`); record policy hash in slice/run manifests.

## TODO matrix — GS‑06 adds
| ID | Description | Priority | Status |
|---|---|---|---|
| V2‑RUN‑001 | Add policy profile hash to all manifests (mannequin/controller/promotion). | P1 | Open |
| V2‑OPS‑002 | Wire Ops preferences (`label_preference`) into Promotion precheck scoring. | P2 | Open |
| V2‑ORC‑001 | Implement FreshStart coverage checks + `iter_report` schema and dashboards. | P1 | Open |
| V2‑MDHG‑002 | Persist compact index→room membership; add `rehydrate_index_map(N)`; regression tests. | P1 | Open |

---

## GS‑07 — Log‑aware ingestion rules & keyword expansion (doc‑grounded)
**Segment typing (LOCK):**
- **Human instruction blocks** begin with **`You said:`** → treat as *directive/intent/context*. (Examples in D1: lines with “You said:” followed by instructions.)
- **AI work/output blocks** begin with **`ChatGPT said:`** (often preceded by “Thought for …”) → treat as *actions/results/artifacts/code/canvas updates*. (Examples in D1: lines with “ChatGPT said:” and subsequent deliverables.)

**Why this matters (fit):**
- Preserves authorial intent vs. execution, enabling *policy, gating, and provenance* to reflect who initiated what. Feeds **Think Tank/DTT/Assembly Line** with the right inputs (human → questions/constraints; AI → evidence/outputs) and binds to **SNAP lineage**.

### Keyword Dictionary Expansion — Protocol v1 (LOCK intent)
**Goal:** eliminate narrow keyword bias; harvest *all* salient tokens per pass and carry them forward.

**Harvest sources (per slice):**
1) **Headers & section titles** in AI blocks (e.g., *“A/B — Raw Results (V0.30.6)”*).  
2) **Artifacts & filenames** (e.g., `snaplat_full_v0_30_6_2025_08_14.zip`, `ab_results.csv`).  
3) **Module/class/function names** appearing in AI work (e.g., `staging_worker.py`, `neg_beacons.py`, `to_points`, `to_graph`).  
4) **Policy/reason codes & flags** (e.g., `sentinel_pass`, `safe_cube_requires_tags`, `dna_overrides`).  
5) **Promoted concepts** in human directives (from `You said:` blocks).  
6) **Embedded tables/labels** that name features, metrics, universes.

**Expansion mechanics:**
- **Alias map:** unify synonyms/renames discovered in logs (e.g., *Playbook vs Direct*, *SnapLat vs AGRM CLI shim*).  
- **Morphological & n‑gram variants:** singular/plural, hyphenation, snake/kebab → canonical snake_case.  
- **Contextual co‑occurrence:** if a term appears near a known module or reason code, bind it to that facet (e.g., “FastLane decision” → `ElevatorDecision`).  
- **Delta carry‑forward:** new tokens are appended to a **rolling dictionary** and backfilled into RAG‑TE query hints and **scout rendezvous semantic hubs**.

**Continuity artifacts:**
- **`keyword_ledger.snap`**: per‑turn additions with source pointers (doc, line range, artifact).  
- **`alias_map.json`**: canonical→aliases (used by RAG‑TE and Ops search).  
- **`facet_index.json`**: term→{module/reason/policy/universe} bindings.

### Validation — ingestion parity checks (ADD)
- Every `You said:` segment must produce a **directive SNAP**; every `ChatGPT said:` segment must produce an **action SNAP** with artifacts listed when present.  
- Each pass must show **non‑decreasing** dictionary size and **≥K** new term bindings when new artifacts appear.  
- Alias collisions are flagged and queued to **E‑DBSU** if unresolved by context.

### TODOs — ingestion overhaul (ADD)
| ID | Description | Priority | Status |
|---|---|---|---|
| V2‑ING‑001 | Implement `log_segmenter` (You‑said vs ChatGPT‑said) and emit directive/action SNAPs with lineage. | P1 | Open |
| V2‑ING‑002 | Build `keyword_harvester` for headers, filenames, code symbols, reason codes; write `keyword_ledger.snap`. | P1 | Open |
| V2‑ING‑003 | Create `alias_map.json` + `facet_index.json`; integrate with RAG‑TE retrieval hints and Ops search. | P1 | Open |
| V2‑ING‑004 | Add Validation: dictionary monotonic growth; alias collision detector → E‑DBSU. | P1 | Open |
| V2‑ING‑005 | Bind directive/action SNAPs into **Weyl pass ledger** so human intents map to 8‑direction sweeps. | P2 | Open |

---

## Evidence anchors (for this rule)
- Pattern occurrences of **`You said:`** (human instructions) and **`ChatGPT said:`** (AI actions) appear throughout D1; see chat for precise line citations and examples.

