# Scope & ground rules
- **No invention / synthesis**: I will only extract, restate, and cross-reference what the documents say. Any new ideas or re-designs will wait for your explicit approval.
- **Treat logs as authoritative**: Handle the log as a step‑by‑step work journal. Preserve order, intent, and implied context.
- **Provenance first**: Every note, TODO, and gap will cite its source doc and exact location.

---

# Working protocol (per batch of docs)
1) **Intake & catalog**  
   • Register each file with an ID `[D#]` and record metadata (title, type, date if present).  
   • Quick skim to map sections, headings, code blocks, figures, and checkpoints.

2) **Deep parse (non‑sparse)**  
   • **Pass A – Structure**: Outline sections, headings, and data/logic flow.  
   • **Pass B – Content**: Extract key statements, definitions, algorithms, claims, and decisions with location anchors.  
   • **Pass C – Linkage**: Cross-link concepts across documents and prior items in this canvas.

3) **Expert lenses (evidence‑based only)**  
   • **Coder review**: API surfaces, invariants, error paths, complexity hot spots, testability, missing contracts.  
   • **Mathematics review**: Definitions → lemmas → theorems/claims, proof status, assumptions, edge cases.  
   • **System design review**: Architecture, interfaces, state and data flow, scaling, failure modes, observability.  
   • **Critical review**: Consistency checks, dependency risks, ambiguities, contradictions.

4) **Gap analysis**  
   • Identify missing pieces, unclear steps, contradictions, or dangling dependencies.  
   • Each gap logged with severity and next step.

5) **Consolidated TODO queue**  
   • Recover TODOs already present in the log first (tag **Recovered**).  
   • Then add strictly derived tasks needed to complete what’s already defined (tag **Derived**).  
   • Prioritize by impact and dependency.

6) **Per‑turn update**  
   • Understanding delta, updated TODOs, updated gap list, and crisp questions (if any).  
   • Await your approval before any synthesis or design changes.

---

# Notation & referencing
- **Docs**: `[D1]`, `[D2]`, …  
- **Locations**: sections `Sx.y`, paragraphs `¶n`, or code lines `Lm–Ln` when available.  
- **Items**: `TODO-###`, `GAP-###`, `Q-###`.  
- **Status tags**: `Open`, `In‑review`, `Approved`, `Blocked`.

---

# Review rubrics (evidence only)
**Coder review checklist**  
- Interfaces & contracts clear?  
- Input validation & error handling paths defined?  
- Complexity hotspots & data structures justified?  
- Concurrency / async boundaries identified?  
- Test surface (unit/integration/property) implied or missing?

**Mathematics review checklist**  
- Definitions precise and minimal?  
- Claim → proof or proof sketch present?  
- Hidden assumptions or domain restrictions?  
- Counterexamples / edge cases addressed?  
- Notation consistency across docs.

**System design review checklist**  
- Components, boundaries, and data flow explicit?  
- State model & idempotency?  
- Scaling strategy and back‑pressure?  
- Failure modes, retries, timeouts, observability?  
- Security & privacy considerations if applicable.

**Critical review checklist**  
- Contradictions or circular dependencies?  
- Unstated prerequisites?  
- Ambiguity hotspots requiring author clarification?  
- Dependencies on missing artifacts.

---

# Templates

# Docs parsed (current batch)

## [D1] full session SnapLat build.txt — Understanding capture (expanded)
Scope: Session log + drop logs + governance/observability scaffolding.
Key constructs & artifacts (evidence-derived):
- PolicyBus for centralized allow/deny across Mannequin/Controller/Promotion; example API and unit tests.
- Run Manifests: per-run SNAP manifests with params/inputs/policies for reproducibility/audit.
- Checkpointing: quiesce tick, flush traces, snapshot AGRM/MDHG params & partitions, enumerate artifacts, write manifest; restore re-applies constraints.
- Critical issues & Validation plan are explicitly listed (determinism/lineage; policy enforcement surfaces; hot/edge maps; testing depth).
- Work order with explicit module/file targets (shelling/glyph; sweeps; MDHG hierarchy+hotmaps; MORSR/Wave Pool; 8DoF; SAP layers).

## [D2] compressed full conversation.txt — Understanding capture (expanded)
Scope: Conceptual statements and intentions.
Key constructs (evidence-derived):
- MORSR algorithm (middle → 240 neighbors → sub-ripples → re-center → stop).
- Wave Pool concept (counter-waves; meet points; hibernate/resume).
- Safe Cube & Universes (User/Document/Work/Governance; cloneable; contextual n≈0.5–1; lattice representations).
- SNAP everywhere (families, Ops Center, taxonomy; combo plays: Entry, Sharded, Glyph-Packed, Elevator Seeder, Quarantine Gate, N‑Lock Firebreaks).
- Design intents: lattices/shells/glyphs non‑optional; fast‑path universe gated by full‑path; MDHG native-hash toggle; Archivist/Projector roles.

## [D4] possible file structure.txt — Understanding capture
Scope: Directory layout draft for an `agrm` installable package and project tree.
Highlights: `snap/taxonomy.py`, `universe/materialize.py`, `mannequin/lattice.py`, `plugins/edges/sfbb_construct.py`, `cli/agrm.py`, `utils/journal.py|metrics.py`, `docs/SAP_GOVERNANCE.md` etc.

---

# Code & process extraction — Slice 1 (text docs only)
_Focus: embedded code/code-like content in D1/D2/D4; D3 archive remains deferred._

## Code ↔ UseCase ↔ WorkHistory (initial rows)
| ID | Doc | Lines | Type | Title | Use case (from text) | Work-history anchors |
|---|---|---|---|---|---|---|
| S-CKPT | D1 | 1065–1088 | YAML schema | Checkpoint manifest (AGR/MDHG/E8 shells) | Record per-run state & params for reproducibility and restore; includes agrm beam, partitions, mdhg hash_version, hot regions digest. | Compute manifest instructions at 1041–1048; file manifest at 1021–1024. |
| S-CLI-SNAPSHOT | D1 | 9112–9120 | CLI | CLI verbs: repo snapshot/restore | Provide snapshot/restore commands to persist and restore repository state; includes conflict policy and tests. | Updated work order block at 9106–9120. |
| S-VWS | D1 | 3152–3164 | Pseudocode | Vector Warm Start + 8‑DoF navigator (sketch) | Initialize from SafeCube + hottest N=1 anchors; compose potentials; apply φ‑stagger/complexity schedule; iterate until convergence. | User question on 8‑DoF conversion at ~2806; navigator objectives at 3231–3238. |
| S-MORSR | D1 | 1475–1482 | Algorithm outline | MORSR: Middle→Ripple→SubRipple | Divide space, take middle; ripple outward; stop on no-middle; E8-aware variant elsewhere. | Linked with Thought Tools integration at 1609–1617. |
| S-MAT | D1 | 8629–8636 | API sketch | Universe materializers: to_points/to_graph | Expose to_points(name), to_graph(name) feeding MDHG/AGRM; uses mannequin slice streaming. | File structure refs at 8375–8381; to_graph signature at 8749–8755. |
| S-MANIFEST | D1 | 1041–1048 | Procedure | Compute manifest & write checkpoint files | Generate path→size→SHA‑256; write checkpoint JSON/YAML for determinism/audit. | Relates to S‑CKPT schema at 1065–1088. |

## Context chains (for clarity)
- **C‑CHKPT**: S‑MANIFEST ↔ S‑CKPT (procedure ↔ schema).
- **C‑MAT**: S‑MAT (D1) ↔ D4 `universe/materialize.py` listing (line ~42).
- **C‑VWS**: S‑VWS (D1 3152–3164) ↔ 8‑DoF question/ref (D1 ~2806) ↔ objectives (3231–3238).
- **C‑MORSR**: S‑MORSR ↔ Thought Tools integration (1609–1617).

_More rows to be added as the sweep continues (PolicyBus examples, SAP gate examples, Safe‑Cube YAML, E‑DBSU triggers)._

# Search discipline & anti-keyword-lock guardrails
- **Vacuum-first extraction:** Every sweep runs *without* relying on prior notes—mask key terms, parse by structure (types, roles, IO, invariants) and only then reconcile with global context.
- **Orthogonal re-sweep:** Re-run with synonym/antonym/near-neighbor terminology and neutral placeholders to catch content missed by keyword anchoring.
- **Negative/contrast sweep:** Search for contradictions, exceptions, and failure-language ("unless", "except", "counter", "degenerate") to surface edge behaviors.
- **Keyword-lock detector:** Track term concentration per extract; if any term >20% of tokens or head-5 terms >60%, trigger an expansion pass.
- **Coverage heatmap:** Maintain per-subsystem/construct coverage to avoid over-sampling a narrow area.
- **Context chaining discipline:** When blocks reference each other across distance, stitch them under a Chain‑ID, preserving all anchors, then integrate.

## Additional decision (record)
- **2025-08-14 — Non-focal, vacuum‑then‑integrate discipline:** Each search acts in isolation, then is applied to total understanding; avoid keyword lock‑in; ensure cross‑system applicability in every pass.


---

# Code & process extraction — Slice 2 (text docs only)
_Approach: vacuum-first sweep with masked keywords, then orthogonal re-sweeps; context chained post hoc. D3 (archive) still deferred._

## Code ↔ UseCase ↔ WorkHistory (new rows)
| ID | Doc | Lines | Type | Use | FirstLine |
|---|---|---|---|---|---|
| S-255fca84 | D1 | 100-105 | YAML-like | General | Schema: Build a small typed hypergraph: Entities (E), Relations (R), Attributes  |
| S-972d2d6d | D1 | 272-283 | YAML-like | MDHG hierarchy/config | Faithfulness: citation-level grounding rate per answer; contradiction/tension fl |
| S-25658933 | D1 | 372-373 | CLI-like | AGRM traversal/config | …/agrm_selector.py — AGRM selection heuristic importing superbuilder.overlap, |
| S-94fbeb08 | D1 | 537-538 | CLI-like | AGRM traversal/config | agrm/agrm_selector.py |
| S-97777c86 | D1 | 615-620 | YAML-like | General | Transforms: … closed set (e.g., substitute, permute, compose, abstract, spe |
| S-2c252c52 | D1 | 733-738 | YAML-like | General | E8: consistent coordinate/codebook across layers; shells can map to N-bands. |
| S-e2dbe351 | D1 | 820-825 | YAML-like | MDHG hierarchy/config | …high-N conclusions trace to lower-N rooms; low sheaf tension. |
| S-4ffbb623 | D1 | 1065-1098 | YAML-like | Checkpoint/Manifest | checkpoint_id: ckpt-2025-08-12T17-43-00Z |
| S-9a91408e | D1 | 1186-1187 | JSON-like | General | "n_limits": {"max_n": 4, "glyph_on_n5": true}, |
| S-b4ae7ced | D1 | 1190-1194 | JSON-like | Checkpoint/Manifest | "agrm_defaults": {"beam": 8, "cost_weights": {"coverage":1.2,"delta_n":1.0,"tens |
| S-be6ec1c8 | D1 | 1207-1212 | YAML-like | Policy/Guard | Sentinel: monitors runtime (policy, leaks, tension spikes). |
| S-ece12ecf | D1 | 1442-1446 | YAML-like | Policy/Guard | Works: modular, auditable, scalable; you can spin “perfect-for-task” agents safe |
| S-ac500c76 | D2 | 86-91 | YAML-like | SNAP schema/op | SNAPKernel … all kernel level data as save state snaps for recovery in data |
| S-2debd16a | D2 | 294-295 | CLI-like | MDHG hierarchy/config | … everything it does now, but via toggle for normal inline/nati |
| S-847e2a5c | D4 | 1-1 | CLI-like | AGRM traversal/config | agrm/ |

## Rationale Map — nearby explanations (auto‑extracted)
- **S-97777c86**
  - Therefore, all reasoning/outputs are also superpermutations of internal choice sets—constrained by N-budgets.
- **S-2c252c52**
  - A concrete, faithful procedure (so we can implement it later)
A) Intake & decomposition

Normalize datum; capture modality and provenance.
- **S-e2dbe351**
  - Measurable outcomes (so we can prove it works)

Budget adherence: zero occurrences of n_level > N_budget in answers.
- **S-be6ec1c8**
  - I’ve added concrete to-dos for SNAP/SNAPDNA/SAP (schema, lif… the running log so we can formalize each piece when you’re ready.

### Context chains added
- **C‑CHKPT**: S‑MANIFEST ↔ S‑CKPT (procedure ↔ schema)
- **C‑MAT**: S‑MAT (D1) ↔ D4 `universe/materialize.py`
- **C‑VWS**: S‑VWS (D1) ↔ 8‑DoF question ↔ objectives
- **C‑MORSR**: S‑MORSR ↔ Thought Tools integration

---

# Code & process extraction — Slice 3 (text docs only)
Focus: policy profiles/loader, slice SNAPs, promotion pipeline, Safe‑Cube gate, and materializer usage.

## Code ↔ UseCase ↔ WorkHistory (new rows)
| ID | Doc | Lines | Type | Title/Use |
|---|---|---|---|---|
| S-POLICY-PROFILES | D1 | 8817–8837, 8913–8919 | Paths & API | `configs/profiles/*_universe.yaml`; loader mentioned; mannequin honors via Sentinel (Safe‑Cube tag criteria). |
| S-SLICE-SNAP | D1 | 8918–8923 | Pseudo‑API | Mannequin persists `slice` SNAPs per wave (lineage = universe + criteria). |
| S-PROMOTION | D1 | 8867–8870, 8891–8897, 8906–8923, 9082–9085 | API sketch | `PromotionManager.stage_result` and `.promote_after_full`; Archivist logs; service stub exposes `/promote`. |
| S-TOGRAPH-EX | D1 | 8790–8795 | Code/CLI | Example `to_graph(...)` call with criteria `{family,type,tags}`, `adj_k`, `adj_backend`, `wave_size`, `elevators_quota`. (**HC‑zone**) |
| S-UNIVERSE-OPS | D1 | 8372–8378, 8626–8632, 8746–8751 | Ops list | Universe ops and materializers; copy‑on‑write overlays; mannequin streaming. (**PP‑zone**) |

### HC/PP tags (queued for next‑turn context expansion)
- **HC‑zone:** S‑TOGRAPH-EX (dense parameters)
- **PP‑zone:** S‑UNIVERSE-OPS (parameterized ops sequence)

### Context chains added
- **C‑POLICY→SAFE‑CUBE:** policy profiles/loader ↔ mannequin Safe‑Cube gate
- **C‑SLICE‑SNAP:** mannequin lattice ↔ Ops Center lineage
- **C‑PROMOTION:** PromotionManager ↔ Archivist `post` ↔ universe overlays
- **C‑MAT‑FLOW:** to_graph → MDHG/AGRM inputs → checkpoint manifests

---

# Interaction matrix (Round 3 — evidence only)
| From → To | Interaction (what) |
|---|---|
| Universe policy loader → Mannequin (Sentinel) | Apply `safe_cube`/tag criteria; deny if missing; reasons recorded. |
| Mannequin (slice stream) → SNAP Ops Center | Persist `slice` SNAPs with lineage; lightweight headers. |
| Materialize.to_graph → MDHG/AGRM | Provides graph/elevator seeds (adj_k, backend, wave_size, elevators_quota). |
| PromotionManager → Archivist/Universe | Stage result to FastLane; after full‑path, promote and overlay into target universe; log events. |
| Controller → Universe | Accept `universe_name`, materialize first, then run controller. |

---

# New TODOs (derived from Slice 3)
| ID | Description | Priority | Status | Depends |
|---|---|---|---|---|
| TODO-036 | Spec **Universe policy profile** schema (YAML) + loader semantics; enumerate built‑ins (User/Doc/Work/Governance). | P1 | Open | TODO-017 |
| TODO-037 | Define **Mannequin deny reasons** enum + logging format (e.g., `safe_cube_requires_tags`). | P1 | Open | — |
| TODO-038 | Define **Slice SNAP** schema (id/meta/lineage/TTL); retention policy. | P1 | Open | TODO-004 |
| TODO-039 | Specify **PromotionManager** state machine & thresholds; Archivist event schema; overlay semantics. | P1 | Open | TODO-021 |
| TODO-040 | Document `to_graph`/`to_points` signatures & param ranges; backend selection rules & fallbacks. | P2 | Open | — |
| TODO-041 | Author **Universe ops** API (clone/fork/compose/diff/materialize/promote/archive) + tests. | P2 | Open | TODO-021 |
| TODO-042 | Mini **Universe Query Language** (JSON/YAML) spec and parser hooks. | P2 | Open | TODO-041 |
| TODO-043 | Package **SAP rule pack** for universes; Safe‑Cube preflight before promotion. | P2 | Open | TODO-005 |
| TODO-044 | Wire **controller_v0_7** to accept `universe_name`; materialize then run path. | P2 | Open | TODO-012 |

---

# Code & process extraction — Slice 4 (text docs only)
Focus: PolicyBus gates, SAP (Sentinel/Arbiter/Porter) surfaces, Safe‑Cube allow/deny, E‑DBSU triggers; keep code incidental.

## Code ↔ UseCase ↔ WorkHistory (new rows)
| ID | Doc | Lines | Type | Title/Use |
|---|---|---|---|---|
| S-POLICYBUS-CALLS | D1 | 9263–9266, 9313–9315 | Notes | Centralize allow/deny checks in **PolicyBus**; used by controller, mannequin, promotion; reduces policy escapes. |
| S-SAFE-CUBE-POL | D1 | 4071–4076, 4892–4896 | Paths/Status | Governance & Safety components; current status: policies allow/deny universes; Safe‑Cube preflight missing at that point. |
| S-SAFE-CUBE-DENY | D1 | 8913–8919, 8985–8990 | Change log | Mannequin Safe‑Cube gate: **deny** slice when `safe_cube=true` and missing tag criteria; family/type allow/deny. (PP‑zone) |
| S-SENTINEL-ARB-PORT | D1 | 1120–1144, 1355–1364 | Roles | SAP roles as runtime guards: **Sentinel** monitor, **Arbiter** adjudicate/stop, **Porter** sole data mover. |
| S-PROMOTION‑SURFACES | D1 | 8867–8870, 8891–8897, 8906–8923, 9082–9085 | API/Service | Promotion surfaces in CLI/service using **PromotionManager**; `/promote` endpoint when FastAPI available. |
| S-EDBSU | D1 | 17408–17411, 17457–17460, 17595–17598 | Concept | **E‑DBSU** quarantined universe: triggers (novel + double‑bridge), evidence graph; promotion/TTL outcomes. |

### HC/PP tags (queued for next‑turn context expansion)
- **PP‑zone:** S‑SAFE‑CUBE‑DENY (parameterized policy process)

### Context chains added
- **C‑POLICYBUS→BOUNDARIES:** PolicyBus listed at controller/mannequin/promotion borders
- **C‑SAFE‑CUBE:** policies.yaml → mannequin gate → deny logs
- **C‑SAP‑ROLES:** Sentinel/Arbiter/Porter definitions ↔ governance hooks
- **C‑PROMO‑FLOW:** stage_result → full‑path check → promote/overlay
- **C‑E‑DBSU:** intake triggers → evidence graph → promotion/retire

---

# Interaction matrix (Round 4 — evidence only)
| From → To | Interaction (what) |
|---|---|
| Controller/Mannequin/Promotion → **PolicyBus** | Single place to call `allow/deny`; capture reasons; prevent escapes. |
| Universe policies → **Sentinel** | Safe‑Cube/tag criteria evaluated; failures cause `deny` (reason captured). |
| **Sentinel/Arbiter** → Controller | On breach: **Arbiter** can quarantine/stop; **Sentinel** emits policy events; **Porter** moves only allowed payloads. |
| **PromotionManager** → Archivist/Universe | Stage result SNAPs; after full‑path, overlay into target universe; log events. |
| **E‑DBSU** → Controller/Policies | Staged ideas enter via intake triggers; TTL determines retire/promote; constraints/hints updated on success. |

---

# New TODOs (derived from Slice 4)
| ID | Description | Priority | Status | Depends |
|---|---|---|---|---|
| TODO-045 | Define **PolicyBus reason format** (boundary: slice/seed/adj/sweep/stage/promote); include machine codes + human text. | P1 | Open | — |
| TODO-046 | Add **Safe‑Cube preflight** to controller path (pre materialize/run); record deny reasons & counts. | P1 | Open | TODO-043 |
| TODO-047 | Specify **deny/allow** event schema (who/where/why), retention, and audit linkage to run manifests. | P1 | Open | TODO-045 |
| TODO-048 | Finalize **E‑DBSU**: intake thresholds, TTL defaults, promotion outcomes (Constraint/Hint/Task/Discard). | P1 | Open | — |
| TODO-049 | Document **Arbiter quarantine** and **Porter routes** (source/target, payload class, checksum). | P2 | Open | — |
| TODO-050 | Add **PolicyBus** test matrix (unit/property/adversarial) across all boundaries. | P2 | Open | TODO-045 |


---

# Code & process extraction — Slice 5 (text docs only)
Focus: expand PP‑zone around Safe‑Cube denials, gather PolicyBus reason usage at boundaries, lift E‑DBSU trigger/TTL details. Code remains incidental.

## Code ↔ UseCase ↔ WorkHistory (new rows)
| ID | Doc | Lines | Type | Title/Use |
|---|---|---|---|---|
| S-SAFE-CUBE-CONTEXT | D1 | 8898–8923 (+/‑) | Rationale paragraphs | Surrounding explanation for Safe‑Cube gate: why tags are mandatory; what risks are mitigated; where denials are logged. (PP‑zone expansion) |
| S-POLICYBUS-REASONS | D1 | 9263–9315 (+/‑) | Notes/examples | Mentions of reason strings at boundaries (slice/seed/adj/sweep/stage/promote) and how they should appear in run manifests/audit logs. |
| S-EDBSU-DETAILS | D1 | 17400–17610 (+/‑) | Concept → ops | E‑DBSU intake conditions (novelty + double‑bridge), TTL defaults, promotion outcomes (Constraint/Hint/Task), retire flow. |

### Context chains added
- **C‑SAFE‑CUBE‑EXP:** S‑SAFE‑CUBE‑DENY ↔ S‑SAFE‑CUBE‑CONTEXT ↔ policy profiles
- **C‑REASONS:** S‑POLICYBUS‑CALLS ↔ S‑POLICYBUS‑REASONS ↔ run manifests
- **C‑E‑DBSU‑FLOW:** S‑E‑DBSU ↔ S‑EDBSU‑DETAILS ↔ PromotionManager

---

# Interaction matrix (Round 5 — evidence only)
| From → To | Interaction (what) |
|---|---|
| PolicyBus → Run Manifest / Audit | Boundary checks emit **reason codes + human text**; stored with tick/promotion context.
| Safe‑Cube gate → PolicyBus | Denials include **deny_reason**; fed into PolicyBus telemetry; surfaced in manifests.
| E‑DBSU → PromotionManager | When evidence crosses thresholds (incl. double‑bridge), stage candidate; TTL expiry retires entry; promotions update constraints/hints/tasks.

---

# New TODOs (derived from Slice 5)
| ID | Description | Priority | Status | Depends |
|---|---|---|---|---|
| TODO-051 | Enumerate **deny_reason** strings observed in text (canonical set) and map to boundaries. | P1 | Open | TODO-045 |
| TODO-052 | Add **run‑manifest fields** for per‑boundary reason logging (key names + types + examples). | P1 | Open | TODO-045 |
| TODO-053 | Record **E‑DBSU** TTL default(s) and promotion scoring formula exactly as written; add to spec. | P1 | Open | TODO-048 |
| TODO-054 | Cross‑link **PolicyBus reason codes** ↔ **SAP events** ↔ **Archivist records** (one row per interaction). | P2 | Open | TODO-047 |

---

# Open questions (pending)
| Q‑ID | Source | Question |
|---|---|---|
| Q‑004 | E‑DBSU | Should **double‑bridge** be treated as strictly additive (Bridge=0.25 → double=0.5) or as a special trigger with its own weight/threshold? |
| Q‑005 | Policies/Manifests | What canonical format do you want for **deny/allow reason codes** (e.g., `snake_case`, `kebab-case`, prefixed namespaces)? Any reserved codes besides `safe_cube_requires_tags`? |
| Q‑006 | Glyph registry | For strict 3‑word glyphs, should tokens be limited to ASCII lowercase, or may we allow Unicode and/or punctuation? |
| Q‑007 | Partition precedence | When geometric and semantic partitions **conflict**, which takes precedence at traversal time (or should PolicyBus decide per profile)? |


## Decision log — addendum
| When | Source | Decision / Rationale |
|---|---|---|
| 2025-08-14 | Q-004 Answer | **Double bridges**: known→treat as **N=0.5** nodes; **unknown** double bridges live in **E‑DBSU** with special distinction (emergent abstraction). |
| 2025-08-14 | Q-005 Answer | **Reason code format**: begin with a **toggle** (snake_case / kebab-case / namespaced) and evaluate empirically. |
| 2025-08-14 | Q-006 Answer | **Glyph tokens**: start **ASCII-only**; expand to broader symbol/glyph types later as abstraction aids. |

## Open questions — addendum (pending)
| Q‑ID | Source | Question |
|---|---|---|
| Q‑007 | Partitions | When geometric and semantic partitions disagree at traversal time, which should take precedence—or should PolicyBus choose via profile? |

## TODO additions — policy answers
| ID | Tag | Source | Description | Priority | Status |
|---|---|---|---|---|---|
| TODO-055 | Derived (policy) | Q‑004 | Handle **double bridges**: known→`n=0.5`; unknown→route to **E‑DBSU** with tag `double_bridge_unknown`; reflect in manifests/PromotionManager. | P1 | Open |
| TODO-056 | Derived (policy) | Q‑005 | Implement **reason format toggle** (snake/kebab/namespaced); add experiment logging for comparison. | P1 | Open |
| TODO-057 | Derived (policy) | Q‑006 | Enforce **ASCII-only** glyph tokens initially; design expansion protocol + audit checks; update Glyph Gate. | P1 | Open |
| TODO-058 | Derived (policy) | Q‑004/Q‑048 | Add manifest fields: `bridge_evidence.count`, `bridge_evidence.class`, `edbsu.flag`. | P1 | Open |


---

# Proposal under review — Differential‑speed, counter‑directional observer arms ("scouts")
**Source:** User proposal (2025‑08‑14). **Status:** Evaluating (no lock‑in). Code remains incidental.

## Summary of the idea (my words)
- Keep the **main AGRM arm** advancing normally.
- When it hits an **intersection/zone of conflict** (e.g., geometric vs semantic guidance disagree), spawn one or more **observer arms** ("scouts").
- Scouts run at a **different tick rate** (faster or slower) and in **counter directions** to probe the neighborhood.
- The **main arm pauses** at the zone until a scout meets it ("rendezvous") and returns **connection/relationship data** to resolve the conflict.

## Fit with existing constructs (compatibility)
- **Wave Pool:** Scouts are a concrete realization of counter‑travelling waves; the **meet point** is a natural place to merge contexts.
- **MORSR:** Scouts resemble shallow ripples out of a local middle; stop on coverage stall.
- **N‑locks & Quarantine:** The pause can be expressed as a **temporary N‑lock**; outputs still pass quarantine.
- **MDHG/Elevators:** Successful rendezvous may justify an **elevator** or a **bridge**; unknown **double‑bridges** route to **E‑DBSU** per policy.
- **PolicyBus/SAP:** Spawning and pausing are policy‑gated; **Porter** must carry any data; **Sentinel/Arbiter** monitor and can deny or quarantine.

## Benefits (why this could work)
- **Faster disambiguation:** Early probe avoids committing the main arm to a costly wrong branch.
- **Boundary sensing:** Counter‑directional scouts find **true boundary geometry** between quadrants and semantic clusters.
- **Data fusion:** Meet‑point allows explicit **potential merging** (geometry + semantics) with an audit trail.

## Risks / failure modes (brutally honest)
- **Stall/deadlock:** If scouts never meet, the main arm could pause indefinitely → requires TTL/backoff.
- **Oscillation/looping:** Conflicting signals can cause bounce at the boundary → need hysteresis and once‑per‑zone rendezvous rules.
- **Budget blow‑ups:** Too many scouts increase compute/IO → hard quotas and admission control.
- **Coverage bias:** Faster scouts may skew hot/edge statistics → compensate metrics and log scout‑adjusted coverage.
- **Determinism:** Async rendezvous makes runs harder to reproduce → sequence/causal ordering must be written to the **run manifest** and checkpoints.
- **Governance:** Scouts must not bypass **Safe‑Cube** or policy denials; every spawn meets PolicyBus/SAP checks with **reason codes**.

## Suggested guardrails (non‑binding; for review only)
- **Spawn trigger:** On **conflict score ≥ θ** between geometric potential vs semantic potential (MDHG/Universe), or when a **rendezvous tag** is present.
- **Quotas:** `max_scouts_per_conflict ≤ 2`; `max_concurrent_scouts` global cap; per‑run budget tracked in manifest.
- **Tick scaling:** `scout_tick_scale ∈ {0.5, 1, 2}`; choose via profile (toggle, then test).
- **Pause window:** Main arm sets **rendezvous N‑lock** with TTL `Δticks`; on timeout → proceed with fallback and log `rendezvous_timeout`.
- **Meeting rule:** On contact, compute **meet‑potential = f(geo, sem, evidence)`; if **unknown double‑bridge** detected → file in **E‑DBSU**.
- **Logging:** Add reason codes: `scout_spawned`, `scout_denied_safe_cube`, `rendezvous_met`, `rendezvous_timeout`, `scout_conflict_resolved`, `scout_conflict_unresolved`.
- **Reproducibility:** Manifests include **scout IDs**, spawn conditions, tick scales, TTLs, rendezvous outcomes, and evidence hashes.

## Minimal spec hooks (where it would live)
- **AGRM controller:** spawn/await rendezvous; pause window; fallback path.
- **PolicyBus/SAP:** preflight for spawn + data movement; deny/allow reasons recorded.
- **MDHG:** mark rendezvous rooms; allow elevator suggestion on success.
- **Checkpoints/Manifests:** new fields (see TODOs below) for scouts and rendezvous logging.

## TODOs (added)
| ID | Description | Priority | Status |
|---|---|---|---|
| TODO-059 | Define **conflict score** and **spawn trigger** for scouts (inputs: geo vs sem potentials, tension). | P1 | Open |
| TODO-060 | Specify **rendezvous N‑lock** semantics + TTL/backoff + hysteresis. | P1 | Open |
| TODO-061 | Add **scout quotas/tick‑scale** fields to profiles; log in manifests/checkpoints. | P1 | Open |
| TODO-062 | Define **meet‑potential merge** function and when to create elevators vs bridges vs send to **E‑DBSU**. | P1 | Open |
| TODO-063 | Extend **reason code** set for scout lifecycle (`scout_*`, `rendezvous_*`). | P1 | Open |
| TODO-064 | Simulation harness: boundary scenarios (stall/oscillation/budget) to validate guardrails before adoption. | P2 | Open |

## Open questions for you
1) **Spawn policy:** Should scouts spawn **only** on geometric↔semantic conflict, or also on high‑tension edges (e.g., uncertainty spikes without explicit disagreement)?
2) **Pause strictness:** Is the main arm **hard‑paused** during rendezvous TTL, or should it take **micro‑steps** with reduced beam (speculative) to avoid stalls?
3) **Meet merge:** Do you prefer a **conservative merge** (min potential) or **evidence‑weighted blend** (e.g., weight by lineage/grounding) for the meet‑potential?
4) **Quotas:** Defaults acceptable as above (`≤ 2` per conflict), or would you like a higher/lower ceiling for early experiments?


## Proposal under review — Augmented (scouts as space seeders & sanity checkers)
**Scope:** User refinement (2025‑08‑14). Scouts drive initial space definition, greedy‑vector discovery, and later “true‑to‑task” verification sweeps. Rendezvous are hemisphere‑assigned and narrow as hierarchical unlocks progress. Treated as tick‑based aggregates with periodic MORSR re‑layout.

### Fit with stack (integration points)
- **Init pass:** Scouts seed **MDHG** edges/elevators and populate early hot/edge maps before main sweeps.
- **Verification pass:** Subsequent ticks spawn scouts as **sanity checkers** (is‑this‑true‑to‑task) at conflict zones and checkpoints.
- **Rendezvous layout:** Hemisphere‑specific rendezvous points guaranteed to cover all hemispheres; progressively refined to floors/rooms as unlocks happen.
- **Work‑order assignment:** Scouts are instantiated from a **work order** that encodes expertise strata and task/context/understanding vectors.
- **Wave/MORSR synergy:** Periodic **tick aggregations** hand the current lattice to **MORSR** to propose re‑layouts.

### User directives — locked in
1) **Spawn conditions:** Scouts spawn on **both** (a) explicit **geo↔sem conflicts** and (b) **high‑tension spikes**. If a task is determined **hard beyond N>4**, **greedy** triggers a **`prep_mdhg` toggle** so MDHG begins **preseeding via scouts** for future iterative runs while greedy continues what it can do now; a **full data handout** occurs before full system operations.
2) **Main‑arm pause behavior:** During a rendezvous pause, the **main arm informs Think Tank** with findings to shape next‑step planning at each pause throughout the attempt.
3) **Think Tank as reasoning hub:** **All tools** inform Think Tank. Think Tank assembles the best expert panel, critiques options, suggests best paths; **DTT/Assembly Line** test when time permits; findings are returned to **all relevant arms**, including neighbor hints (e.g., “node x closer to y than z”).
4) **Resource‑aware scaling:** Start small and expand scout counts as work/resources allow. **Main arms always run**; if system resource **utilization < 50%**, add scouts as best determined. 
5) **Wording correction:** “expired” was a typo; intended phrasing: **“created to exact need.”**

### Critical review (honest risks & mitigations)
- **Risk: stall/deadlock** if a rendezvous never happens → *Mitigate:* rendezvous TTL + fallback, once‑per‑zone rendezvous, hysteresis.
- **Risk: budget blow‑up** from many scouts on complex tasks → *Mitigate:* complexity‑aware caps + global `max_concurrent_scouts`, admission control.
- **Risk: biasing MDHG** with premature edges → *Mitigate:* mark scout‑discovered edges as **provisional** (lower weight) until confirmed; decay unconfirmed edges.
- **Risk: coverage skew** (fast scouts dominate stats) → *Mitigate:* keep separate **scout‑adjusted coverage** channel; normalize in metrics.
- **Risk: governance bypass** → *Mitigate:* PolicyBus/SAP preflight for every spawn; **Safe‑Cube** gate; log deny/allow reasons.
- **Risk: determinism** with async rendezvous → *Mitigate:* sequence numbers, scout IDs, causal ordering in manifests/checkpoints.

### Suggested (non‑binding) mechanics
- **Scout lifecycle FSM:** `Spawned → Traveling → Rendezvous → Report → Merge/Retire` (+ `Denied`, `Timeout`).
- **Spawn policy:** trigger on **geo↔sem conflict ≥ θ** *and/or* high **tension spikes** (toggle), bounded by complexity‑based quota.
- **Quota heuristic:** initial `max_scouts_per_conflict ≤ 2`; global cap; per‑run budget in manifest.
- **Tick scaling:** profile flag `scout_tick_scale ∈ {0.5, 1, 2}`; choose in experiments.
- **Greedy vectors:** scouts emit **candidate vectors** and **bridge/elevator proposals**; unknown double‑bridges → **E‑DBSU** tagging.
- **Rendezvous narrowing:** start at hemisphere hubs; on unlocks, shift rendezvous to floor/room granularity.
- **Sanity checks:** “true‑to‑task” predicate uses active **work‑order** filters + lineage strength; results logged as confirmations or discrepancies.

### Run‑manifest / checkpoint additions (fields)
- `prep_mdhg: {enabled, trigger: {reason: 'n>4_hard'|'geo_sem_conflict'|'tension_spike', at_tick}, data_handoff_at_tick}`
- `scouts: [{id, conflict_score, tick_scale, hemisphere, rendezvous_id, ttl, spawned_at_tick, outcome, reason}]`
- `scout_edges: [{edge, weight, provisional, confirmed_by, decay_ttl}]`
- `sanity_checks: [{zone_id, predicate, result, evidence_hash}]`
- `work_order: {groups: [...], expertise: [...], context_vectors: [...], understanding: [...]}`
- `think_tank: [{pause_tick, panel, hypotheses, recommendations, tested: {dtt:bool, assembly_line:bool}, propagation:{arms:[...]}}]`

### New TODOs
| ID | Description | Priority | Status |
|---|---|---|---|
| TODO-065 | Define **work‑order schema** (groups/skills/context/understanding) and how scouts are instantiated from it. | P1 | Open |
| TODO-066 | Specify **hemisphere rendezvous** catalog & refinement rule as unlocks occur. | P1 | Open |
| TODO-067 | Mark **scout‑discovered edges** as provisional with decay & confirmation rules; feed weights to MDHG safely. | P1 | Open |
| TODO-068 | Define **sanity‑check predicate** (is‑true‑to‑task) and evidence thresholds; log format. | P1 | Open |
| TODO-069 | Add **scout metrics** (disambiguation latency, rendezvous rate, timeout rate, coverage delta, false‑confirm rate). | P2 | Open |
| TODO-070 | Schedule **tick‑based aggregation cadence** and MORSR re‑layout handoff criteria. | P2 | Open |
| TODO-071 | Specify **`prep_mdhg`** semantics and detection of **N>4 hard** tasks; define greedy→MDHG preseeding protocol and toggle lifecycle. | P1 | Open |
| TODO-072 | Design **resource autoscaler**: define utilization metric (CPU/GPU/Mem/IO), smoothing window, and spawn policy when utilization < 50%. | P1 | Open |
| TODO-073 | Formalize **Think Tank handshake**: payload schema from main arm pause; expert panel selection; result propagation back to arms (incl. neighbor hints). | P1 | Open |
| TODO-074 | Couple **DTT/Assembly Line** with scout outputs; define injection points and back‑pressure rules. | P1 | Open |
| TODO-075 | Extend **reason codes**: `mdhg_prep_enabled/skipped`, `greedy_preseed_complete`, `think_tank_consulted`, `think_tank_plan_ready`, `assemblyline_test_passed/failed`, `dtt_experiment_queued/completed`. | P1 | Open |

