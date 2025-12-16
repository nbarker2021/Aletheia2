# Beam→E8→Leech→Monster v0.2 — Retooled Spec & Function Map

**Status:** Retooled and restructured from v0.1. This is a faithful, non‑abbreviated restatement that preserves original meaning and intent while tightening definitions, factoring responsibilities, and making the function map implementation‑ready. All modifications are explicitly called out in the “Change Log and Rationale” section; all uncertainties are logged in the “Uncertain Items for Next Review” section.

---

## 0. Purpose (Restated without altering meaning)
A complete, end‑to‑end specification for a seeded, auditable, compute‑once system that scales beyond greedy methods once problem complexity reaches approximately n ≥ 6. The design emphasizes explainability, safety, and reusability via geometric containers (E8 and Leech), glyphic compression, state snapshots (SNAP), and a three‑role governance model (Sentinel, Arbiter, Porter). Every component is adjustable to fit the core idea. All randomness is seeded and replayable. All computations are cached and referenced by content address.

---

## 1. Principles and Operating Assumptions (P1–P8; clarified wording only)
- **P1. Greedy first, geometry when needed.** Default to inexpensive greedy or heuristic solvers. Escalate to the geometric synthesis pipeline (Beamline → E8 → Leech → Monster) when effective dimensionality thresholds are crossed (typically n ≥ 6; optionally at n = 5 when cross‑metric hops are justified).
- **P2. Compute once, recall always.** Treat every decision, state, and synthesis product as an immutable, content‑addressed artifact with complete provenance; prefer reuse over recomputation.
- **P3. Multiplicity is a feature.** Preserve two, four, or eight independent valid paths when they exist; multiplicity increases robustness rather than signaling indecision.
- **P4. Safety is structural.** Governance, legality, feasibility, and observer constraints are represented as first‑class safety gates (Sentinel) and mirrored in geometry through helix rungs.
- **P5. Evidence and governance are dual strands.** A validated result must satisfy both evidence and governance strands with cross‑rung checks before it is stamped as complete.
- **P6. Seeds everywhere.** All stochasticity is explicit and replayable; cross‑seed analysis is a primary learning mechanism.
- **P7. Pluggable containers.** E8 and Leech are defaults; substitute alternative quantizers (for example, OPQ, PQ, HNSW) when residuals or stability indicate the need to do so.
- **P8. Explainability first.** Every transform is recorded as operators over state (SNAP) with shape fingerprints and warrants through Sentinel, Arbiter, and Porter.

---

## 2. Architecture Overview (new: explicit factoring of responsibilities)
**Execution Plane:** Beamline Engine; E8 Lattice Synthesis; Leech Lattice Layouts; Monster Layer.

**Control and Governance Plane:** Sentinel, Arbiter, Porter; Master Tick Scheduler; Review Cues, Health Metrics, Safety Gates.

**Data and Provenance Plane:** SNAP bundles; Glyphs and Glyph Chains; Per‑Shell Indices; Master Digest (Merkle‑DAG); Shape Fingerprints; Seed Lineage; Residual Ledger; Calibration Records.

**Plugin Plane:** Quantization modules; Calibration and Conformal modules; Domain microkernel packs; Bridge detection; Cost and budget models.

**Persistence and Addressing:** Immutable content addressing for SNAP, Glyph, Warrant, Index, and Master artifacts. Cache keys include inputs, policy, layout, monster policy, and seed vector.

---

## 3. Seeded Operation and Cross‑Seed Learning (retained, made implementable)
### 3.1 Seed Vector (unchanged semantics)
- **seed.views:** per‑view noise or perturbation in scoring.
- **seed.mirrors:** micro‑rotations per beamline stage.
- **seed.tiebreak:** deterministic tie‑break ordering for merges and selections.
- **seed.layout:** jitter for choosing among Leech layouts.
- **seed.arbiter:** stochasticity in gray judgments and path selection.
- **seed.porter:** operation ordering within a scheduler tick.

Each execution logs `{vector, parent, generation}` for lineage.

### 3.2 Cross‑Seed Metrics (unchanged semantics; definitions clarified)
- **Consensus cell rate:** percentage of items mapping to identical E8 and Leech cells across seeds.
- **Disagreement entropy map:** spatial heatmap of cross‑seed decisions.
- **Multiplicity profile:** counts of independent paths per seed (two, four, eight) based on the e‑graph of equivalence‑classed derivations.
- **Mutual information from seed lane to outcome:** identifies undue influence of particular lanes.
- **Coverage and novelty:** union and Jaccard measures of cells visited per seed.
- **Pareto set:** seeds on the frontier of quality, risk, and cost; retire dominated seeds.

---

## 4. Beamline Engine (Gold and Gray; +2 → +4 → 8 → 16)
### 4.1 Views and Margins
For item *x*, independent views produce signed scores \( s_i(x) \) and margins \( m_i(x) = |s_i(x)| \).

### 4.2 Gold versus Gray lanes
- **Gold lane:** retain items that are diagonally unambiguous (all margins \( m_i \) greater than or equal to thresholds \( \tau_i \)), or pass an extreme‑view override.
- **Gray lane:** admit low‑margin items with weight \( \alpha \) (typical range 0.4–0.6). A gray item may graduate to gold after later unambiguous confirmation.

### 4.3 +2 → +4 → 8 → 16 synthesis
- Form **+2** from coherent pairs (cosine similarity greater than or equal to \( \gamma \), compatible widths), then merge **+2 ⊕ +2 → +4**, merge two **+4 → 8**, and merge two **8 → 16**.
- Alternate **mirror banks** (sideways checks) and require **diagonal checks** at each level.

### 4.4 Chambering (Optics model)
Four mirror, lens, and gate stages produce four bits that assign items to sixteen chambers. Items accumulate a bit signature used for gating, sensitivity analysis, and explainability.

### 4.5 Confidence and calibration
- Attempt a sixteen‑merge only when **calibrated joint confidence** is greater than or equal to 0.98.
- Calibrate per‑view probabilities and the joint decision boundary (for example, temperature scaling, isotonic aggregation, or conformal aggregation).
- Track contradiction rate; adapt thresholds \( \tau \) and \( \alpha \) accordingly.

**Tunable structure:** permit two to five stages; sixteen is a policy ceiling and not a structural requirement.

---

## 5. Fractal Path Ensemble (two, four, eight)
Maintain an e‑graph of equivalence‑classed derivations so that counts of independent paths are meaningful and do not double‑count commutations. Prefer decisions that retain multiple admissible paths unless governance or cost mandates pruning. Treat multiplicity as a robustness signal and a target to preserve subject to safety and budget constraints.

---

## 6. Effective Dimensionality and the n → Dimension Ladder
### 6.1 Effective dimension (unchanged meaning; measurement made explicit)
- **Linear degrees:** \( \mathrm{rank}_\varepsilon \) of the covariance of view or score trajectories.
- **Fractal component:** correlation dimension derived from neighborhood scaling.

**Decision rule (illustrative; thresholds are policy parameters):**
- \( D_{\text{eff}} < 2.5 \): planar beamline is adequate.
- \( 2.5 \le D_{\text{eff}} < 3.5 \): add a latent half axis; use a partial E8 slice.
- \( D_{\text{eff}} \ge 3.5 \): engage full E8 and Leech.

### 6.2 0 → 8 ladder (formalized; unchanged intent)
- **0 → 1 (Void to Dot):** create initial SNAP; no edges.
- **1 → 2 (Dot to Line):** create first binary edge; log rationale.
- **2 → 3 (Triad):** construct the first triadic glyph (idea, fact, context) with at least ninety percent coverage to qualify.
- **3 → 4 (Square):** add contradiction and support; eligible for +4.
- **4 → 5 (Octant step):** choose one of eight invariant directions; allow a latent half step axis if needed.
- **5 → 6 (Three‑dimensional solid):** any lawful three‑dimensional shape produced from five binary steps; first stable E8 coordinate.
- **6 → 7 (Beyond three‑dimensional):** Sentinel requires external evidence or a bridge; integrate new layout or data.
- **7 → 8 (Full four‑dimensional):** unify valid paths into a four‑dimensional admissible shape; consolidate across layouts.

After eight, the helix phase appears during consolidation (see Section 11, Helix Rungs).

---

## 7. E8 Lattice Synthesis (unchanged intent; clarified data flow)
### 7.1 Coordinates
Divide work into four **+4 groups**. Each group produces a two‑dimensional subspace (for example, learned consensus or novelty through methods such as linear discriminant analysis or canonical correlation analysis) to form an eight‑dimensional vector.

### 7.2 Quantization
Snap to the nearest E8 lattice point (even‑integer or half‑integer cosets). Store residuals. When residuals persist above policy thresholds, switch to an alternate quantizer while retaining parity checks and provenance.

### 7.3 Shells and bridges
- **Shell:** the floor of the eight‑dimensional norm; measures distance from the origin within E8.
- **Bridges:** high‑cosine edges across nonadjacent shells that share invariants, for example feasibility and coverage agreement.

### 7.4 Sensitivity
Maintain a heatmap of chamber and cell flips under small mirror or threshold perturbations. Sentinel uses this map to slow upgrades at ridges and to request more evidence.

---

## 8. Leech Lattice Layouts (twenty‑four dimensions)
### 8.1 Triple projection
For each E8 universe, create three eight‑dimensional projections:
- **E8 for governance and rules.**
- **E8 for context and need.**
- **E8 for observation and limits.**
Concatenate these three projections to twenty‑four dimensions and quantize to the Leech lattice.

### 8.2 Multiple layouts
Maintain a family of layouts, each with at least eight E8 universes. Use covering designs to minimize the count while guaranteeing pairwise and role coverage. For each query, score and consult the top layouts by fit.

### 8.3 Residuals and fallback
Record Leech quantization residuals. If residuals remain high, switch the container to an alternate quantizer such as OPQ, PQ, or HNSW while keeping parity checks for integrity and maintaining address compatibility.

---

## 9. Monster Layer (Holistic understanding and option superstate)
### 9.1 Two roles (definitions made precise; meaning unchanged)
- **Understanding:** the consolidated state across layouts with stable invariants and motifs.
- **Option Superstate:** an e‑graph of admissible operators (lift, project, merge, split, summarize, recall) constrained by governance and budgets.

### 9.2 Budgeted saturation
Limit expansion through explicit cost models and Sentinel, Arbiter, and Porter gates. Expand only warrantable branches and enforce per‑tick option budgets.

---

## 10. Index Hierarchy and Master Digest (compute‑once semantics)
### 10.1 Per‑shell indices
Each shell stores: node content identifiers, neighbors and bridges, addresses (chamber, E8, Leech), stance hash, glyph pointers, SNAP identifiers, provenance (version of RDI, thresholds, mirrors), and Sentinel, Arbiter, and Porter stamps.

### 10.2 Master digest
Maintain a Merkle‑directed acyclic graph over shells and over digests of RDI, Layout, and Monster to name the current understanding. Any policy or layout change produces a new Master.

### 10.3 Cache keys
`Key = H(inputs || rdi || layout || monster_policy || seed_vector)`

---

## 11. Glyphic Compression and Decompression (triadic glyphs)
### 11.1 Codebooks
Learn three codebooks for Governance, Context, and Observation. A glyph is a triple of indices with small weights, a parity value, and back pointers.

### 11.2 Requirements
- Encode only one hundred percent gold and stable items.
- Coverage must be at least ninety percent of the base meaning as validated by reconstruction tests.
- Codebooks are versioned and expanded when a non‑encoded view becomes active.

### 11.3 Chains and quasi‑hashing
Chains of glyphs are Merkle‑hashed with a semantic checksum such as the set of visited cells. Use equivalence checks to skip redundant recomputation.

---

## 12. SNAP: State Nascent Adapting Process
### 12.1 Concept
SNAP is both the unit of work and the unit of recall. A SNAP is a structured bundle that contains stance, location, evidence pointers, shape fingerprint, seed, required microkernels, and replay information.

### 12.2 State swapping
Work proceeds by loading the best SNAP, applying minimal operators, validating through Sentinel, Arbiter, and Porter, and writing a SNAP delta.

### 12.3 Helix rungs (post‑eight validation)
Represent dual strands of Evidence and Governance. Required rungs are:
- **Legality:** explicit rule compliance.
- **Safety:** risk thresholds enforced.
- **Fairness:** observer constraints respected.
- **Provenance completeness:** sources and derivations present and linked.
- **Observer repeatability:** results reproducible under stated instruments.

A missing rung triggers Sentinel stop, demotes synthesis to eight, or fetches additional data.

---

## 13. Shape Fingerprints (stance, location, and subgraph trace)
Create a canonical, typed subgraph of nodes and edges touched under a stance. Compute the Weisfeiler‑Lehman hash for structure and a BLAKE3 digest for content. Store multiplicity, shell span, and involved layouts. The shape identifier is a pair of the structural hash and the content digest.

---

## 14. Sentinel, Arbiter, and Porter Governance (separation of powers)
### 14.1 Roles and write permissions
- **Sentinel:** monitors and gates through vetoes, parity checks, confidence checks, and policy alignment. The Sentinel has no write permissions.
- **Arbiter:** resolves gray space and issues move warrants with rationale and fallbacks. The Arbiter has no write permissions.
- **Porter:** is the only mover. The Porter executes warrants by writing SNAP deltas and is idempotent with post checks.

### 14.2 Escalation
At each shell and layer (Beamline, E8, Leech, Monster), operate local units and family heads. Any unresolved top‑level item triggers a stop‑the‑world pause for that scope until intervention.

### 14.3 Warrants
A warrant includes stance hash, RDI digest, shape identifier, identifiers for the before and after SNAPs, confidence, veto scan, scope, time‑to‑live, alternatives, and rationale.

---

## 15. Master Tick Scheduler (importance‑modulated sub‑ticks)
**Goal:** bound the amount of work performed per global tick, modulated by task importance.

- Allocate quanta such that each class receives at least a configured minimum to avoid starvation and the sum of quanta across classes equals the available budget per tick. Distribute any remaining quanta in proportion to importance and risk reduction value. Sentinel may veto or rate limit; Arbiter may boost gray resolution or consensus; Porter executes within warrant time‑to‑live. Maintain explicit classes for gray resolution, governance audits, calibration refresh, layout consensus, and option exploration.

*Note: the original draft contained placeholders for explicit formulas. This version states the policy without fixing numeric constants; see “Uncertain Items for Next Review.”*

---

## 16. Review Cues, Health Metrics, and Safety Gates
### 16.1 Review cues (selected)
Calibration drift; rising reliance on extreme views; persistent E8 or Leech residuals; cell ping‑pong under micro‑perturbations; multiplicity erosion from four to two to one; rising Porter rollback rate; Sentinel stop storms; seed monoculture; spikes in mutual information from seed lane to outcome; SNAP churn; glyph collisions; Master thrash; persistent layout disagreement; information gain plateaus; sensitivity ridges.

### 16.2 Metrics
- **Safety:** veto hits, parity failures, contradiction rate below eight percent, confidence calibration error.
- **Efficacy:** objective quality, multiplicity realized, plan stability.
- **Efficiency:** cache hit percentage, recomputation avoided, glyph coverage percentage, SNAP reuse percentage, time to redline limits.
- **Geometry:** cell stability, shell coverage, and bridge density.
- **Governance:** warrant acceptance, rollback rate, and time to clear Sentinel stops.

### 16.3 Gates
- **Base‑idea closure gate:** run shadow glyph tests; require a closure score before attempting a sixteen‑merge.
- **Ninety‑eight percent joint confidence** required before any sixteen‑merge attempt, using calibrated or conformal aggregation.

---

## 17. Pluggable Quantization and Microkernel Packs
### 17.1 Quantizers
Defaults are E8 and Leech with residual tracking. Fallbacks include OPQ, PQ, and HNSW cell grids when residuals or stability fail. Retain Golay‑style parity as integrity checks even with alternate quantizers.

### 17.2 Microkernel packs (toggle on capabilities)
`e8_quant`, `leech_quant`, `glyph_codec`, `bridge_detector`, `beamline_scheduler`, `arbiter_costs`, `calibration_pack`, `conformal_pack`, domain packs such as `superperm_pack`, and governance packs such as `legal_vetoes` and `fairness_vetoes`.

---

## 18. APIs and JSON5 Schemas (clarified types; unchanged meaning)
### 18.1 SNAP (Node)
```
{
  id: "snap:node:…",
  version: 1,
  rdi_digest: "rdi:…",
  stance: { governance: {...}, context: {...}, observation: {...}, stance_hash: "st:…" },
  seed: { vector: { views:…, mirrors:…, tiebreak:…, layout:…, arbiter:…, porter:… }, parent: "seed:…", generation: 2 },
  location: { chamber: "0b1011", e8: { coord: [ …8 ], cell: "E8⟨…⟩", shell: 2 }, leech: { coord: [ …24 ], cell: "LL⟨…⟩", layout: "L-023" } },
  evidence: { gold_glyphs: [ "g:…" ], coverage: 0.93, confidence: 0.991, contradiction_rate: 0.03, invariants: [ "feasibility", "overlap" ] },
  neighbors: [ { id: "snap:…", edge: "feasible_next" } ],
  shape: { shape_id: "shp:…", wl: "wl:…", b3: "b3:…", multiplicity: 4, shell_span: [ 2, 3 ], layouts: [ "L-023", "L-017" ] },
  master_links: { shell_index: "sh:…", master: "m:…", cids: [ "cid:…" ] },
  microkernels: { required: [ "e8_quant", "glyph_codec" ], optional: [ "bridge_detector" ] },
  resume: { ledger_offset: 18421, cache_keys: [ "key:…" ] }
}
```

### 18.2 SNAP (Glyph Chain)
```
{
  id: "snap:glyphchain:…",
  glyphs: [ "g:1a2b", "g:5c6d", "g:77ee" ],
  chain_merkle: "mr:…",
  semantic_checksum: "sc:ll:LL⟨…⟩+e8sum:…",
  backptrs: [ "cid:a1…", "cid:b2…" ],
  coverage: 0.95,
  expands_to: { min_ops: 2, max_ops: 12 },
  location_hint: { layout: "L-023", shell: 2 },
  stance_hash: "st:…"
}
```

### 18.3 Warrant (Sentinel, Arbiter, Porter)
```
{
  id: "warrant:…",
  issued_by: "arbiter:leech:k=2",
  stance_hash: "st:…",
  shape_id: "shp:…",
  before: "snap:…",
  after:  "snap:…",
  confidence: 0.985,
  veto_scan: { legal: false, feasibility: false, fairness: false },
  scope: { layer: "leech", shells: [ 2, 3 ] },
  ttl_sec: 900,
  alternatives: [ "path:A", "path:B" ],
  rationale: "layout consensus across L-023, L-017; minimal residual"
}
```

### 18.4 Glyph
```
{
  id: "g:1a2b",
  dict_version: 3,
  triad: { g_idx: 12, c_idx: 87, o_idx: 5, weights: [ 0.9, 1.1, 0.8 ] },
  parity: "golay24:…",
  coverage: 0.92,
  backptrs: { e8_cell: "E8⟨…⟩", leech_cell: "LL⟨…⟩", cids: [ "cid:…" ] }
}
```

---

## 19. Cycle of Work Algorithms (kept; sequencing made explicit)
### 19.1 Per‑tick (high level)
1. Select stance for governance, context, and observation vectors.
2. Allocate seeds for exploration versus exploitation according to the scheduler.
3. Load the best SNAP per seed; if none exists, initialize the zero to one step.
4. Run the Beamline to propose quartets; update gold and gray lanes; refresh calibration and confidence.
5. Synthesize +2 to +4 to 8; attempt 16 only if calibrated joint confidence is at least 0.98.
6. Perform E8 and Leech coordinate assignment, quantization, and residual checks; discover and record bridges.
7. Expand Monster options within budgets; respect Sentinel, Arbiter, and Porter gates.
8. Encode stable gold into glyphs; update glyph chains.
9. Write SNAP delta after Sentinel pass, Arbiter warrant, Porter commit, and Sentinel post check.
10. Update per‑shell index and Master digest.
11. Compute cross‑seed consensus and disagreement; adapt \( \alpha \) and \( \tau \); plan the next seeds.

### 19.2 Base‑idea closure gate (before a sixteen‑merge)
Run shadow glyph tests by randomly decompressing and recomposing K glyphs under a fresh seed. If any mismatch occurs, block the sixteen‑merge. Compute a closure score as the percentage of dependencies covered by glyphs with intact helix rungs; require a policy threshold.

---

## 20. Example Walkthrough (n = 5 → n = 6 → n = 7)
- **n = 5:** Effective dimension is approximately 2.6; add a latent half axis; form a +4; choose one of eight directions; quantization residual is modest; continue the Beamline with a partial E8 slice.
- **n = 6:** Effective dimension is approximately 3.1; promote to full E8; build an eight‑dimensional coordinate from four +4 groups; shells one to two are occupied; attempts at sixteen are blocked until joint confidence is at least 0.98.
- **n = 7:** Effective dimension is approximately 3.6; engage Leech layouts; two layouts disagree; Arbiter triggers context‑seeking seeds; bridges form to incorporate external evidence; consensus is reached; a sixteen‑merge is attempted and validated after helix rungs pass.

---

## 21. Risks, Redesign Options, and Open Questions (no change in meaning)
- **Risk of option explosion:** maintain budgeted e‑graph expansion; enumerate only warrantable branches; use a cost‑aware beam search.
- **Risk of over‑reliance on E8 and Leech:** keep OPQ and HNSW fallbacks; treat parity as a checksum; choose the container by residuals and stability.
- **Risk of calibration drift:** use conformal aggregation and periodic recalibration; Sentinel monitors joint reliability curves.
- **Risk of glyph drift and collisions:** version codebooks; auto‑expand for missing views; retain parity and back pointers and semantic checks.
- **Risk of Sentinel stop deadlocks:** use scope‑limited freezes; fall back to read‑only stances; time‑box arbitration.
- **Risk of seed monoculture:** enforce seed family diversity; use low‑discrepancy sequences for mirrors; use stratified sampling for tie‑breaks.

**Open questions:** best universal two‑dimensional subspace per +4; when to prefer continuous neighborhoods over strict cell equality for layout consensus; how to price governance rungs in the cost model for e‑graph expansion; minimal rung sets per regulated domain.

---

## 22. Implementation Notes (v0.1 roadmap preserved; sequencing clarified)
- Calibration pack for per‑view and joint calibration with conformal guard.
- Adaptive parameters for \( \alpha \) and \( \tau \) based on contradiction rate feedback.
- Pluggable quantizers with residual logs and switch policies.
- Glyph codec version one (triad, parity, back pointers).
- SNAP version one schema wired to a ledger with shape fingerprints.
- Sentinel, Arbiter, and Porter version one (warrants, idempotent Porter, scoped Sentinel stop).
- Seed dashboards for consensus, entropy, multiplicity, residuals, and Sentinel stop storms.

---

## 23. Function Map (new: explicit inputs, outputs, preconditions, postconditions)
### 23.1 Beamline Engine
- **Inputs:** set of items with view scores and margins; thresholds for \( \tau \), \( \alpha \), and \( \gamma \); seed vector; chamber configuration.
- **Outputs:** updated gold and gray partitions; proposed +2, +4, 8, and optional 16 merges; chamber bit signatures; calibrated joint confidence.
- **Preconditions:** calibrated per‑view probabilities within error limits; thresholds set; seed vector present.
- **Postconditions:** merges recorded with provenance; contradictions logged; items carry updated chamber signatures.

### 23.2 E8 Lattice Synthesis
- **Inputs:** outputs from Beamline; four +4 groups; subspace methods; quantizer; residual policy.
- **Outputs:** eight‑dimensional coordinates; E8 cell and shell; residuals; bridges; sensitivity map.
- **Preconditions:** +4 groups satisfy coherence; calibration current; governance gates satisfied for promotion.
- **Postconditions:** quantization recorded; residuals logged and monitored; bridges added to indices.

### 23.3 Leech Lattice Layouts
- **Inputs:** E8 universes; triple projection definition; layout family; residual policy.
- **Outputs:** twenty‑four‑dimensional coordinates; Leech cell; selected layout identifiers; residuals.
- **Preconditions:** E8 coordinates valid; layout scoring policy defined.
- **Postconditions:** layout decisions recorded; fallbacks applied when residuals exceed thresholds.

### 23.4 Monster Layer
- **Inputs:** consolidated indices; glyph chains; governance budgets; option operators.
- **Outputs:** admissible option superstate; executed warrants; updated SNAP deltas; budget utilization reports.
- **Preconditions:** Sentinel gates satisfied; Arbiter warrants issued; Porter ready to commit.
- **Postconditions:** options expanded within budgets; understanding updated; Master digest incremented.

### 23.5 Sentinel, Arbiter, and Porter
- **Inputs:** candidate actions, confidence and calibration records, veto scans, scope, and time‑to‑live.
- **Outputs:** warrants, vetoes, rollbacks, and commits; post checks.
- **Preconditions:** sufficient evidence; rungs intact; scope defined.
- **Postconditions:** actions either committed or blocked with explicit rationale.

### 23.6 Glyphic Compression
- **Inputs:** gold and stable items; codebooks; reconstruction tests.
- **Outputs:** glyphs and glyph chains; semantic checksums; parity and back pointers.
- **Preconditions:** stability verified; coverage tested.
- **Postconditions:** encoded artifacts addressable by content identifier; chains hashed in Master.

### 23.7 SNAP and Indices
- **Inputs:** stance, location, evidence, shape; seed vector; microkernels; ledger offsets.
- **Outputs:** immutable SNAP artifacts; per‑shell indices; Master digest updates.
- **Preconditions:** warrant present for writes; cache keys prepared.
- **Postconditions:** artifacts committed; indices updated; Master advanced.

---

## 24. Change Log and Rationale (all changes explained)
1. **Architecture overview added.** Purpose: make component boundaries and planes explicit so that the function map is implementable without guessing responsibilities. Meaning preserved.
2. **Function map added.** Purpose: bind each major component to inputs, outputs, preconditions, and postconditions to reduce ambiguity during implementation. Meaning preserved.
3. **Scheduler policy wording completed without fixing formulas.** Purpose: the v0.1 text had placeholders for formulas. This version states the allocation policy in words to avoid introducing invented constants while keeping the intent. Meaning preserved.
4. **Explicit separation of data and provenance plane.** Purpose: clarify where immutable artifacts and content addressing live. Meaning preserved.
5. **Clarified definitions in E8 and Leech sections.** Purpose: remove implicit assumptions and make residual handling and fallback policies explicit. Meaning preserved.
6. **Expanded warrant definition usage narrative.** Purpose: ensure the steps of Sentinel pass, Arbiter warrant, and Porter commit are operationally clear. Meaning preserved.
7. **Sensitivity analysis called out in E8.** Purpose: highlight use of ridge awareness by Sentinel to slow upgrades. Meaning preserved.
8. **No change to thresholds or policy constants.** Purpose: avoid inventing numeric meaning; retain original thresholds exactly as stated.

---

## 25. Uncertain Items for Next Review (logged verbatim and non‑inventive)
1. **Master Tick Scheduler formulas were placeholders.** The exact mathematical form for “allocate quanta such that …” was not specified in v0.1. This v0.2 states a verbal policy only. Decision needed: exact formula and constants for importance weights, minimum quanta, and risk reduction value.
2. **Best universal two‑dimensional subspace per +4 group.** Options include linear discriminant analysis, canonical correlation analysis, or domain meta‑features. Decision needed: default method and switching policy.
3. **Consensus decision when to prefer continuous neighborhoods versus strict cell equality for layout consensus.** Decision needed: conditions for neighborhood tolerance and the metric used.
4. **Pricing of helix rungs in the cost model for option expansion.** Decision needed: explicit costs per rung and how they interact with the budget.
5. **Minimal rung sets per regulated domain.** Decision needed: domain templates for safety‑critical domains versus research domains and the process for activation.
6. **Definition of “widths compatible” in +2 formation.** Decision needed: formal criterion for width compatibility beyond cosine similarity.
7. **Definition of “extreme‑view override.”** Decision needed: policy for allowing an extreme view to override diagonal ambiguity, including calibration guardrails.
8. **Residual thresholds for switching quantizers.** Decision needed: explicit residual metrics and values that trigger alternate quantizer selection while maintaining parity guarantees.
9. **Bridge creation policy across shells.** Decision needed: quantitative definition for “high‑cosine edges across nonadjacent shells” and any governance checks specific to bridges.
10. **Gray graduation to gold.** Decision needed: exact rules and evidence required for a gray item to become gold, including time or tick‑based constraints.
11. **Seed allocation for exploration versus exploitation.** Decision needed: specific policy for splitting seeds between exploration and exploitation as a function of uncertainty and budget.
12. **Equivalence‑classing rules for the e‑graph.** Decision needed: a canonical set of rewrite rules and commutations that define “independent paths.”
13. **Shadow glyph test parameter K.** Decision needed: choice of K and the draw policy for fresh seeds during closure checks.
14. **Observer repeatability instruments.** Decision needed: enumerations of allowed instruments and the reproducibility protocol per domain.

---

## 26. Ready‑to‑Build Checklist (non‑inventive tasks derived from v0.1)
- Establish immutable content addressing and the ledger for SNAP, Glyph, Warrant, Index, and Master artifacts.
- Implement per‑view and joint calibration with conformal guard; record calibration curves and drift monitors.
- Implement Beamline with chamber bit signatures, gold and gray partitions, and +2, +4, 8, and 16 synthesis gates.
- Implement E8 quantization with residual tracking, sensitivity mapping, and bridge detection.
- Implement Leech layouts with triple projection and residual‑based fallback to alternate quantizers.
- Implement Sentinel, Arbiter, and Porter with warrants and scoped stop handling; ensure only the Porter writes.
- Implement glyph codec with triads, parity, back pointers, and chain hashing with semantic checksums.
- Implement the Master Tick Scheduler with minimum quanta per class and proportional allocation of remaining quanta; leave numeric constants configurable.
- Implement cross‑seed metrics and dashboards for consensus, disagreement entropy, multiplicity, residuals, and stop storms.
- Finalize policies identified in “Uncertain Items for Next Review.”

---

**End of v0.2.** This document preserves the original conceptual structure and terminology while restructuring for buildability and auditability. No new meaning has been introduced; where numeric or policy details were absent, they remain uncommitted and are logged for decision.

---

## 27. Main and Sub-Tick Policy (User-Directed Focus)

User directive: The main tick focuses on exploration of possible options. The sub-ticks focus on filling all internal gaps and solving all tasks requested of the system.

### 27.1 Invariants (apply at all times)
1. Lexicographic gates: Hard constraints (Legality, Safety, Fairness, Provenance completeness, Observer repeatability) must pass before any scoring. Violations trigger Sentinel STOP and are handled before exploration or execution.
2. Compute-once semantics: All actions produce immutable artifacts (SNAP, Warrant, Index, Master) with content addressing.
3. Budget bounds: Each global tick has a fixed compute budget B_total. Sub-ticks partition their share but may not exceed it.

### 27.2 Definitions
- Option (o): An admissible operator sequence in the Monster option e-graph (lift, project, merge, split, summarize, recall), pre-gated by Sentinel.
- Gap (g): Any flagged deficiency from Section 16 (e.g., calibration drift, high residuals, missing helix rung, layout disagreement, multiplicity erosion, contradiction spikes, sensitivity ridges).
- Requested task (t): A concrete, user-requested outcome with a dependency DAG expressible as SNAP deltas.
- Budgets: B_main (exploration), B_sub (gap filling + task execution), with B_main + B_sub = B_total.

### 27.3 Main tick objective (exploration-first)
Maximize expected frontier expansion under cost and risk limits.

Score S(o):
S(o) = lambda1*UCB(u_hat(o), sigma(o))
       + lambda2*VoI(o)
       + lambda3*DeltaMultiplicity(o)
       + lambda4*DeltaCoverage(o)
       + lambda5*DeltaBridgeDensity(o)
       - lambda6*Cost(o)

UCB is an upper-confidence bound term (e.g., u_hat + beta*sigma). VoI is expected reduction in uncertainty relevant to decisions. The deltas reference geometry/index metrics from Sections 6, 7, and 16. All lambdas and beta are policy parameters learned or tuned via tests; no numeric values are fixed here.

Constraints:
- Hard gates must pass (Section 11 helix rungs).
- Risk ceilings per class (e.g., contradiction rate, calibration error) cannot be exceeded.
- Option budgets: per-tick cap on number or total cost of expanded branches.

Selection: Choose top-K options by S(o) within B_main; issue Arbiter warrants; Porter executes; Sentinel performs post-checks.

### 27.4 Sub-ticks objective (gap-first and task-complete)
Minimize unresolved gap load while completing requested tasks inside B_sub.

Gap load L:
L = sum_over_g ( w_g * max(0, m_g - theta_g) )
where m_g is the current measured value of gap metric g (e.g., residual, drift, contradiction), theta_g its policy threshold, and w_g its priority weight.

Execution order (deterministic policy):
1) Helix rung completion (make results admissible).
2) Calibration and conformal refresh for flagged views.
3) Gray->Gold migrations and consensus consolidation.
4) Layout consensus and residual reduction.
5) Index/Master updates and glyphization.
6) User requested tasks (respect TTL and dependency DAG), preempting steps 1–5 only if tasks are gated on them.

Completion rule: Iterate through the above until B_sub is consumed or L = 0 and all requested tasks in scope are complete.

### 27.5 Seed policy coupling
Exploration choices update seed allocation for the next main tick using cross-seed consensus, disagreement entropy, and multiplicity profiles. Maintain diversity via stratified sampling and low-discrepancy sequences for mirrors.

### 27.6 Test plan (to determine exact variables without inventing numbers)
1. Offline replay: Use ledgered runs to estimate u_hat, sigma, VoI proxies, and geometry deltas; grid-search or Bayesian optimize lambdas and beta against Section 16 metrics.
2. Cross-seed A/B: Run paired seeds with alternative weights; compare multiplicity realized, bridge density, closure scores, and time-to-clear Sentinel stops.
3. Conformal guard: Validate that confidence calibration remains within bounds when exploration pressure increases.
4. Safety audits: Verify no increase in parity failures or contradiction rates beyond policy limits.

### 27.7 API stubs (for configurability)
```
SchedulerPolicy {
  B_total,
  B_main_fraction,
  K_options,
  lambdas: { ucb, voi, multiplicity, coverage, bridges, cost },
  beta_ucb,
  gap_weights: { residual, drift, rung_missing, layout_disagree, contradiction, multiplicity_erosion, sensitivity_ridge },
  thresholds: { ... },
  risk_ceilings: { contradiction, cal_error, veto_rate },
  preemption: { allow_task_preempt: true|false }
}
```

### 27.8 Open decisions (recorded, no new meaning)
- Exact lambdas, beta, gap weights, and thresholds.
- VoI estimator form (analytic vs. Monte Carlo under seeds).
- K (number of options) selection as a function of budget and uncertainty.
- Preemption policy for urgent user tasks versus gap-filling when gates already pass.


