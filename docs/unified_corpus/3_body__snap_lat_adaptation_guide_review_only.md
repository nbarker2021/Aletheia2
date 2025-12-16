# 3‑Body → SnapLat Adaptation Guide (Review‑Only)

**Intent**
Translate the human‑reasoning "3‑body" document into SnapLat operational constructs. No design lock; this is a creative cross‑domain adaptation that preserves your goals: Gate‑3‑6‑9, Lens‑8 rotation, N_label vs N_bits, containment C*, SnapPacks, Trails, CSCs, and deliberate faulting.

---
## A. Concept Map (Doc → SnapLat)

| Doc notion | SnapLat construct | Short rationale |
|---|---|---|
| Three bodies / triplets | **Gate‑3 Triad** (triad selector) | Minimal contrast unit to force ΔU (bit reduction). |
| Anti‑tensor / tensor / neutralizer | **Variant± + Neutral** roles in triad | Gives a falsifiable invariant at Gate‑6. |
| Rotation around idea | **Lens rotation** (swap outside context) | Unlocks collapse when ΔU stalls; triggers PerspectiveShift. |
| Mirror/palindrome law | **MirrorPass** check (1…n…1) | Promotion guard; requires Echo<N> witnesses. |
| Contextual awareness / field lens | **Lens L** operator | Truth/legality/context constraints; orients next tests. |
| Single snapshot infers motion | **Static‑shape prepass** | Low‑compute predictor of collapse path before heavy work. |
| Two triads ⇒ fork at N=5 | **N=5 gate w/ Lens‑8** | Try 8 rotated lenses to cover alternatives before promote. |
| Human cognition triad | **Calc / Recall / Counter‑tensor axes** | Diagnoser; map to scheduler priorities and safety. |

---
## B. Gate‑3‑6‑9 Operational Pattern (under Lens L)

### Gate‑3 (Triad)
- **Select 3 bodies** `{b1,b2,b3}` by acquisition score `Score = α·ΔU + β·Bridge + γ·WeylGain + δ·ContainmentGain – λ·Cost`.
- **Run binary checks** picked by Lens L (maximize ΔU/time).
- **Exit**: ΔU improves and polarity conflict drops; else `PerspectiveShift` (rotate lens or bring new ontology).

### Gate‑6 (Hexad)
- **Add 3 bodies** maximizing coverage gaps and bridge impact.
- **Derive an invariant** from two opposing variants (V+ vs V−). Make it falsifiable with adversarial counter‑predicates.
- **Exit**: invariant validated under Lens L and ΔU still improving.

### Gate‑9 (Ennead)
- **Compose 8 facets + Lens** (center). Require MirrorPass, reversibility witness, polarity bound, and sufficient containment `C*`.
- **Outcome**: promote to **CSC** (Checkpoint Safe Cube) or demote unstable parts to **E‑DBSU** with reasons.

---
## C. Lens‑8 Rotation (9th as rotating lenses)

**Idea**: The 9th is not a peer facet; it is a **cycle of 8 lenses** `{L₁..L₈}` (distinct outside contexts). For N=5 ambiguity, evaluate triad/hexad/ennead under each Lᵢ before promotion.

**Policy**
- Maintain a **Lens‑8 schedule** with non‑isomorphic constraint stacks.
- Rotate lenses when (i) ΔU stalls with high `C*`, (ii) MirrorPass fails, or (iii) Weyl coverage is uneven.
- Tie superpermutation **n=5 octet solves** to lenses: each solve is validated under at least one Lᵢ; coverage aims to span all Lᵢ.

**Acceptance**
- Keep Lens‑8 only if it reduces ticks‑to‑promotion, increases MirrorPass rate, and controls E‑DBSU growth.

---
## D. Metrics & Witnesses
- **ΔU_top / tick**; **C*** roll‑up; **Weyl coverage**; **neg_conflict**.
- **MirrorPass** with Echo<N> trace; **1729 coverage**; **reversibility**.
- **CSC yield**; **demotion reasons**; **drift** of SnapPacks.

---
## E. Traits & Metaclasses (compile‑time guards)
- **Glyphable**: 3‑word glyph encode/decode; emits ΔU, ΔC* deltas.
- **Shellable**: reports shell depth, containment vector.
- **BridgeAware**: surfaces bridge touches and E‑DBSU risks.
- **TrailEmitting**: all actions logged (agent+snap fingerprints).
- **Isolatable**: mannequin/porter boundaries enforced.
- **Reconstructible**: reversibility witness generators.

Metaclasses (`GlyphMeta`, `MDHGMeta`, `AGRMMeta`) enforce trait presence and pre/post hooks (e.g., mirror checks on class creation for spec objects).

---
## F. SnapPacks & CSCs
- **SnapPack**: compiled state (glyph record, decision graph, Weyl table, witnesses, seeds, Trails, TTL/version, codebook hash).
- **CSC**: a SnapPack that passes Gate‑9 under Lens‑8 policy; becomes a safe waypoint for scouts.
- **Freshness**: TTL + drift monitors; dedup by codebook hash.

---
## G. Fault‑by‑Design
- **ContextFlip**: swap lens mid‑gate; must not break true CSCs.
- **BridgeFlip**: temporarily elevate double‑unknown bridge to test stability.
- **NegationProbe**: assert invariant inverse; require bounded failure.
- **Outcome**: unstable assets demoted with reason; perspective shifts queued.

---
## H. Superpermutation Alignment (n=5)
- Maintain **8 distinct n=5 solves** (octet).
- Map each to a **lens profile**; require that the family of solves covers all Lens‑8 cases or justify exceptions.
- Promotion at N=5 requires **octet witnesses** + Lens rotation evidence (where used).

---
## I. Example Flow (condensed)
1) TaskSpec detects ≥k fields → compute complexity vector; set initial lens L₁.
2) **Gate‑3**: pick triad; ΔU improves; Trails record.
3) **Gate‑6**: add 3, derive invariant; polarity ok; L₁ passes.
4) ΔU stalls; rotate to **L₂**; run adversarial probes.
5) **Gate‑9**: complete facets; MirrorPass holds; **CSC** minted; SnapPack stored.
6) Next tasks mostly use CSC via Recall Router; recompute only on drift/fault.

---
## J. Risks & Mitigations
- Lens overfit → TTL + drift + diversity checks.
- State bloat → dedup + cap per topic; merge identical codebook cells.
- Confirmation bias → enforce adversarial probes before promotion.
- Scheduler starvation → preemptive ticks, hemisphere quotas, admission control.

---
## K. What to Derive/Prototype Next (still review‑only)
- **Lens‑8 interface** (constraints, orientation, refutation hooks) and a minimal harness to rotate lenses across Gate‑3‑6‑9.
- **Invariant Engine** spec (two‑variant → invariant with adversarial margins).
- **Static‑shape prepass** template to predict collapse vectors cheaply.
- **CSC Builder** draft (inputs/outputs, witnesses, pack format) with Trails.
- **Superperm binding**: policy for mapping octet solves to lens profiles and coverage checks.

> When approved for build, these sections turn into specs with tests and CLI targets (gate369, lens, csc, recall).



---

## Lens‑8 Rotation + Superpermutation Alignment — Derived Operations (Review‑Only)
**Perspective:** 9th = rotating context lens; 8 facets are evaluated under each lens. Use for N=5 ambiguity clearing and CSC formation.

### Operational Rules
- **Gate‑3:** form triads (Variant⁺ / Variant⁻ / Neutral). Choose by ΔU/time, Bridge impact, WeylGain, ContainmentGain, Cost. Evaluate under current Lens Lᵢ.
- **Gate‑6:** add 3 bodies; assert a falsifiable invariant (two‑variant max‑margin). Validate under Lᵢ with adversarial counter‑predicates.
- **Gate‑9:** complete 8 facets + Lᵢ; require MirrorPass (1…n…1), reversibility, polarity bounds, sufficient containment **C***.
- **Rotate lenses** {L₁…L₈} when ΔU stalls at high **C***, MirrorPass fails, or Weyl coverage is uneven. Treat rotation as “outside universe comparison.”
- **Octet binding (n=5):** maintain 8 distinct n=5 solves; each must be validated under at least one Lᵢ. Favor even coverage across lenses before promotion.

### Data Structures
- **TriadRecord:** {members, tests, ΔU trace, polarity, witnesses}.
- **HexadInvariant:** {V⁺, V⁻, invariant, margin κ, counter‑witnesses}.
- **EnneadPackage:** {8 facets, lens Lᵢ, MirrorPass, Weyl table, containment vector, reversibility}.
- **SnapPack/CSC:** packaged Ennead with Trails, TTL, dedup hash; eligible for recall routing.

### Scheduler & FSM (additions)
- Tick scheduler selects next binary predicate maximizing expected ΔU/time subject to raising low‑**C*** shells first.
- FSM substates: G₃/G₆/G₉ run **under Lᵢ**; promotion requires `[C*≥θC ∧ U≤θU ∧ MirrorPass ∧ Lᵢ(pass)]`.
- **PerspectiveShift:** if ΔU ≈ 0 at high **C*** under Lᵢ, rotate to Lᵢ₊₁ or import new ontology; re‑run G₃/₆/₉.

### Fault‑by‑Design
- **ContextFlip:** lens swap mid‑gate; CSCs must remain stable or demote with reason.
- **BridgeFlip:** transient promotion of double‑unknown bridges to test stability; revert on failure.
- **NegationProbe:** run invariant inverse; allow bounded failures only.

### Metrics & Acceptance
- ΔU/tick, **C*** roll‑up, Weyl coverage, MirrorPass rate, demotion reasons, CSC yield, Lens‑8 coverage uniformity, E‑DBSU budget.
- Keep Lens‑8 rotation only if it lowers median ticks‑to‑promotion at N=5, raises MirrorPass, and keeps E‑DBSU in budget.

### Open Research Hooks (non‑binding)
- Map “two‑triad ⇒ N=5” hybrids to facet **intersections** under lenses; test against octet solves.
- Static‑shape prepass to predict collapse vectors before heavy work; embed as an early scout heuristic.
- Learn lens ensembles with diversity regularizers to avoid redundant contexts.



---

## Lens Bank — Cross‑Domain Adaptations (Review‑Only)
**Goal:** Mine the 3‑body log through multiple creative lenses and translate into SnapLat operations. No design lock; review and hypothesis only.

### 1) Lens‑8 Baseline (rotating outside contexts)
- **What:** Eight non‑isomorphic context stacks `{L₁…L₈}`; the 9th is the rotation operator.
- **Use:** At N=5 ambiguity or ΔU stall with high containment **C***, rotate lenses across Gate‑3/6/9 before promotion.
- **Measure:** ΔU/tick, MirrorPass, Weyl coverage uniformity, CSC yield, E‑DBSU growth.

### 2) Legality‑First Lens (PolicyGuard)
- **Objective:** Maximize ΔU under strict legality/ethics constraints; block predicates that violate rules.
- **Selectors:** Prefer tests with strong polarity reductions that are certifiable (witnessable) and reversible.
- **Deployment:** Must pass mannequin/porter isolation; logs decision basis for audit.

### 3) Novelty‑First Lens (Discovery)
- **Objective:** Surface **facet intersections** that produce high mutual information and low redundancy.
- **Selectors:** Highest expected information gain; require adversarial counter‑predicates before acceptance.
- **Safeguard:** Ambiguity budget; demote if novelty lacks cross‑lens stability.

### 4) Safety‑First Lens (Risk‑minimizing)
- **Objective:** Minimize neg_conflict and hazard scores while preserving ΔU.
- **Selectors:** Prefer invariants with wide margins and low drift; conservative promotion thresholds.
- **Use:** Default for live‑adjacent tasks.

### 5) Completeness‑First Lens (Containment)
- **Objective:** Maximize **C*** (interior shelledness) before pushing ΔU → 0.
- **Selectors:** Tests that lift the lowest shell components; prioritize missing witnesses (1729, reversibility, Weyl gaps).
- **Trigger:** If ΔU/tick drops but **C*** is low, switch here.

### 6) Efficiency‑First Lens (Compute/time)
- **Objective:** Maximize ΔU/time subject to minimal cost; strong for early scouting.
- **Selectors:** Cheap predicates predicted by static‑shape prepass; low I/O, high return.
- **Use:** First passes and time‑boxed sweeps.

### 7) Symmetry‑First Lens (Mirror)
- **Objective:** Enforce palindromic structure; if symmetry breaks, force Echo<N> walkback.
- **Selectors:** Tests that prove or refute mirror compliance quickly.
- **Outcome:** Failures queue PerspectiveShift; successes become strong CSC candidates.

### 8) Counterfactual‑First Lens (Negation)
- **Objective:** Stress invariants; attempt the inverse under controlled scope.
- **Selectors:** Maximize boundary coverage around the invariant margin; measure robustness.
- **Use:** Pre‑promotion sanity check; pairs naturally with Novelty.

### 9) Stability‑First Lens (Drift)
- **Objective:** Prefer low‑drift claims; minimize future freshness cost.
- **Selectors:** Tests whose results persist across time/context perturbations.
- **Policy:** SnapPacks with high drift are never canonical; require re‑verification windows.

### 10) Cross‑Domain‑First Lens (Ontology Jump)
- **Objective:** Pull in an alternate ontology/universe to break local minima.
- **Selectors:** Bridges that connect to other corpora; require Weyl coverage improvement.
- **Use:** When repeated single‑domain rotations fail.

---

## Gate‑3‑6‑9 Under Lenses — Operational Template
- **Gate‑3 (Triad):** pick `{b1,b2,b3}` by acquisition score `α·ΔU + β·Bridge + γ·WeylGain + δ·ContainmentGain − λ·Cost`, evaluated under lens Lᵢ. Run binary predicates; record Trails.
- **Gate‑6 (Hexad):** add 3; derive **two‑variant invariant** with adversarial counter‑witnesses; verify under Lᵢ.
- **Gate‑9 (Ennead):** assemble 8 facets + Lᵢ; require MirrorPass, reversibility, polarity bound, **C*** ≥ θC. Promote to CSC or demote with reasons.
- **Rotation policy:** Bandit‑style lens scheduler (UCB or Thompson) with reward = ΔU/time − penalty(E‑DBSU growth); hard caps on rotation frequency to avoid thrash.

---

## Static‑Shape Prepass (Scout Heuristic)
- **Input:** Single snapshot of triad/hexad graph with provisional bridges.
- **Output:** Ranked binary predicates with estimated ΔU/time and symmetry impact; candidates for Efficiency/Novelty lenses.
- **Benefit:** Cheap entropy removal before heavy shelling.

---

## Superpermutation (n=5) Alignment — Review Hypotheses
- **H‑Perm‑Lens:** Bind each distinct n=5 solve to at least one lens profile; rotating lenses should increase evenness of coverage and reduce ticks‑to‑promotion at N=5.
- **H‑Triad‑Intersections:** Two‑triad hybrids correspond to **facet intersections** under some lens; use as candidates for Ennead completion.

---

## Witnesses & Metrics (for later testing)
- ΔU/tick, **C*** roll‑up, Weyl coverage index, MirrorPass rate, polarity conflicts, E‑DBSU growth, CSC yield.
- Lens rotation effectiveness: reward curve per lens; demotion reasons; stability under ContextFlip.

---

## Risks & Mitigations (Review‑Only)
- **Over‑rotation:** Use bandit scheduler + cool‑down; stickiness to high‑reward lenses.
- **Lens redundancy:** Diversity regularizers; prune lenses with near‑identical reward/trace profiles.
- **State bloat:** Dedup by codebook hash; TTL + drift checks; cap CSCs per topic.

---

## Next Review‑Only Derivations
- Lens interface sketch (constraints/orientation/refutation hooks) with harness for Gate‑3‑6‑9.
- Invariant Engine outline (two‑variant → invariant + margin κ + counter‑predicates).
- CSC Builder flow (inputs, outputs, witness set) with Trails.
- Static‑shape prepass scoring rubric (symmetry impact, ΔU/time).

