# UVIBS Alpha‑Draft — Full Claim & Exhaustive Test Suite (v0.1)

> **Status:** Alpha draft; provisional and fully contestable. All placements are soft. Every assertion below is intended to be **tested, falsified, or refined**. Where algebraic need contradicts text, **algebra wins**.

---

## 0. Purpose & Scope
This document states a **complete, disputable claim** about the UVIBS system and defines an **exacting test suite** covering every constituent in Buckets 1–8:
1) Primitives & Base Operators; 2) Windows & Rests; 3) Braids & Multi‑Body Operators; 4) Superpermutations & Route Patterns; 5) Entropy & Reconciliation; 6) Governance & Projections; 7) Windows, Cycles & Braid Structure; 8) Threads & Multi‑Body Braids.

It includes all algebraic objects and rules currently defined in the working notes and canvases (space, quadratic forms, modular constraints, CRT rules, entropy, Crooks, shells, Alena modulation, PAL/MIRROR, Mirror‑CRT, AMP repair, expansion lanes, geodesic/geometry aides, taxicab/Ramanujan links). Each subsection lists **claims**, **formal definitions**, **pass/fail criteria**, **null hypotheses**, and **logging specifications**.

---

## 1. Core Space & Notation (Bucket 1)

### 1.1 Ambient Space
- Let \(D = 8k\) with default \(D=80\).
- State vectors \(v \in \mathbb{Z}^D\).
- Decompose \(v\) into 10 blocks of 8: \(v = [v^{(1)},\dots,v^{(10)}],\ v^{(b)}\in\mathbb{Z}^8\).

### 1.2 Geometry
- **E8 blocks:** each block is evaluated with the **Cartan Gram** \(G_{E8}\); define blockwise quadratic form
  \[ Q_b(v) = (v^{(b)})^T G_{E8} v^{(b)}. \]
- **Global form:** \(Q(v) = \sum_{b=1}^{10} Q_b(v).\)

### 1.3 Modular Sums
- Block sum \(S_b(v) = \sum_{i=1}^{8} v^{(b)}_i\), global sum \(S(v) = \sum_{i=1}^{D} v_i\).

### 1.4 Chinese Remainder Theorem (CRT)
- For residues \(r_i \equiv x\ (\bmod\ m_i)\) with pairwise coprime \(m_i\), CRT gives a unique class modulo \(M=\prod m_i\).
- **Fail‑fast rule:** if \(\gcd(m_i,m_j)\nmid (r_i-r_j)\) for any pair, **jam** (incompatible residues).

**Logging keys:** `block_Q_mod4_pass`, `block_sum_mod8`, `global_sum_mod8`, `crt_status`.

**Null hypothesis H0‑1:** Lifting into \(\mathbb{Z}^{80}\) with E8 blocks offers **no** statistical advantage in pass/repair rates vs. any other 8k lattice.  
**Falsifier:** Significant, reproducible improvement (α=0.01) in strict W80 acceptance and AMP budgets under E8 vs. ablated Gram.

---

## 2. Primitive Operators — Digits 1–8 (Bucket 1 & 2)

### 2.1 Operator Semantics
- **1 (Primitive/Driver):** inserts a *rest/repair tick* and forces parity routing toward W4; may release cached entropy.
- **2 (Dual/Line):** enforces MIRROR/PAL checks and enables CRT pairing; aims at codec joins (10↔16 → 80).
- **3 (Triad/Stabilize):** opens **7/72** lane (\(\Sigma v \equiv 0\ \bmod 9\) & \(Q \equiv 0\ \bmod 7\)); braid‑open.
- **4 (First Rest):** **W4**: \(S(v) \equiv 0 \ (\bmod 4)\).
- **5 (Pentadic/Braid bundle):** realizes bundles of triads (hypothesis: 5×3 triads) and introduces φ‑guided modulation.
- **6 (Triad aggregation rest):** stabilizes compounded triads; tends toward organization.
- **7 (Isotropy/Cache):** odd‑prime isotropy; cache channel and re‑entry bias.
- **8 (Octadic rest):** **W80**: \(S(v)\equiv0\ (\bmod 8)\) and \(Q(v)\equiv0\ (\bmod 4)\) (global or strict per block).

### 2.2 Action vs Organization Law (trend, not absolute)
- **Odd** digits → action/expansion bias; **Even** digits → organization/closure bias; **exceptions** allowed at “entropy handshake” steps.

**Pass criteria:** Operator sequences map to legal window transitions without violating rests/gates; see §3–4.

**Null hypothesis H0‑2:** The odd/even action‑organization correlation is random.  
**Falsifier:** Odd steps show ≥\(\delta\) higher expansion/repair invocation; even steps show ≥\(\delta\) higher closure/organization (effect size + CI).

---

## 3. Windows & Rests (Bucket 2)

### 3.1 Definitions
- **W4 (Parity rest):** \(S(v) \equiv 0\ (\bmod 4)\).
- **W80 (Codec rest):** \(S(v) \equiv 0\ (\bmod 8),\ Q(v) \equiv 0\ (\bmod 4)\).  
  *Global:* evaluate \(Q\) once; *Strict:* require \(Q_b\equiv0\ (\bmod 4)\) for all 10 blocks.
- **Expansion windows Wexp(p,\nu):** \(Q(v) \equiv 0\ (\bmod p),\ S(v) \equiv 0\ (\bmod \nu)\).  
  Canonical lanes: **7/72** (p=7, ν=9 with optional LCM hook 72), **φ‑lane 11/40** (p=11, ν=10 with LCM hook 40).

### 3.2 Forcing Rules
- All legal routes **begin** at W4 and **close** at W80; expansions are optional but must re‑enter and close.
- If **strict** W80 jams, attempt Wexp; if multiple expansions used, they must compose without violating rests.

**Logging keys:** `W4_pass`, `W80_mode`, `W80_pass`, `Wexp_used`, `reentry_success`, `jam_type`.

**Null hypothesis H0‑3:** Expansion lanes do **not** improve strict W80 reconciliation rates beyond chance.  
**Falsifier:** Statistically significant uplift in re‑entry success and ΔS monotonicity when 7/72 or 11/40 used.

---

## 4. Gates & Symmetries (Bucket 4)

### 4.1 PAL, MIRROR, Mirror‑CRT
- **PAL\_b:** palindrome in base \(b\) (string‑level).  
  *Claim:* “Mid‑solve free embedding”: once the midpoint is validated, the reverse half is logically implied.
- **MIRROR(m,b):** \(x \equiv \text{rev}_b(x)\ (\bmod\ m)\).
- **Mirror‑CRT:** enforce MIRROR over \(\{(m_i,b_i)\}\), then CRT on congruences.

### 4.2 Palindrome Advantage (Claim)
- PAL+MIRROR increases acceptance probability and reduces repair budgets in strict regimes.

**Pass criteria:** PAL/MIRROR prefiltered routes exhibit lower AMP steps and higher W80 strict pass rates vs. random.

**Null hypothesis H0‑4:** Palindromic/mirror bias yields no measurable benefit.  
**Falsifier:** Effect size ≥\(\epsilon\) with p<0.01 over ≥N trials.

---

## 5. Superpermutations & Route Patterns (Bucket 4)

### 5.1 Superpermutation Basics
- A length‑minimal word containing every permutation of N symbols as contiguous substrings.  
  For N=3: **`123121321`** (palindromic). All 3! = 6 permutations appear once.

### 5.2 UVIBS Mapping
- Substrings ↔ windows; overlaps ↔ CRT joins; palindromicity ↔ PAL/MIRROR friendliness; full traversal ↔ braid word.

**Pass criteria:** Parsing a superpermutation yields a sequence of legal window transitions, with entropy ledger closed (ΔS_total ≥ 0).

**Null hypothesis H0‑5:** Minimal superpermutations confer no legality/entropy advantage vs. random permutations same length.

---

## 6. Braids & Multi‑Body Operators (Bucket 3)

### 6.1 Braid Formalism
- Threads \(t_i\) (see §8) intertwine; crossings modeled by braid generators \(\sigma_i\).  
- **Crossing legality:** only if CRT residues are compatible and window constraints satisfied.

### 6.2 Digit–Braid Operators (working semantics)
- 3 → braid‑open; 5 → braid‑bundle (hypothesis: 5× triads); 7 → cache/reconnect; 8 → braid‑close.

**Pass criteria:** Braid words translated from digit strings pass rests, gates, and governance; ΔS across crossings aggregates to monotone closure.

**Null hypothesis H0‑6:** Braid mapping is cosmetic; no reproducible legality/entropy influence vs. unbraided sequences.

---

## 7. Entropy, Capacity & Reversibility (Bucket 5)

### 7.1 Weights & Capacity
- **Shell weights** by cycles: triad (3), pentad (5), octad (8).  
- **Alena modulation (24‑slot φ‑cosine):** for slot k,
  \[ w_k = w_{\text{shell}} \cdot \cos\!\left(\frac{2\pi k}{24} + \varphi\right). \]
- **Capacity:** \(\mu = \sum_i w_i\).
- **Entropy:** \(S = \ln \mu\).

### 7.2 Entropy Deltas & Crooks
- \(\Delta S = S_{B} - S_{A}\) across rests.  
- **Crooks:** \(P_{rev}/P_{fwd} = e^{-\Delta S}.\)

### 7.3 Monotone Reconciliation (Claim)
- Over any **closed window**, \(\sum \Delta S_i \ge 0\). Local negatives allowed if compensated by caches and later closure.

**Logging keys:** `mu`, `S`, `DeltaS`, `Crooks` per segment; `cache_debt` for negative ΔS tracked.

**Null hypothesis H0‑7:** ΔS trajectories are random; Crooks does not correlate with acceptance/reversibility.

---

## 8. Threads & Braids (Bucket 8)

### 8.1 Threads
- Each facetized subproblem becomes an 8‑digit window stream lifted to 80D; evaluated by gates & rests.

### 8.2 Braid Merge
- **Input:** thread residues \(\{(r_i,m_i)\}\).  
- **Merge:** CRT if compatible; else expansion; else BRAID_FAIL with full ledger.

**Pass criteria:** Multi‑thread merges converge (re‑entry to W80) with ΔS_total ≥ 0 and bounded AMP.

**Null hypothesis H0‑8:** Multi‑thread CRT merges do not outperform naïve concatenation in legality or entropy.

---

## 9. Governance & Projections (Bucket 6)

### 9.1 Monster Layer & Projections
- **24D projections:** modular (mod‑24), affine (e.g., \(5i+7\)), shifts (e.g., +12).  
- **Legal kernel K:** intersection of projection constraints.

### 9.2 Governance Tests
- **Per‑block parity:** \(S_b(v) \equiv 0\ (\bmod 4)\) ∀b.  
- **Global isotropy:** \(Q(v) \equiv 0\ (\bmod 7)\).

### 9.3 AMP Repair
- Find \(\delta v\) (prefer within a +8 DOF control block) so that \(v' = v+\delta v\in K\) with minimal \(|\delta v|\) and minimal added entropy.

**Pass criteria:** For random illegal \(v\), AMP converges within budget B to \(v'\in K\) with recorded repair cost.

**Null hypothesis H0‑9:** Governance layer adds only overhead; it does not improve legality stability or ΔS monotonicity.

---

## 10. Geometry‑in‑the‑Loop Aides (from prior notes)
- **Geodesic certificates:** small discrete curvature and blockwise \(Q_b\) mod‑4 constancy on accepted hops.
- **Weyl reflections:** reflect within E8 across a root \(\alpha\) to reduce parity violation, then retest W80.
- **Geometry‑aware split:** choose split points to improve subsequent mod‑8/9/10 sums and \(Q\) mod‑4 checks.

**Pass criteria:** Adding aides reduces jam/repair rates or ΔS debt at fixed acceptance.

**Null hypothesis H0‑10:** Geometry aides provide no measurable lift.

---

## 11. Special Numbers & Resonances

### 11.1 Taxicab Numbers
- e.g., **1729** = \(1^3+12^3 = 9^3+10^3\).  
**Claim:** Such integers mark **free embedding points** where distinct braid routes reconcile to a common rest with reduced ΔS.

**Test:** Construct threads encoded by the cube pairs, attempt CRT/braid merge, measure ΔS vs. matched non‑taxicab totals.

### 11.2 Ramanujan Links
- **Partition functions \(p(n)\)** and congruences mod 5, 7, 11.  
**Claim:** Expansion lanes (7/72, 11/40) align with these congruences; ΔS distributions cluster by partition classes.

**Test:** Histogram ΔS modulo {5,7,11} under controlled runs; evaluate clustering vs. null.

---

## 12. Global Routing Policies
- **Direct codec:** 4 → 80 (global) when unjammed.
- **Strict codec:** 4 → 7/72 (or 11/40) → 80 when per‑block W80 jams.
- **Reconciliation windows:** begin at rest, end at rest; ledger ΔS, Crooks, AMP cost; record failures (W4_FAIL, GOV_FAIL, EXP_FAIL, REENTRY_FAIL, BRAID_FAIL).

**Pass criteria:** Policy produces ≥99% closure under bounded retries or proves illegality with explicit clashes.

**Null hypothesis H0‑12:** Policy provides no advantage over unconstrained search.

---

## 13. Data Artifacts & Logging (exhaustive)

### 13.1 CSV/Parquet Schemas
- **phase_metrics.csv:** `run_id, step, window_id, route_id, S4, S80, Sexp, DeltaS_4_80, DeltaS_80_exp, DeltaS_total, Crooks, cache_debt, pal_flag, mirror_flag, crt_status, W4_pass, W80_mode, W80_pass, Wexp_used, reentry_success, amp_steps, amp_norm`.
- **direct_hits.csv:** `run_id, window_id, straight_4_to_80, Monster_legal, geodesic_ok`.
- **failures.csv:** `run_id, window_id, fail_type{W4,GOV,EXP,REENTRY,BRAID}, invariants_snapshot, residues_snapshot, DeltaS_segment, amp_steps_at_fail`.
- **geometry.csv:** `run_id, block_id, Qb_mod4_pass, curvature_proxy, weyl_reflect_used, split_positions`.
- **braids.csv:** `run_id, braid_word, length, crossings, DeltaS_crossings, cache_events, closure_success`.
- **governance.csv:** `run_id, projections_passed, kernel_membership, amp_steps, amp_energy, alena_phase, regime{baseline,alena,monster}`.

### 13.2 Reproducibility
- Seed, proposer config, strictness, lane toggles, digit biases (odd/even), φ‑modulation flag, geometry aides flag.

---

## 14. Acceptance Thresholds & Statistical Testing
- Define α=0.01 for primary effects (palindrome advantage, expansion uplift, braid legality, governance value‑add).
- Use paired runs (with/without feature) over ≥1000 windows per condition; report CI and effect sizes.
- **Multiple hypothesis control** (e.g., Benjamini–Hochberg) across families of tests.

---

## 15. Explicit Falsifiers (per family)
- **F1 (PAL/MIRROR):** No reduction in AMP steps or strict W80 pass rate vs. random → drop PAL advantage claim.
- **F2 (Expansion lanes):** No uplift in re‑entry or ΔS monotonicity → treat 7/72, 11/40 as optional heuristics only.
- **F3 (Governance):** AMP rarely converges or increases ΔS debt → governance layer re‑specification required.
- **F4 (Entropy law):** Closed windows frequently show ΔS_total < 0 → revise capacity model and Crooks usage.
- **F5 (Braid law):** Multi‑thread merges show no advantage vs. unbraided concatenations → restrict braid semantics.
- **F6 (Resonances):** No ΔS clustering by {5,7,11} or taxicab reconvergence → remove Ramanujan/taxicab hypotheses.

---

## 16. Worked Micro‑Example (N=3 superperm)
- Word: **123121321**.  
- Map: `1→REST_AND_GOV`, `2→DUAL_JOIN`, `3→TRIAD_STABILIZE`.  
- Windows: W4 → W80 (strict jam) → 7/72 → W80.  
- Expected: PAL pass, MIRROR pass, ΔS_total ≥ 0, Crooks consistent, AMP bounded.

**Artifact expectation:** entries in phase_metrics, braids, failures (0 if success), governance with kernel pass.

---

## 17. Implementation Hooks (for harness)
- **Controllers:** strict/global codec; expansion policy; retry budget; AMP max steps.
- **Proposers:** pal‑biased, mirror‑biased, random, adversarial.
- **A/B toggles:** PAL on/off, MIRROR on/off, 7/72 vs 11/40, φ‑mod on/off, geometry aides on/off, Monster regime on/off.

---

## 18. Meta‑Position
This is a **first principles alpha claim**: UVIBS is proposed as the **bare‑bones operational code** for multi‑body closure. Every section is disputable. Every test is designed to validate, refine, or reject parts of the whole. **No component is monolithic**; all must bend to algebraic evidence and empirical outcomes logged by the harness.

> **If a claim fails, we rewrite the claim.** If a test fails to discriminate, we strengthen the test. If an algebraic conflict appears, we repair the algebra and respecify the operators.

**Deliverables of this charter:** the **spec above**, the **test suite below**, and the **logging schemas** enabling fully auditable runs.

---

## 19. Comprehensive Test Suite Index (Checklist)

### Family A — Operators & Gates
- A1 Odd/Even action‑organization bias
- A2 PAL/MIRROR acceptance lift (strict mode)
- A3 Mirror‑CRT vs simple MIRROR

### Family B — Windows & Expansions
- B1 W4→W80 direct vs strict
- B2 7/72 re‑entry uplift
- B3 φ‑lane vs 7/72 comparative
- B4 Composite expansions (7/72 ∘ 11/40)

### Family C — Governance
- C1 AMP convergence distribution
- C2 Kernel pass rate vs ablated projections
- C3 Alena modulation effect on ΔS

### Family D — Entropy
- D1 ΔS monotonic closure frequency
- D2 Crooks correlation with reversibility
- D3 Cache‑debt persistence & repay

### Family E — Braids & Threads
- E1 Braid legality vs unbraided
- E2 5×3 triad hypothesis (5‑braid structure)
- E3 Taxicab embedding uplift (1729, …)
- E4 Three‑body toy dynamics with braid corrections

**Each test**: inputs, toggles, metrics, acceptance rule, falsifier, artifacts.

---

## 20. Closing
This canvas is the **exhaustive, disputable** specification for UVIBS in alpha. It is intended to be **run, audited, and revised**. Nothing here is sacred; everything is **meant to be tested**.

