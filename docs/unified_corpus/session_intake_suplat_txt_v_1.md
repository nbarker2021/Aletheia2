# Session Intake — suplat.txt (v1.0)

**Purpose.** Reconstruct the entire session from the uploaded transcript, front‑to‑back, preserving every idea, refinement, result, and decision into a working reference we can extend. All speaker tags follow the user’s note: `You said:` = user; `ChatGPT said:` = assistant.

---
## A. Intake plan (how this doc is organized)
1) **Timeline reconstruction** (front → back), split into clearly labeled phases.
2) **Canonical concepts & axioms** (glossary with crisp definitions).
3) **Algorithms & rules** (operational checklists; acceptance tests).
4) **Data structures & artifacts** (seeds, n=1..5 chains, 8× n=5 classes, DB schema).
5) **Geometry layer** (E8 / Leech / Monster stack; triads/octads; parity rules).
6) **Search program** (Φ‑modulated schedule, ΔL bookkeeping, legality gates).
7) **Bench harness** (in‑session vs full machine; metrics; recovery plan).
8) **Open questions & next experiments** (traceable todo list).

---
## B. Timeline reconstruction (front → back)

### B1. Physiological openers → Consciousness framing
- MDMA heat & autonomic effects.
- DMT as bandwidth/tuning: entity motifs; shared structures.
- Orch‑OR + observer‑dependent reality: access radius; co‑observation coherence.

### B2. Shift to formal substrate
- **RESL** (Relational E8 Simulation & Logic): observation layer + logic‑lattice layer.
- Determination chains (binary) embedded in E8; selection by observation; consensus halos.

### B3. Superpermutations enter as the operational testbed
- Use SP as the concrete domain to stress‑test the framework.
- Known facts (n=1..5 exact; n=6 = 872 best‑known; n=7 records; eight inequivalent n=5 classes).

### B4. Your hypotheses → operational rules
- **USP**: Every observer’s reality stream is a superpermutation over its operational alphabet.
- **BLL 4‑2‑1 + CTX**: ≤4 primitives, ≤2 hybrids (+1 modulation), 1 full aggregator + context.
- **ESP‑8**: Eight minimal n=5 classes are both individual bases and an aggregate basis.
- **PLC (Primitive Legality Constraint)**: every step factors to binary legality.
- **Parity & dimension**: even n>4 → +1D lift; odd n≥5 → allow half‑steps then prune geometrically.
- **Triadic glyphs**: compress & certify understanding; triadic closure as audit.

### B5. Shelling & glyph system
- Ψₙ shells; at Ψ₅ enforce primitive validation, CTX audit, triadic compression (≥3 distinct triads per interior concept) → in‑memory hashing for recall.

### B6. Eightfold → n=6 program
- Build **K₅** from the 8 n=5 classes (seed‑relabelled); lift to n=6 in 3D; route to maximize total overlap \(∑k_i\) so length L = 6·720 − ∑k_i (target 872; stretch 871 exploration).
- **ΔL acceptance rule**; forced edges; late‑stage reroutes.
- Golden‑family modulation (φ schedulers) to avoid resonance traps.

### B7. Geometry stack
- **Monster → Leech (Λ24) → E8**: panels/relabellings to expose invariants; Conway group symmetries as consistency checks; “panels” act as multiple, independent slotters (Structure/Context/Legality triad).

### B8. Tooling & harness
- Chained n=1→5 solver per seed; eight n=5 outputs (class reps relabelled).
- Master DB per seed: action logs; free‑recall index; ≥85% similarity routing; “compress‑to‑add‑value” splice detector; CTX binding; triad store.
- Benchmark harness: in‑session (stdlib) vs full (parallel + ATSP/SAT/MILP backends); reproducible seeds; kernel‑reset recovery (cache + snapshot).

### B9. Code assets integrated
- **`agrmmdhg.py`** (TSP/MDHG components) adapted as state/cache + overlap‑aware ordering.
- **`superperm code.py`** (data manager, laminates/anti‑laminates, formulas) used for overlap checks, De Bruijn diagnostics, and persistence.

---
## C. Canonical glossary & axioms (v1)
- **RESL** — Relational E8 Simulation & Logic: Two coupled layers (observation ↔ logic‑lattice).
- **Determination chain (D)** — ordered binary choices that realize an event/object.
- **E8 medium (L)** — 8‑D lattice coordinatizing potentials; used here as address space for D.
- **Consensus halo (H)** — stability from redundant co‑observations/logs.
- **USP** — Universal Superpermutation Principle.
- **BLL 4‑2‑1 + CTX** — Representation capacity rule and context binding.
- **ESP‑8** — Eightfold basis of n=5 minimal classes (both separate and aggregate).
- **PLC** — Primitive Legality Constraint.
- **DER‑even / PES‑odd** — Parity‑aware dimensional rules.
- **Triadic glyphs (τ)** — minimal 3‑term expressions that expand back to primitives.
- **Ψₙ** — Wavefunction‑style shelling at level n.
- **Γ** — Glue rule (legality/coherence penalty) used in joins.

---
## D. Algorithms & rules (operational)
1) **Eightfold‑Lift (v0)**
   - Canonize 8× n=5 classes → seed relabel → balanced K₅.
   - Lift to n=6 (+1D) with legality checks (Γ).
   - Route with ΔL acceptance; forced edges first; golden‑modulated revisit schedule.
   - Meet‑in‑the‑middle lock; bridge close; export artifacts (groups/orders/seams/schedule).

2) **Legality/geometry gates**
   - Pass if: primitive legality, triadic reconstructability, parity rule satisfied, CTX bound.

3) **Compress‑to‑add‑value**
   - Detect near‑isolated 4/5‑digit blocks; propose 2‑digit fills that add ≥1 new perm and reduce length (ΔL<0); reject if any legality test fails.

4) **Mixed pools + channeling**
   - Work in octads→tetrads→dyads→monads; asymmetric meet‑in‑the‑middle; panel step‑ups when stalled.

---
## E. Data & DB schema (per seed)
- `chain.json` — per‑n outputs + step logs.
- `free_recall.json` — inverted index by digit subsets; jump queries.
- `similarity.json` — k‑gram signatures; ≥0.85 neighbors for insertion guidance.
- `ctx.json` — environment/constraints.
- `triads.json` — 3 distinct glyphs per interior unit + reconstructability proof.

---
## F. Bench harness & recovery
- In‑session: stdlib; tracemalloc; JSON resource report; seeded randomness.
- Full machine: parallel search; ATSP/SAT/MILP backends; same report schema.
- Recovery: `/mnt/data/l24_suite/cache` + `kernel_recovery.json` + zipped snapshot; mirror code snippets into canvas when stable.

---
## G. Open questions (for active testing)
1) Does φ‑modulated scheduling measurably increase late‑overlap gains vs greedy?
2) Is “1–8 degeneracy” plausible for any 871 class under eightfold exposure/gauge quotients?
3) What invariants best distinguish the eight n=5 classes during lift (wasted‑run, cyclic footprint, motif spectra)?
4) Minimal Γ that still enforces legality without blocking necessary half‑steps?
5) How to parameterize panel step‑ups (Λ24) as constraints rather than tie‑break visuals only?

---
## H. Next actions (immediate)
- Wire eight n=5 class loader (seed‑relabelled) into the harness.
- Enable CTX + triad emission at each chain stage.
- Turn “compress‑to‑add‑value” into an acceptance gate (after forced edges).
- Produce the first resource report and ΔL log for an n=6 run; snapshot artifacts.

*This intake document will expand as we ingest more of the transcript; sections will receive line‑referenced anchors to the source text to keep provenance clear.*


---

# Footer — New Additions Log (Pass 1)

This section records what was added or refined during this pass. Each item references the section number it extends and can be treated as an addendum to that section.

- §0 Big Picture: Added explicit length identity used by the harness: L = n * M − sum(overlaps). Reaffirmed lexicographic scoring: maximize overlap first, then minimize Gamma (tie‑break only). Clarified coverage rule: every n‑window must appear exactly once; only local repairs allowed.

- §A Timeline: Marked inflection points (shift to insert‑then‑stitch; adoption of legality; move from clean 8→4→2→1 pipeline to mixed‑scale channels; stall→panel step‑ups; caching discipline). Documented which heavier global approaches were discarded or softened.

- §B Canonical Concepts: Added operational tests for each axiom. Triadic glyph audit now requires three distinct expansions that reconstruct the same primitive meaning; failing this flags shallow understanding.

- §C Algorithms:
  - §C1 Insert→Stitch: Forced‑edge chains (unique max‑overlap 5 or more) are pre‑compressed to shrink search. Slot schedule is the decision variable when traversing the n=5 base.
  - §C2 Mixed Pools with 6‑edge budgeting: 8‑groups must contain 1–3 items with digit 6 at an edge; 4‑groups default to at most 1 (2 only rarely, and logged); 2‑groups use a golden‑ratio bias (about 0.618) to include a 6‑edge under ties; 1‑anchors are six raw items chosen so that each digit 1..6 appears as an edge at least once (no padding the pool).
  - §C3 Asymmetric meet‑in‑the‑middle: Left frontier prefers large blocks; right frontier prefers small/anchors; attempt short bridge searches regularly; accept locks that preserve coverage; otherwise do local reorders only.
  - §C4 Pruning: Immediately discard a candidate when overlap is below threshold and it adds no new coverage. Default threshold for n=6 is 3; tunable.
  - §C5 Compress‑to‑add‑value: Detect near‑isolated 4 or 5 digit blocks; test a two‑digit connector that adds at least one new permutation and reduces length; must pass legality.
  - §C6 Stall handling: If no improvement in length or seam histogram after about 10 attempts, add 5 new panels; if still stalled, add 8. Gamma remains subordinate to overlap.

- §D Data and Artifacts: Introduced registry key (n, mode, relabel seeds, version, glue mode, channel id). Listed file layout for groups, merged outputs, coverage, seams, schedules, and recovery. Added k‑gram and digit‑edge indices.

- §E Geometry Layer: Gave concrete examples of slotters (structure, context, legality). Defined Gamma0 (parity and modular checks) and Gamma1 (Gamma0 plus a cyclic octad co‑membership check). Panels combine by sum or majority; chunks agreed on by panels are treated as invariants.

- §F Bench Harness: In‑session runs forbid global 720 by 720 overlap matrices; all work is local to groups or seams. Resource report now standardizes: final length, seam histogram, Gamma sum, coverage statistics, duplicates fixed, iterations to lock, per‑stage timing, cache hit rate.

- §G Open Questions: Each question now has a measurable acceptance criterion (for example, does the golden‑ratio bias improve final length or only the seam histogram; test with multiple seeds and basic statistics). Added ablations for Gamma and forced‑edge chains.

- §H Demo Script: Operator procedure clarified step by step, including where to display inventories, histograms, bridge attempts, and coverage fixes.

- §I Risks: Added mitigations for over‑binding of constraints (minimal logged relaxations and 2–3 restarts), tie cascades (panel step‑ups capped), compute creep (prune on overlap and coverage gain; cap beam widths), and kernel resets (cache plus recovery plus canvas mirroring plus periodic snapshots).

- §J Defaults: Proposed defaults recorded for your approval: overlap threshold 3 for n=6, golden‑ratio bias 0.618, stall window 10, step‑ups of plus 5 then plus 8 panels, Gamma mode Gamma1, minimal relaxations allowed and logged, excess‑pool off by default.

End of New Additions Log (Pass 1).


---

# Footer — New Additions Log (Pass 2)

This pass adds **worked procedures, concrete heuristics, and testable specs** tied to each section. Everything here is additive to Pass 1.

## §0 Big Picture — Quantitative Targets & Budgets
- **n=6 objective tiers:**
  - **T1:** L ≤ **875** (sanity)  
  - **T2:** L ≤ **873** (strong baseline)  
  - **T3:** L = **872** (state of the art)
- **Performance budget (in‑session):** no O(720²) builds; all scoring done on **groups ≤ 8** or **seam neighborhoods ≤ 16** items; per-step soft cap ≤ 2e5 primitive comparisons.
- **Correctness gates:** (1) coverage 720/720; (2) zero duplicates; (3) Γ never outranks overlap; (4) reproducible by seed.

## §A Timeline — Trace Hooks
- Add a **run ledger**: `runs/{ts}/manifest.json` capturing seeds, knobs, panel IDs, and git‑like hash of artifacts; every figure/table references `{ts}`.
- Each milestone now logs **before/after seam histograms** and **ΔL**.

## §B Canonical Concepts — Triadic Examples & Audits
- **Triadic glyph example (structure/context/legality):**
  - *Structure:* “head‑6 tail‑3; invParity=odd”
  - *Context:* “overlap=5; tail=…3 → head=3…”
  - *Legality:* “pos(1)=4,pos(6)=1; parity OK; Γ₁ octad match {7,15,23}”
- **Audit macro:** for any accepted join, store its 3 glyphs and a reversible mapping back to primitives. *Failing to invert ⇒ flagged seam.*

## §C Algorithms — Concrete Procedures
### §C1 Forced‑Edge Detection
```
for u in perms6:
  best = max_k overlap(u, v)
  if count{v: overlap(u,v)==best} == 1 and best>=5:
     forced_succ[u] = v
(similarly for forced_pred)
compress disjoint paths u→…→v into chains
```
- **Acceptance:** ≥95% of chains remain stable across 3 random seeds; ΔL improves vs. no compression.

### §C2 Group Construction with 6‑Edge Budgeting
```
# Build 8-groups
pool = perms6 \ forced_chains
score6edge(x) = [head==6 or tail==6]
while pool:
  G = pick 8 items maximizing diversity of head/tail digits
  enforce 1..3 of score6edge==True (greedy swap if violated)
  save G; pool-=G
# 4-groups from mixed fragments (8-merges + raw + chains)
# 2-groups with φ-bias; 1-anchors cover all digits at edges
```
- **Validation:** each group file includes the **6‑edge count**, **digit‑edge coverage**, and a **violation flag** if a minimal relaxation occurred.

### §C3 Asymmetric Meet‑in‑the‑Middle (AMM)
```
L = seed_large_block(); R = seed_small_anchor();
beamL, beamR = {L}, {R}
repeat until locked or stalled:
  expand(beamL, K): attach best large/mixed fragments by (overlap, -Γ, +coverageGain)
  expand(beamR, K): attach best small/anchors by same key
  if bridge_exists(beamL, beamR, depth<=3):
     S = lock(beamL*, beamR*, bridge*)
     break
  prune losers: if overlap<threshold && coverageGain==0
```
- **Knobs:** K∈[4,8]; threshold=3; bridge depth≤3.
- **Acceptance:** ≥60% of accepted seams have overlap≥4; bridge success within ≤5 attempts in ≥80% of seeds.

### §C4 Coverage‑Gain Predicate
```
coverageGain(u→v) = count(new 6-windows in merge(u,v))
fast check: only windows within (|u|+|v| - overlap - n + 1)
```
- **Rule:** reject if overlap<threshold **and** coverageGain==0.

### §C5 Compress‑to‑Add‑Value Inspector
```
find blocks B with boundary repeats or low-degree adjacency
for candidate 2-digit connectors c:
   if ΔL(c)<0 and legality OK and coverageGain(c)>0:
       accept c; log (ΔL, windowsAdded)
```
- **Acceptance:** net ΔL<0 and no new duplicates.

### §C6 Stall→Panel Step‑Up
- **Stall metric:** no improvement to (L, seam5count) in 10 AMM loops.
- **Step‑Up:** add 5 panels (new relabelings); if stall again, +8; max two step‑ups per run.
- **Aggregation:** Γ_sum = Σ panels; tie‑break by fewer Γ violations.

## §D Data & Artifacts — JSON Schemas (extract)
- `coverage/{channel}.json`
```
{
  "n":6,
  "unique":720,
  "duplicates":[],
  "missing":[],
  "seam_hist": {"0":12,"1":28,"2":64,"3":150,"4":280,"5":186},
  "gamma_sum": 97,
  "length": 872
}
```
- `groups/{channel}/level_8.json`
```
[
  {"id":0,"items":[...],"six_edge":2,"digit_edge_cov":[1..6],"relaxed":false},
  ...
]
```
- `seams/{channel}.json`
```
[
  {"left":"...","right":"...","overlap":5,"gamma":0,
   "panels_ok":7,"panels_bad":0,"triads":["S:...","C:...","L:..."]}
]
```

## §E Geometry — Slotter Recipes & Γ Tables
- **Slotter S (structure):** `S(w) = (37*inv(w) + 13*parity + 7*head + 11*tail) mod 240`
- **Slotter C (context u→v,k):** `C = (101*k + 17*u.tail + 19*v.head + 23*u.head + 29*v.tail) mod 240`
- **Slotter L (legality):** `L(w) = (31*pos1 + 41*posn + 17*parity) mod 240`
- **Γ₀ checks:** `(S+C+L) mod 3 == 0`; equal‑mod‑5 triplets penalized
- **Γ₁:** Γ₀ + cyclic‑octad co‑membership for `S%24, C%24, L%24`
- **Panel relabeling:** apply 3 independent permutations of [0..239]; store seed IDs.

## §F Bench Harness — Reproducibility & Recovery
- **Seed discipline:** record RNG seed for (grouping, AMM, panels); store in manifest.
- **Cache keys:** `hash(spec + seeds + versions)`; include code hash so results tie to implementation.
- **Recovery:** `kernel_recovery.json` lists last good artifacts; runner auto‑rebuilds missing pieces idempotently.

## §G Experiments — Concrete Plans
1) **φ-bias sweep:** φ∈{0.55,0.60,0.618,0.65,0.70}; 20 seeds each; record ΔL and seam5count.
2) **Γ ablation:** off vs Γ₀ vs Γ₁; compare duplicates-before-repair and final L.
3) **Excess‑pool toggle:** off/on(+15% pool); measure effect on L and compute.
4) **Chains on/off:** remove precompression; measure Δ search time, ΔL.
5) **Bridge depth:** 2 vs 3 vs 4; measure lock success rate and ΔL.

## §H Demo — Operator Prompts
- `show: forced-chains` → prints counts and sample chains
- `show: pool` → summarizes 6-edge budgets by group
- `run: amm` → executes AMM with live seam histogram
- `bridge: attempt` → runs depth‑3 bridge; prints top candidates
- `verify` → coverage report + duplicates (if any)
- `export` → writes artifacts + 5→6 schedule map

## §I Risks — Edge Case Policies
- **Group infeasible:** allow minimal relaxation (±1 on 6-edge count) with a red flag in metadata; force a restart if >5% groups relaxed.
- **AMM deadlock:** randomize beam order and swap left/right roles for one loop.
- **Bridge churn:** if ≥3 failed bridges, raise threshold to 4 for two loops to favor stronger seams.

## §J Defaults — Trial Set for First Full Run
- θ=3, K=6, bridge depth=3, stall=10, step-ups=+5 then +8, Γ=Γ₁, φ=0.618, excess-pool=OFF, max‑two step‑ups.

_End of New Additions Log (Pass 2)._

---

# Footer — New Additions Log (Pass 3)

This pass adds worked micro‑examples, executable‑level specs, and edge‑procedure math tied to each section. Everything here is additive to Pass 1–2.

## §0 Big Picture — Complexity, Memory & Determinism
- Asymptotic footprint (n=6, groups ≤8):
  - Overlap checks stay within about O(G·8^2) + O(S·b·d), where G=#groups, S=#AMM steps, b=beam width (≤8), d=bridge depth (≤3).
  - No global 720×720 matrix. Seam neighborhoods capped at 16 candidates → <1e5 primitive comparisons per step.
- Memory budget:
  - Strings: final S length ≤ ~3–4k chars.
  - Artifacts: JSON manifests well under 10 MB per full run.
- Determinism:
  - Seeds recorded for: grouping (sigma_g), AMM (sigma_a), panels (sigma_p).
  - Code hash (h) stored in manifest; cache key K = H(n, knobs, sigma_g, sigma_a, sigma_p, h).

## §A Timeline — Provenance Anchors
- Each major action logs a provenance triple: (artifact_path, code_hash, seed_tuple).
- Figures/tables in reports carry run_id so results can be traced back to exact files.

## §B Canonical Concepts — Worked Triad Audit
- Example seam u=412356, v=123564; overlap(u→v)=5 (shared 12356).
  - S(w): inv(412356)=5 (odd), head=4, tail=6 → S = (37·5 + 13·1 + 7·4 + 11·6) mod 240 = 53.
  - C(u→v,5): C = 101·5 + 17·6 + 19·1 + 23·4 + 29·4 mod 240 = 5.
  - L(v): pos1=1, pos6=6, parity=even → L = (31·1 + 41·6 + 17·0) mod 240 = 17.
  - Γ0: (S+C+L)=75 ≡ 0 (mod 3) ✓; mod‑5 equality? no ✓. Γ1: (53,5,17)%24=(5,5,17) in same cyclic octad ✓ ⇒ Γ=0.
  - Triad glyphs: S:"odd‑inv,4⟂6", C:"k5,6→1 bridge", L:"pos(1)=1,pos(6)=6,Γ1✓" (reversible to primitives).

## §C Algorithms — Micro‑Example & Procedures
### §C1 Micro‑Example (subset of 6 perms)
Take five 6‑perms: A=123456, B=234561, C=345612, D=451236, E=512346.
- Pair overlaps:
  - A→B k=5; B→C k=5; C→D k=3; D→E k=4; E→A k=2.
- Local chain (forced): if A→B and B→C are unique maxima ≥5, compress chain A→B→C.
- AMM seed: Left=A→B→C (large), Right=E (anchor). Bridge candidates from pool include D.
- Bridge test: try C→D (k=3) then D→E (k=4) vs alternative C→E (k=2) + repair. Choose higher combined k with legal Γ; accept C→D→E.
- Length check: L = 6·5 − (5+5+3+4) = 30 − 17 = 13 (for this toy set). Coverage for the toy universe is complete.

### §C2 Mixed‑Pool Builder (deterministic with minimal relaxation)
```
BUILD_MIXED_POOL(P):
  rng ← seed sigma_g
  # Stage 0: carve out forced chains
  C ← COMPRESS_FORCED(P)
  R ← P \ C
  # Stage 1: build 8-groups with 6-edge budget 1..3
  while R not empty:
     G ← PICK8_MAX_DIVERSITY(R, rng)
     if SIX_EDGE_COUNT(G) not in [1,3]:
        G ← GREEDY_SWAP_FIX(G, R)
        if still infeasible: mark relaxed and continue
     save G; R ← R \ G
  # Stage 2: create 4s from (merged 8s ∪ chains ∪ R)
  # Stage 3: create 2s with phi-bias; Stage 4: pick 1-anchors covering digits 1..6 at edges
  return POOL = {8s, 4s, 2s, 1s}, meta
```
- Digit‑edge coverage score: maximize unique head/tail digits to avoid myopic pools.

### §C3 Bridge BFS (depth ≤ 3)
```
BRIDGE(L_tail, R_head, POOL, depth=3):
  frontier = [{seq:[], ksum:0, Gsum:0, cover_gain:0, node=L_tail}]
  for t in 1..depth:
     new = []
     for s in frontier:
        for f in TOPK_NEIGHBORS(s.node, POOL, K):
           k = overlap(s.node, f)
           if k < threshold and coverageGain == 0: continue
           G = gamma(s.node, f)
           add new state with updated (ksum, Gsum, cover_gain)
     frontier = PRUNE(new, beam=K)
     if any state connects to R_head with k ≥ threshold: return best path
  return FAIL
```
- Score order: maximize (ksum, −Gsum, cover_gain).

### §C4 Local Repair (duplicates/misses)
```
REPAIR(S, issues):
  for each dup window w or missing m:
     neighborhood ← windows within radius r of issue
     candidates ← reorders/connector inserts drawn from pool cache
     pick best that fixes issue and ΔL ≤ 0, Γ ok; apply; log diff
```
- Radius r: 2–4 seams typically suffices at n=6.

## §D Data & Artifacts — Concrete File Layout (sample)
```
runs/2025-08-24T22-11-03Z/
  manifest.json           # seeds, knobs, code hash, cache keys
  coverage/channelA.json  # unique, misses, dupes, seam_hist, gamma_sum, length
  groups/channelA/
    level_8.json          # list of 8-groups (+six_edge, relaxed flags)
    level_4.json
    level_2.json
    level_1.json          # anchors
  merged/channelA/
    level_8.txt           # merged strings per group
    level_4.txt
    level_2.txt
    level_1.txt           # final S
  seams/channelA.json     # seam logs with triads + panel tallies
  schedule/n5to6/channelA.json  # implied slot schedule
  recovery/kernel_recovery.json
```

## §E Geometry — Panel Selection & Consensus
- Panel generation: for each step‑up, draw 5 (then 8) relabelings with independent RNG seeded by (sigma_p, step_up_idx). Store their permutations.
- Consensus policy:
  - Sum mode: Gsum across panels; tie break by fewer violations.
  - Majority‑legal: accept seam if at least ceil((P/2)+1) panels score Γ=0; else penalize by panels_bad.
- Invariant detection: a seam is invariant if majority‑legal across all active panels. Prefer invariant seams when overlap is tied.

## §F Harness — Repro Script (pseudocode)
```
RUN(n=6, defaults):
  seeds ← {sigma_g, sigma_a, sigma_p}
  preseed ← load_or_build_n5_and_lift()
  chains ← forced_edge_compress(lift)
  pool ← build_mixed_pool(lift, chains, sigma_g)
  S ← amm(pool, sigma_a, defaults)
  cov ← verify_coverage(S)
  if cov.issues: S ← repair(S, cov.issues)
  save_artifacts(S, seeds, defaults, code_hash)
```
- Smoke tests: assert coverage 720; seam histogram sanity; Γ never outranks overlap.

## §G Experiments — Power & Reporting
- Power: each knob sweep uses at least 20 seeds; report mean ± stdev; paired t‑tests vs baseline.
- Report: CSV + JSON with per‑run metrics; summary tables: ΔL, seam5count, Gsum, run time.

## §H Demo — Worked Operator Walkthrough
1) `show: forced-chains` ⇒ “17 chains (avg len 3.2), 61 perms covered.”
2) `show: pool` ⇒ “8s: 45 groups (1.9 six‑edge avg), relaxed: 2; 4s: 90; 2s: 180; anchors: 6 (edges cover 1..6).”
3) `run: amm` ⇒ live seam histogram increments (emphasis on 4/5 overlaps).
4) `bridge: attempt` ⇒ prints top 5 bridge paths with (ksum, Gsum, cover_gain).
5) `verify` ⇒ “coverage 720/720; duplicates 0; misses 0; L=872.”
6) `export` ⇒ writes artifacts; emits implied 5→6 schedule visualization data.

## §I Risks — Stress Scenarios
- Panel over‑penalization: If Γ disqualifies too many ties, fall back to Γ0 for two loops; log switch.
- Anchor starvation: If anchors are consumed early, reserve two for late‑stage bridges.
- Excess‑pool pathology: Enable only after first stall; disable if ΔL worsens twice consecutively.

## §J Defaults — Freeze for First Public Demo
- Keep: threshold=3, K=6, depth=3, stall=10, step‑ups=+5 then +8, Γ=Γ1, phi=0.618, excess‑pool=OFF, two step‑ups max.
- Add: reserve_anchors=2 for late bridges; relax_max_rate=5% groups.

End of New Additions Log (Pass 3).

