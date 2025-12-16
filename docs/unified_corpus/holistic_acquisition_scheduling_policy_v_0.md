# Holistic Acquisition & Scheduling Policy — v0.1

**Guiding Principle: “Uniquely Holistic.”** Every decision must optimize what is best for the *whole system*, not just a local metric. Safety and governance gates are lexicographic hard constraints; within the admissible region, choices maximize a single holistic objective that balances exploration, gap reduction, and task completion under compute budgets.

> This canvas extends the prior spec with a concrete, testable policy for coupling constraints and objectives, acquisition scheduling, and integration with our AutoSwitch/AutoTopo surrogate stack. No numeric constants are invented; all thresholds remain policy knobs to be learned.

---

## 1) Holistic Objective (symbolic, no numbers fixed)
Define a **Holistic System Utility** \(U_H\) per tick as a weighted, auditable composition of:
- **Frontier expansion:** expected information gain about optimal options and layout consensus (geometry deltas: coverage, multiplicity, bridge density).
- **Gap load reduction:** decrease in the weighted sum of active gaps (calibration drift, residuals, rung misses, layout disagreement, contradiction rate, sensitivity ridges).
- **Task completion credit:** progress along user‑requested dependency DAGs (successful SNAP deltas that close open tasks within TTL).
- **Governance health margin:** distance from safety/fairness/legal thresholds (positive margin is rewarded; violations remain hard‑blocked by helix rungs).
- **Master progress:** consolidation improvements (closure scores, glyph coverage, Master digest stability).

**Lexicographic rule:** If any helix rung fails (Legality, Safety, Fairness, Provenance completeness, Observer repeatability), the action is inadmissible regardless of its \(U_H\).

All weights are exposed as policy parameters; they default to *learned via replay tests* (see §7).

---

## 2) Coupling Policy — “Coupled by Default, Opportunistically Decoupled”
- **Default:** Treat objective and constraints as **coupled**; propose and evaluate **joint measurements** (utility + safety rungs) for candidate options.
- **Opportunistic decoupling:** When instrumentation supports *constraint‑only* or *objective‑only* probes and the **expected global improvement per cost** (EGI/Cost) is higher than any joint probe, schedule the decoupled measurement.
- **Safety front‑end:** The safe set is computed before scoring (using current surrogates). Outside the safe set is never selected.

This policy satisfies “uniquely holistic” by always choosing the action whose *system‑wide EGI/Cost* is highest, whether that action is joint or decoupled.

---

## 3) Action Space and Scoring
Let \(\mathcal{A}_t\) be the actions available at tick \(t\):
- **Joint option eval:** evaluate \(o\in\mathcal{S}_t\) for utility and all measurable constraints.
- **Constraint‑only eval:** measure a specific rung signal (e.g., calibration, safety risk) at targeted contexts/options.
- **Objective‑only eval:** (when constraints are already well‑bounded) measure utility at selected contexts.
- **Calibration refresh:** targeted data for models with drift alerts.

For each action \(a\), compute **EGI(a)** = expected \(\Delta U_H\) from taking \(a\). Select a batch by maximizing \(\sum_{a\in B}\text{EGI}(a)\) subject to the main‑tick budget and risk ceilings (BwK‑style knapsack with safety). Sub‑tick budgets prioritize gap‑load reduction and task completion (§6).

---

## 4) Surrogate Stack Integration (AutoSwitch & AutoTopo)
- **AutoSwitch (Section 28):** Contextual Linear pre‑ranks; top‑M promoted to GP (or RFF) when non‑linearity, calibration error, residual clustering, or budget allow.
- **AutoTopo (Section 29):** Separate models in early/low‑complexity shells, Multi‑Output GP (ICM/LMC) in advanced shells or when cross‑signal correlation is high. Fallback to independent GPs with shared kernels if multi‑output numerics fail.
- **Safety models:** per‑rung surrogates share feature maps with the utility model; joint feasibility uses conservative bounds if multi‑output is disabled.

---

## 5) Holistic Acquisition Scheduler (HAS) — Algorithm Sketch
1. **Safe set update:** Build \(\mathcal{S}_t\) using current safety surrogates and helix rungs.
2. **Candidate generation:**
   - Joint evaluations for top options from Contextual Linear ranking.
   - Constraint‑only probes where uncertainty is high and gap weights are large.
   - Calibration refresh for views with drift alerts.
3. **Scoring:** Estimate EGI/Cost for each candidate using IDS/KG‑style utilities and gap‑reduction forecasts; apply risk ceilings.
4. **Budgeted selection:** Solve a BwK batch selection for the main tick; commit via SAP (Sentinel check → Arbiter warrant → Porter write → Sentinel post‑check).
5. **Sub‑ticks:** Execute a prioritized gap‑reduction and task‑completion schedule until the sub‑tick budget is consumed or gap load is zero and tasks are complete.
6. **Logging:** Record action type, EGI estimate and realized \(\Delta U_H\), safe‑set margins, and surrogate snapshots in SNAP; update per‑shell indices and Master digest.

---

## 6) Data Structures and API Stubs
- **SNAP additions:**
  - `acquisition: { action_type, candidates: [...], egi_estimate, expected_cost, safe_margins, chosen_surrogates }`
  - `holistic: { dU_frontier, dU_gap, dU_tasks, dU_governance, dU_master, U_H_before, U_H_after }`
- **SchedulerPolicy additions:**
```
HolisticPolicy {
  objective_components: [frontier, gaps, tasks, governance, master],
  weights: { frontier, gaps, tasks, governance, master },
  decouple_allowed: true|false,
  egi_method: { ids|kg|pesc },
  batch_solver: { bwk_primal_dual|ucb_knapsack },
  risk_ceilings: { contradiction, cal_error, veto_rate },
  logging_level: { summary|detailed }
}
```

---

## 7) Test Plan (to learn thresholds/weights without inventing numbers)
- **Offline ledger replay:** compare AutoSwitch/Au toSwitch + AutoTopo under coupled‑only vs coupled+decoupled (opportunistic) policies; fit weights to maximize realized \(\Delta U_H\) while respecting safety ceilings.
- **Cross‑seed A/B:** vary shortlist size \(M\), decoupling flag, and BwK solver; measure info‑per‑cost, safety violations, bridge discovery, multiplicity realized, time‑to‑clear Sentinel stops.
- **Stress regimes:** high non‑linearity, high constraint uncertainty, and calibration drift; ensure STOP storms do not increase.

---

## 8) Open Decisions (recorded without fixing values)
1. Exact component weights for \(U_H\) and their learning schedule.
2. Formal EGI estimator choice (IDS vs KG vs PESC) by regime and compute budget.
3. Cost model granularity for candidate actions, including constraint‑only probes and calibration refresh.
4. Joint feasibility combination rule under multi‑output vs independent surrogates.
5. Default shortlist size \(M\) for GP promotion in AutoSwitch.
6. Stable logging schema for realized \(\Delta U_H\) and audit trails.


---

## 9) Context-Conditioned U_H ("Focus-Shifting Lenses")
Goal: Allow U_H to adapt in real time to context and relevance traits without inventing numbers, while keeping governance gates lexicographic.

### 9.1 Lenses and context
- Lenses: Governance, Context/Need, Observation/Limits (from prior spec), plus optional Domain lenses.
- Context features (c): shell index, effective dimension D_eff, layout disagreement, residuals, calibration drift, contradiction rate, bridge density, user intent tag, task TTL, and seed lineage stats.

### 9.2 Context-Conditioned Random Scalarization (CCR-Scalar)
- Maintain a distribution over scalarizations p_phi(w|c) where weights w mix the components of U_H: frontier, gaps, tasks, governance margin, master progress.
- Parameterizer phi: a small adapter (linear or shallow net) mapping context c to Dirichlet parameters for w.
- Each main tick, sample K weight vectors w^(k) ~ p_phi(.|c), compute candidate scores using IDS/KG/PESC under each w^(k), then solve BwK for a batch. Sub-ticks use the posterior mean of w for gap reduction and task completion ordering.

### 9.3 Preference & feedback loop
- Preference feedback: when available, collect pairwise or listwise judgments (dueling style) over executed options and update p_phi to tilt toward preferred scalarizations in the current context.
- Stability: apply a forgetting factor on phi and constrain KL divergence per tick to avoid thrash; fall back to a uniform Dirichlet when evidence is weak.

### 9.4 Safety coupling
Safety gates remain lexicographic and outside scalarization; CCR-Scalar only ranks among admissible actions. For constraint-only probes, temporarily tilt w toward gap and governance margin components.

### 9.5 API stubs
```
HolisticScalarizationPolicy {
  type: "CCR-Scalar",
  lenses: [governance, context, observation, domain?],
  context_features: [shell, D_eff, layout_disagree, residuals, cal_drift, contradiction, bridge_density, user_intent, ttl, seed_stats],
  dirichlet_adapter: { hidden: 0|32, reg: {...}, kl_cap: ... },
  samples_per_tick: K,
  feedback: { dueling: true|false, window: ... },
  forgetting: { decay_rate: ... }
}
```

---

## 10) Testing Harness for Context-Dependence (no invented constants)
1. Ledgered replay grid: define a grid over contexts (shells, D_eff, drift levels, disagreement levels). Replay past ticks with CCR-Scalar vs fixed weights; measure realized Delta U_H, safety ceilings, info-per-cost.
2. Preference A/B: inject synthetic pairwise preferences consistent with target scalarizations; verify the adapter recovers those under noise.
3. Non-stationarity stress: simulate context shifts (e.g., residual spikes, sudden disagreement). Evaluate stability (KL per tick), STOP storms, and closure scores.
4. Compute budget sweep: vary BwK budgets to study trade-offs in exploration vs gap reduction under CCR-Scalar.

---

## 11) Open Decisions (recorded)
1. Feature list c finalization for context adapter.
2. Choice of acquisition (IDS vs KG vs PESC) per regime and their mixing with CCR-Scalar.
3. Preferred preference signal format (pairwise, listwise, numeric ratings) and its frequency.
4. Forgetting and KL caps to balance adaptability vs stability.
5. Samples per tick K and shortlist size for GP promotion under context shifts.


---

## 12) Preference Instrumentation Policy — Staged by n & Context
**User directive (accepted):**
- **n ≤ 4:** use **Pairwise Duels** only.
- **n = 5–6:** introduce **Top‑k Rankings** in addition to duels.
- **n ≥ 7:** enable **Sliders** (lens weights) as a prior, while keeping duels and top‑k.

### 12.1 Rationale in this system
- Early stages (n ≤ 4) benefit from low‑burden, robust signals while geometry and calibration settle.
- Mid stages (n = 5–6) have richer option sets; rankings accelerate CCR‑Scalar learning with manageable user effort.
- Advanced stages (n ≥ 7) require context‑sensitive emphasis; sliders serve as a *prior* over lens weights, but do not override Sentinel gates.

### 12.2 Hysteresis and context coupling
- **Hysteresis:** require stage change confirmation across consecutive ticks to avoid oscillation.
- **Effective‑dimension guard:** if D_eff indicates higher complexity than n suggests, allow early promotion; if D_eff drops, delay demotion to avoid thrash.
- **Override triggers:** temporarily promote the stage when layout disagreement, calibration drift, or STOP storms exceed policy thresholds; return when cleared.

### 12.3 Operational mix per stage
- **Stage 0 (n ≤ 4):**
  - Preference events: pairwise duels only.
  - Sampling: interleave easy/hard pairs; maintain ~60–80% win‑rate band.
  - Update: Bradley‑Terry/Thurstone; CCR‑Scalar adapter nudged via duels.
- **Stage 1 (n = 5–6):**
  - Preference events: top‑k rankings at scheduled intervals (e.g., per layout cycle) + background duels.
  - Shortlist source: BwK pre‑batch (K_pre) from Section 28 AutoSwitch pre‑ranker.
  - Update: Plackett‑Luce/ListNet for rankings; dueling updates continue.
- **Stage 2 (n ≥ 7):**
  - Preference events: sliders act as **Dirichlet prior** over lens weights; apply decay and per‑tick KL cap.
  - Rankings and duels remain to validate or counteract slider drift.
  - Governance: sliders cannot move actions outside the safe set; Sentinel remains lexicographic.

### 12.4 Logging and audits (SNAP additions)
`preference_event: { stage, type: duel|topk|sliders, payload, context: {n, shell, D_eff, layout, seed}, policy_version }`
- Record before/after CCR‑Scalar weight distribution, KL change, and realized ΔU_H.
- Audit checks: bias tests (order, position), stability (KL caps), and safety (no increase in veto/contradiction rates).

---

## 13) PreferencesPolicy API Stub
```
PreferencesPolicy {
  stages: {
    stage0: { n_max: 4, modes: [duel], hysteresis_ticks: H0 },
    stage1: { n_range: [5,6], modes: [duel, topk], hysteresis_ticks: H1, topk_interval: I1 },
    stage2: { n_min: 7, modes: [duel, topk, sliders], hysteresis_ticks: H2, slider_decay: D, kl_cap: KLC }
  },
  context_coupling: { use_D_eff: true, promote_if: [layout_disagree, cal_drift, stop_storms], demote_delay_ticks: HD },
  shortlist: { K_pre: ..., sampler: "bwk_prebatch" },
  bias_controls: { randomize_order: true, balance_winrate: true },
  logging: { level: "detailed" }
}
```

---

## 14) Test Plan for Staged Preferences (no invented constants)
1. **Replay by n and D_eff:** evaluate Stage 0/1/2 against fixed‑mode baselines on realized ΔU_H, safety ceilings, and info‑per‑cost.
2. **Hysteresis sweep:** vary hysteresis ticks; measure Master stability and STOP storms.
3. **Cognitive load probe:** track interaction counts and completion times per stage; ensure sub‑tick overhead stays within budget.
4. **Drift & disagreement stress:** trigger override conditions; verify promotion/demotion logic and recovery.
5. **Bias audits:** order randomization, balanced win‑rates, and pilot studies against intransitivity.


---

## 15) Adaptive Acquisition Policy — AutoAcquire (Scaled Usage)
**User directive:** Scale usage of acquisition rules by regime rather than pick one. The scheduler selects among IDS, KG, PESC, and qEHVI per tick (and can mix them in a batch) according to context—without inventing numeric constants.

### 15.1 Signals used for routing
- **Constraint uncertainty (CU):** variance of safety surrogates; safe‑set size |S_t|; feasibility rate.
- **Objective sharpness (OS):** posterior variance and predicted regret for utility.
- **Nonlinearity (NL):** Section 28 NL markers; residual clustering across shells/bridges.
- **Multi‑objective emphasis (MO):** current CCR‑Scalar weight spread (how many components of U_H carry weight).
- **Batch & budget (K_main, B_main):** batch size and per‑candidate cost.

### 15.2 Routing policy (conceptual, no numbers)
- **PESC route** when **CU is high** and |S_t| is small or volatile → prioritize feasibility discovery.
- **IDS route** when **OS is high** with moderate CU → balance regret² vs information to steer exploration efficiently.
- **KG route** when **OS is moderate** and **K_main is large** → myopic value‑of‑information with strong batched performance.
- **qEHVI route** when **MO is high** (frontier components matter simultaneously) → explicit hypervolume growth under constraints.

**Mixing:** allow a **convex mixture** within the same batch: a fraction α_PESC + α_IDS + α_KG + α_qEHVI = 1 chosen by BwK to maximize expected ΔU_H per cost. If any route violates compute or numerics, reassign its mass to the next best route.

### 15.3 Batch & budget solver
- Default **BwK primal–dual** for multi‑resource budgets (time, memory, human‑label effort).
- Use **UCB‑knapsack** if there is a single resource and arms are independent.

### 15.4 Fallbacks
- If nonlinearity markers are weak and compute is tight → **Linear‑only + KG** on the shortlist.
- If constraints are well‑known (stable safe set) → **IDS/KG** focus; turn off PESC.
- If U_H collapses to a single dominant component → fall back to **single‑objective KG/IDS**.

### 15.5 API stub
```
AutoAcquirePolicy {
  routes: [pesc, ids, kg, qehvi],
  router_signals: { CU, OS, NL, MO, K_main, B_main },
  mixture: { enabled: true, optimizer: "bwk_primal_dual" },
  fallbacks: { linear_only_ok: true, disable_pesc_if_stable_safe: true },
  logs: { per_candidate: true }
}
```

### 15.6 Test plan (no invented constants)
- **Routing ablations:** single‑route baselines vs AutoAcquire on ledger replays; measure realized ΔU_H, safety ceilings, and info‑per‑cost.
- **Mixture learning:** vary mixture freedom (fixed vs optimized) and evaluate stability and compute.
- **Stress:** constrained‑heavy starts (PESC‑dominant), then shift to uncertainty‑dominant (IDS), then to batch exploitation (KG); verify smooth transitions and no STOP storms.


---

## 16) Cost Model & Budgeting Policy — AutoCost (Requires Testing)
**User directive:** Cost modeling *must* be validated empirically and depends on system-wide and per-component performance. This section defines a scalable policy without fixing numbers.

### 16.1 Resources and costs
- **Resource vector r:** { time, memory, I/O, human‑interaction, network } per candidate action.
- **Cost model c(a):** predictive distribution over r for action a with mean and quantiles; learned from ledger traces.
- **Features:** option type, microkernels, data size, chamber bits, E8/Leech coords & shells, seed lineage, prior runtimes, cache status, device type.

### 16.2 Predictors
- **Fast ranker:** gradient‑boosted trees or linear model for mean cost.
- **Uncertainty:** quantile regression or Bayesian last‑layer to provide PI (prediction intervals).
- **Cold‑start:** similarity in (E8, Leech, chamber) feature space and option family priors.

### 16.3 Budgeting & selection
- **Single‑resource:** maximize acquisition per second (e.g., KG/IDS score per predicted time); guard with PI to avoid tail latency.
- **Multi‑resource:** **BwK primal–dual** selection; maintain shadow prices λ for each resource and choose batches that maximize (expected ΔU_H − λ·E[c(a)]).
- **Dynamic surge:** increase λ for resources at high utilization; decay when pressure eases (no fixed constants here).

### 16.4 Fairness & tenancy
- Apply **Dominant Resource Fairness (DRF)** when multiple tenants or task classes contend; enforce minimum quanta from Section 15 and per‑class ceilings.

### 16.5 Cost‑aware acquisitions
- Allow cost‑aware variants: EI per cost/second, log‑EI per cost, Pandora‑Box/Gittins‑style indices; switchable via AutoAcquire.
- **Multi‑fidelity:** when actions offer fidelity knobs (e.g., subset size, resolution), treat fidelity as an environmental variable; pick cheap probes that maximize value per cost and extrapolate to full fidelity.

### 16.6 Scheduler hooks
- **Pre‑batch:** estimate E[c(a)], PI, and resource margins; compute batch via BwK.
- **Runtime drift:** if realized cost breaches PI, raise λ for that resource and back‑off similar actions.
- **Post‑batch:** log realized r, update predictors and λ.

### 16.7 SNAP logging additions
```
cost_event: {
  predictors: version_id,
  r_pred: { mean, q10, q90 },
  r_realized: { ... },
  lambda_before: { ... }, lambda_after: { ... },
  selection_rule: { route_mix, per_cost_metric },
  violations: { tail_latency, mem_oom, io_stall }
}
```

### 16.8 Test plan (no invented numbers)
1. **Trace modeling:** k‑fold CV on historical ledger to fit c(a); report MAPE, PI coverage, and tail error by action family.
2. **Policy ablations:** single‑resource vs multi‑resource; static vs surge λ; DRF on/off; compare ΔU_H and safety ceilings.
3. **Stress tests:** synthetic bursts for each resource; evaluate STOP storms, backlog growth, and Master stability.
4. **Cold‑start:** unseen action families; assess fallback accuracy and penalty.
5. **Multi‑fidelity:** measure time‑to‑frontier with subset/resolution knobs vs full‑fidelity only.


---

## 17) Surrogate Feature Schema — Base & Extended (Utility + Safety)
**Goal:** Lock a minimal, stable, content-safe feature set shared by utility and safety surrogates. No raw content; only anonymized identifiers, hashes, or geometry summaries. Numbers are not fixed and remain to be learned.

### 17.1 Base features (always on)
- **Geometry:**
  - E8: coord[8], shell index, residual norm, sensitivity score (flip rate under micro-perturbations).
  - Leech: coord[24], shell index, residual norm.
  - Chamber signature: 4-bit code (one-hot or integer id 0–15).
- **Topology:**
  - Bridge count and mean bridge cosine; local degree in e-graph; path multiplicity (2/4/8).
  - Layout disagreement score across top-k layouts.
- **Calibration & reliability:** per-view calibration error, joint calibration error, contradiction rate.
- **Progress:** closure score, glyph coverage %, Master stability indicator (thrash vs stable), SNAP reuse %.
- **Seed lineage:** generation, lineage entropy, consensus rate across seeds.
- **Cost hints:** cached size/bytes, microkernels required, last runtime quantiles.

### 17.2 Extended features (on when available)
- **Glyph/codebook:** triad indices (g_idx, c_idx, o_idx) embedded via hashing trick or small learned embeddings; parity pass/fail bit.
- **Shape fingerprint:** WL-hash bucket id (hashed), BLAKE3 content digest bucket id (hashed), shell span length.
- **Governance rungs:** legality/safety/fairness/provenance/repeatability margins (distance to threshold), veto hit counts (short window).
- **Temporal:** tick age of stance, TTL remaining, drift indicators per view.
- **Environment:** device class, memory pressure level (coarse), data-fetch latency quantiles.

### 17.3 Transformations
- Standardization/whitening for numeric blocks; clip heavy tails with winsorization.
- Categorical blocks: hashing trick with stable seed; optional learned embeddings for GP feature maps via kernel on embeddings.
- Composite distances for kernels: RBF on (E8, Leech, residuals), Hamming on chamber bits, dot/RBF on embeddings, graph-aware kernel on (degree, bridges).

### 17.4 Feature groups per signal
- **Utility:** geometry + topology + calibration + progress + cost hints.
- **Legality/Safety/Fairness:** governance margins + geometry/topology + drift; optionally include environment.
- **Provenance/Repeatability:** progress + governance margins + seed lineage + environment.

### 17.5 Featurizer API
```
Featurizer {
  version,
  inputs: { snap_id | candidate_spec },
  outputs: { x_numeric, x_categorical, meta },
  hashing: { seed, dims: { glyph: Dg, wl: Dw, layout: Dl } },
  transforms: { standardize: true, winsorize: { p: 0.99 } }
}
```

### 17.6 Tests (no invented constants)
1. **Ablations:** base vs base+extended; per-signal feature drop tests; measure Δ log‑likelihood, calibration, and safety ceilings.
2. **Leak checks:** verify no raw content or PII; only hashed ids and geometry summaries.
3. **Stability:** replay under seed perturbations; ensure features change smoothly with micro‑rotations.
4. **Cost prediction tie‑in:** confirm added features improve AutoCost accuracy without overfitting.

---

## 18) Open Decisions (Feature Schema)
1. Embedding sizes for glyph/WL/layout buckets (hash vs learned embeddings).
2. Exact residual and sensitivity metrics used in geometry block.
3. Graph features to include by default (degree, betweenness, clustering) vs keep behind a toggle.
4. Normalization policy for per-view calibration errors across domains.
5. Privacy budget for hashed categorical dims (collision rate targets).


---

## 19) Per‑Tick Categorical Representation — AutoEmbed (Hash ⇄ Embedding)
**User directive:** The choice between hashing and learned embeddings must be **decided each tick** based on current context, data, and budgets.

### 19.1 Signals (evaluated per tick)
- Data volume and concentration per categorical family (unique ids, head vs long tail).
- Collision rate in hashed features, gradient norm magnitude, and calibration error deltas.
- Mutual information between categorical ids and residuals (utility and safety models).
- Nonlinearity markers (from Section 28), D_eff, and layout disagreement.
- Budget signals: B_main, memory pressure level, and expected featurization cost.
- Drift indicators (appearance of new ids, shift in frequency).

### 19.2 Policy (no fixed numbers)
- Default to **Hashing** for all families.
- Promote a family to **Learned Embeddings** when predicted info‑per‑cost gain exceeds a policy threshold and budgets allow.
- Demote back to **Hashing** if memory pressure rises, calibration degrades, or overfitting indicators trip (gap load increase, instability, or STOP storms).
- Allow **Hybrid mode:** head categories get embeddings; long tail stays hashed.

### 19.3 Selection mechanism
- **Bandit routing per family:** maintain a small bandit (UCB/EXP3) over {hash, embed, hybrid}; reward = realized ΔU_H per cost with safety ceilings enforced.
- **Warm starts:** persist embeddings across ticks; apply decay/regularization when demoted.
- **Checkpointing:** snapshot embedding tables with content‑address; roll back on instability.

### 19.4 Featurizer contract (extension)
```
CategoricalRepPolicy {
  families: [glyph, wl_hash, layout_id, chamber_code, ...],
  per_family: {
    name,
    mode: hash|embed|hybrid,
    dims: { hash: D_h, embed: D_e },
    router: { type: bandit, reward: Delta_U_H_per_cost }
  },
  budgets: { mem_cap: ..., featurize_time_cap: ... },
  stability: { decay_rate: ..., rollback_on: [calibration_spike, stop_storm] }
}
```

### 19.5 Logging
`representation_event: { family, mode_before, mode_after, context: {n, shell, D_eff}, signals: {...}, reward_estimate, realized_Delta_U_H }`

### 19.6 Test plan
- **Ablations:** hash‑only vs embed‑only vs AutoEmbed; measure ΔU_H, calibration, and cost.
- **Stress:** long‑tail surges, memory pressure bursts, and rapid id churn.
- **Hybrid validation:** head vs tail split quality and overall stability.


---

## 20) Gold/Gray Semantics & Graduation — Zero‑Ambiguity Gold
**User directive:** A gold item MUST contain **0 ambiguous understanding**. All other states are “shades of gray,” scaling by ~10% biases.

### 20.1 Ambiguity detectors (all must be clear for Gold)
- **Evidence ambiguity:** any unresolved contradiction or missing provenance for required sources/derivations.
- **Governance ambiguity:** any helix rung (legality, safety, fairness, provenance completeness, observer repeatability) uncertain or unverified.
- **Chamber stability ambiguity:** chamber bits flip under micro‑perturbations or mirror banks within the current tolerance policy.
- **Cross‑seed ambiguity:** disagreement across the active seed family on cell/chamber assignment or signs of undue seed‑lane influence.
- **Cross‑layout ambiguity:** disagreement among the consulted top‑k Leech layouts (beyond any explicitly enabled neighborhood tolerance).
- **Residual/sensitivity ambiguity:** persistent residual anomalies or sensitivity ridge flags around the current address.
- **Calibration ambiguity:** per‑view or joint calibration failing current error bounds.

> **Gold condition:** All ambiguity detectors above must be **off** (no active flags) **and** helix rungs pass. Otherwise, the item is not Gold.

### 20.2 Shades of Gray — decile bands
Define a **continuous ambiguity mass** from the detectors (monotone aggregation over standardized signals). Map this mass to **decile bands** to label shades:
- **Gray10, Gray20, …, Gray90**

“~10% biases” are represented by these deciles; higher deciles indicate stronger reliance on biased/uncertain factors (e.g., single‑lane dominance, unstable chambers, residual anomalies). Exact standardization and aggregation are learned/tested; no numeric constants are fixed here.

### 20.3 Graduation rule (Gray → Gold)
Promote from Gray to Gold **only if**:
1) All helix rungs pass (lexicographic gates),
2) All ambiguity detectors are clear (no flags),
3) Cross‑seed and cross‑layout addresses are stable (no disagreement under current policy),
4) Calibration checks are current, and
5) Sentinel post‑check is clean; Arbiter warrant includes rationale; Porter commit succeeds.

If any detector re‑flags after promotion, **auto‑demote** to the appropriate Gray decile and attach a Review Cue.

### 20.4 SNAP additions
```
ambiguity: {
  detectors: {
    evidence:false, governance:false, chamber:false,
    cross_seed:false, cross_layout:false, residual:false, calibration:false
  },
  gray_decile: null | 10|20|...|90,
  rationale: ["which detectors fired"],
  last_promotion_tick: T?
}
```

### 20.5 Tests (no invented constants)
- **Zero‑ambiguity feasibility:** replay ledger to measure attainable Gold rates and detector false‑clear/false‑flag behavior; tighten/relax detectors if Gold is unattainable in practice.
- **Stability:** shock tests (seed perturbations, mirror rotations) after promotion; ensure no thrash between Gold and high Gray deciles.
- **Attribution:** verify that Gray deciles correlate with bias sources (seed‑lane MI, chamber instability, residual anomalies) and improve with targeted sub‑tick gap work.


---

## 21) Cross‑Layout Consensus — Neighborhood Tolerance & External Evidence (NTC)
**User directive:** Allow tolerance. Outside‑lattice / outside‑universe data must be used to form deeper connections and inform best pathing first; most invalid answers should inform a valid answer elsewhere.

### 21.1 Concepts
- **Neighborhood (Leech):** cells within a policy radius in Leech space (decoded/coset distance) and an allowed shell window.
- **Neighborhood (E8):** cells within a policy radius in E8 space; may include bridge‑adjacent cells.
- **Chamber tolerance:** chamber bitstrings within a small Hamming radius.
- **External evidence:** observations or metadata not represented by the current lattice/layout projections (raw features, domain tags, instruments, documents); used to re‑project or enrich context.

*(All radii/shell windows are policy parameters learned by tests; no numbers fixed here.)*

### 21.2 NTC consensus rule (tolerant voting)
1. **Top‑k layouts:** select k layouts by fit.
2. **Neighborhood sets:** for each layout, compute the neighborhood around its Leech cell (plus optional E8 back‑projection and chamber tolerance).
3. **Consensus:** accept agreement if a **majority** of layouts overlap on a non‑empty neighborhood intersection that also satisfies governance gates and residual bounds.
4. **Escalate to external evidence** when overlap is empty or unstable: acquire exogenous signals, update projections, or spawn a new layout candidate; then recompute neighborhoods.
5. **Fallback container:** if residuals remain high, switch to alternate quantizer (OPQ/PQ/HNSW) while retaining parity checks; re‑attempt consensus under NTC.

### 21.3 Using invalid answers as guidance (negative evidence)
- **Witness SNAPs:** any vetoed/invalid merge or failed consensus is logged as a **witness** with the conflicting addresses, residuals, and rungs that failed.
- **Forbidden masks:** update per‑shell **forbidden regions** or **anti‑bridges** so seeds avoid known dead‑zones.
- **Counterfactual branching:** generate nearby neighborhoods (via bridges/coverings) most likely to flip the outcome; schedule targeted probes in the next main tick.
- **Glyph impact:** negative evidence creates **guard glyphs** that can block recomputation paths unless new external evidence appears.

### 21.4 Safety & governance coupling
- Tolerance is **disabled or tightened** in high‑risk domains (legal/fairness‑critical) via domain packs; strict equality can be forced by Sentinel.
- Neighborhood intersections must pass helix rungs; Sentinel can lower tolerance near sensitivity ridges or calibration drift.

### 21.5 Metrics & audits
- NTC acceptance rate, rollback rate (false consensus), Δ closure score, Δ residuals, bridge discovery rate, and info‑gain per cost.
- Negative‑evidence reuse rate and time‑to‑resolution for previously invalid regions.

### 21.6 API stub
```
TolerancePolicy {
  k_layouts,
  e8: { radius: R_e8, shell_window: S_e8 },
  leech: { radius: R_ll, shell_window: S_ll },
  chamber: { hamming_radius: H },
  enable_external_evidence: true,
  fallback_quantizers: [opq, pq, hnsw],
  domain_overrides: { legal: strict, fairness: strict, research: tolerant }
}

NegativeEvidencePolicy {
  witness_snap: true,
  forbidden_masks: true,
  counterfactual_branching: { max_neighbors: M },
  glyph_guards: true
}
```

### 21.7 Tests (no invented constants)
- **Strict vs tolerant:** compare rollback rate and Δ closure under both modes across shells.
- **External evidence ablations:** with/without exogenous probes; measure consensus speed and safety ceilings.
- **Mask & branching efficacy:** time‑to‑avoidance of dead‑zones, and conversion rate of invalid→valid via counterfactual probes.


---

## 22) Asymmetric Neighborhood Tolerance — Domain & Space Aware (Requires Testing)
**User directive:** Use **asymmetric tolerance** in natural layouts if testing supports it. Tolerance may differ across Leech, E8, and chamber spaces and may be domain‑dependent by default.

### 22.1 Default asymmetry (no fixed numbers)
- **Leech (24‑D):** most tolerant. Rationale: triple projection (governance, context, observation) aggregates structure; near‑neighbors often represent semantically compatible states.
- **E8 (8‑D):** moderate tolerance. Rationale: mid‑level synthesis; bridges can justify limited radius and cross‑shell connections.
- **Chamber bits (4‑bit):** strictest tolerance. Rationale: beam routing and optics gates; flips indicate instability and require higher evidence.

### 22.2 Domain presets (overridable)
- **Legal / Fairness critical:** strict chamber, stricter E8, cautious Leech; external evidence mandatory for widening.
- **Safety‑critical (engineering, medical):** strict chamber, moderate E8, cautious Leech.
- **Research / Discovery:** relaxed Leech, moderate E8, cautious chamber; aggressive use of external evidence.

### 22.3 Context coupling
Tighten tolerance near sensitivity ridges, high calibration error, or STOP storms. Loosen with strong external evidence, high cross‑seed consensus, or stable bridges.

### 22.4 Learning the asymmetry
- **Offline replay:** grid search / Bayesian optimization over radii per space to minimize rollback rate and maximize closure and ΔU_H subject to safety ceilings.
- **Online adaptation:** small bandit over {strict, moderate, tolerant} per space with reward = realized ΔU_H per cost and penalties for rollbacks and veto hits.

### 22.5 API additions
```
AsymmetricTolerancePolicy {
  per_space: {
    leech: { mode: tolerant|moderate|strict, radius: R_ll, shell_window: S_ll },
    e8:    { mode: tolerant|moderate|strict, radius: R_e8, shell_window: S_e8 },
    chamber:{ mode: tolerant|moderate|strict, hamming_radius: H }
  },
  domain_presets: { legal: {...}, fairness: {...}, safety: {...}, research: {...} },
  adapt: { ridge_tighten: true, external_evidence_relax: true, bandit_update: true }
}
```

### 22.6 SNAP logging
`tolerance_vector: { leech: {mode, R_ll, S_ll}, e8: {mode, R_e8, S_e8}, chamber: {mode, H} }` with context (n, shell, D_eff) and reason codes (ridge, evidence, domain override).

### 22.7 Tests (no invented constants)
- **Preset ablations:** compare domain presets under matched workloads.
- **Ridge stress:** verify automatic tightening reduces rollbacks without stalling closure.
- **Evidence uptake:** measure gains from relaxing after strong external evidence.


---

## 23) Bridge Creation Policy — Confidence Thresholding (Requires Testing)
**User directive:** A bridge must exceed a **confidence threshold**; exact specifics are to be learned by testing. This section defines the score, gates, and lifecycle without inventing numbers.

### 23.1 Bridge Confidence Score (BCS)
A monotone aggregation of signals in [0,1] with governance as hard gates:
- **Similarity (S_sim):** residual‑aware similarity between endpoints (Leech + E8 composite distance) with chamber compatibility penalty.
- **Neighborhood consensus (S_ntc):** overlap depth from Section 21 (NTC) across top‑k layouts and optional E8 neighborhoods.
- **Stability (S_stab):** 1 − flip rate under micro‑perturbations; ridge penalty from sensitivity map.
- **Cross‑seed support (S_seed):** fraction of seed families that propose the same bridge; MI(seed lane→outcome) penalty.
- **Residual quality (S_res):** inverse residuals at endpoints and along the local path.
- **Directional consistency (S_dir):** consistency with beamline mirrors/diagonals and option flow direction.
- **External evidence (S_evd):** strength of witness/evidence links (documents/instruments/observations) explaining the connection.
- **Governance (G_gate):** helix rungs must pass for both endpoints (legality, safety, fairness, provenance completeness, observer repeatability). If any rung fails, **BCS is undefined** and the bridge is blocked.

BCS can be implemented as a calibrated logistic model or a learned weighted sum with non‑negativity and Lipschitz constraints. No coefficients are fixed here; they are learned in replay tests.

### 23.2 Thresholded lifecycle (θ values learned)
- **Candidate bridge (C‑bridge):** BCS ≥ θ₁ and G_gate passes → record, **no traversal** yet; enqueue for probes.
- **Provisional bridge (P‑bridge):** BCS ≥ θ₂, confirmed by ≥2 seed families and stable for K micro‑ticks → allow **read‑only traversal** for exploration and scoring; no write‑side effects.
- **Stable bridge (S‑bridge):** BCS ≥ θ₃, NTC consensus holds across top‑k layouts, sensitivity below ridge bound, and zero governance alerts over a grace window → allow **full traversal** for options and indexing.

Auto‑demote if stability fails, residuals spike, or governance flags appear. Attach a Review Cue on demotion.

### 23.3 Safety & domain overrides
- **Regulated domains (legal/fairness/safety‑critical):** higher θ and longer grace windows; external evidence required for P‑bridge; strict chamber compatibility.
- **Research/discovery:** lower θ₁ for candidate discovery; θ₂/θ₃ remain guarded by governance and stability.

### 23.4 SNAP & index additions
```
bridge: {
  endpoints: [snap_id_a, snap_id_b],
  bcs: { sim, ntc, stab, seed, res, dir, evd },
  state: C|P|S,
  gates: { helix_pass: true, domain_override: null|{...} },
  ttl: { created: T0, last_check: T1, grace_window: Gw },
  evidence_ptrs: [cid:...],
  demotions: [ {tick, reason} ]
}
```

### 23.5 Tests (no invented numbers)
1. **Offline replay labeling:** construct positives from human‑validated or long‑lived S‑bridges; negatives from rollbacks and vetoed bridges. Fit/calibrate BCS and pick θ₁/θ₂/θ₃ to meet target FPR/FNR under safety ceilings.
2. **Ablations:** drop each signal term; measure rollback rate, Δ closure, and safety hits.
3. **Stress:** seed monoculture spikes, ridge neighborhoods, residual surges; verify demotion logic and STOP behavior.
4. **Domain audits:** verify overrides reduce violations without stalling frontier growth.


---

## 24) All‑Resources Contextual Placement Check (ARPC)
**Intent:** Before committing any placement (chamber bits, E8 cell, Leech cell/layout) or upgrading state (Gray→Gold, +16 attempt, Bridge promotion, Master update), perform a **holistic placement verification** that can use **all available resources** (internal artifacts and allowed external evidence) **as long as they are valid to context**. No numeric constants are fixed here; thresholds are learned/tested.

### 24.1 What “placement” means here
A placement \(P\) is the tuple: `{ chamber_signature, E8:{cell,shell}, Leech:{cell,layout}, stance_hash }` associated with a SNAP.

### 24.2 Context envelope (must hold for any evidence)
- **Scope:** stance vectors (governance, context/need, observation/limits) and domain pack.
- **Temporal:** time window / TTL alignment with the observation and instrument timestamps.
- **Observer/instrument:** declared instruments or evaluation pipelines; reproducible under those instruments.
- **Policy:** domain overrides (e.g., strict tolerance for legal/fairness), helix rungs.

Any evidence used by ARPC must satisfy the envelope; else it is ignored or queued for re‑projection.

### 24.3 Resource graph (sources ARPC may consult)
1) Internal: SNAPs, glyph chains, per‑shell indices, witness SNAPs, bridge records, residual/sensitivity maps, seed lineage dashboards.
2) Geometric: neighborhood sets under NTC (Section 21), candidate bridges (Section 23), AutoSwitch/AutoTopo surrogate posteriors.
3) External: domain adapters (documents/measurements/metadata) passed through governance packs and content sanitizers into hashed/embedded features (no raw content in features).

### 24.4 Validation rubric (RACT+G)
For a candidate placement \(P\), ARPC scores evidence with a rubric (learned aggregation):
- **Relevance (R):** semantic/geometric proximity to \(P\) (neighborhood overlap, residual‑aware distance, feature match).
- **Attribution (A):** provenance completeness and back‑pointers (traceable, hashed identifiers, parity checks).
- **Consistency (C):** cross‑seed, cross‑layout, and bridge consistency; no contradictions with witness SNAPs.
- **Temporal (T):** time alignment with stance/instrument TTL.
- **Governance (G):** helix rungs pass; domain overrides respected.

**Placement verdict** requires passing G and exceeding learned thresholds on R, A, C, T aggregates. Otherwise, return “unresolved” with next‑best neighborhoods for probing.

### 24.5 Actions on failure/uncertainty
- **Counterfactual probes:** schedule targeted evidence acquisition via AutoAcquire to the nearest neighborhoods/bridges most likely to resolve ambiguity.
- **Witnessing:** log mismatch as witness SNAPs and update forbidden masks.
- **Re‑projection:** when external evidence is relevant but out of envelope, queue for context‑preserving re‑projection (adapter‑pack) and re‑run ARPC.

### 24.6 When ARPC runs
- Pre‑Gold promotion; pre‑+16 attempt; pre‑Bridge Provisional→Stable; pre‑Master digest update; and whenever Review Cues flag residual/sensitivity anomalies.

### 24.7 API stubs
```
ARPCRequest {
  snap_id,
  proposed_placement: { chamber, e8:{cell,shell}, leech:{cell,layout} },
  context_envelope: { stance, temporal, instruments, domain_policy },
  resources: { internal:true, geometric:true, external:true }
}

ARPCVerdict {
  decision: accept|reject|unresolved,
  scores: { R:..., A:..., C:..., T:..., G: pass|fail },
  relied_evidence: [cid:...],
  counterfactual_targets: [ { neighborhood, expected_info_gain } ],
  notes: [ "witness:...", "forbidden_mask:update" ]
}
```

### 24.8 Tests (no invented constants)
- **Envelope validity checks:** inject out‑of‑scope evidence and verify it is ignored or re‑projected; measure false accepts.
- **Replay validation:** measure acceptance accuracy vs human‑validated placements and long‑lived Gold SNAPs.
- **Stress:** conflicting external evidence, high drift, STOP storms; verify ARPC prefers safe unresolved → probe rather than unsafe acceptance.
- **Cost/latency:** ensure ARPC fits within B_main/B_sub via AutoCost; back‑off gracefully under resource pressure.


---

## 25) Adaptive ARPC for +16 — Lightweight Tentative Acceptance
**User directive:** If all checks are valid, perform a **lightweight validity check** before accepting +16 merges, and **tentatively allow** the +16 **until** it is *proven vague in any way*.

### 25.1 Lightweight ARPC (L‑ARPC) criteria (no new numbers)
Run **L‑ARPC** instead of full ARPC when **all ambiguity detectors are clear** (Section 20) and NTC consensus is strong (Section 21):
- **Helix rungs pass** (lexicographic gates).
- **Cached evidence only:** reuse SNAP indices, glyph chains, residual maps, seed consensus, and tolerance neighborhoods; **no new external fetch** unless missing provenance.
- **Quick stability checks:** micro‑perturbation flip scan at minimal budget; confirm chamber and E8/Leech addresses unchanged.
- **NTC majority** holds with current radii; no ridge hot‑spots flagged.
- **Calibrations current**; joint reliability curve within error bounds.
If all pass, proceed with **tentative +16**; otherwise, fall back to full ARPC.

### 25.2 Tentative +16 semantics
- **Conditional warrant:** Arbiter issues a **conditional** +16 warrant with TTL; Porter commits a **tentative SNAP delta**.
- **Monitoring window:** for a short, policy‑defined window (ticks/micro‑ticks), Sentinel runs background monitors (cross‑seed agreement, residual drift, ridge watch, layout disagreement).
- **Full promotion:** if the window completes with no triggers, remove the tentative flag; +16 stands.

### 25.3 “Proven vague” triggers (auto‑rollback/demotion)
- Any **ambiguity detector** turns on (evidence, governance, chamber stability, cross‑seed, cross‑layout, residual/sensitivity, calibration).
- **NTC intersection** collapses or shrinks below policy overlap.
- **Residual spike / ridge alert** at the +16 address.
- **Witness SNAP** arises contradicting the +16 consolidation.

**Action:** Sentinel issues a STOP for that scope; Arbiter prepares a rollback warrant; Porter **demotes** to 8 (or the last consistent state) and logs a Review Cue with counterfactual probe targets.

### 25.4 SNAP & Warrant fields
```
+16_tentative: {
  enabled: true|false,
  ttl: { start_tick, expires_tick },
  l_arpc_checks: { helix:true, cached_evidence:true, ntc_majority:true, stability:true, calibration:true },
  monitors: { cross_seed:true, residual_watch:true, ridge_watch:true, layout_disagree:true },
  triggers_seen: [ ... ]
}

warrant: {
  type: "conditional+16",
  ttl_sec: ...,
  rollback_on: [ ambiguity_detector, ntc_collapse, residual_spike, witness_conflict ]
}
```

### 25.5 Tests (no invented constants)
1. **Replay:** compare L‑ARPC + tentative vs full ARPC on realized closure, rollback rate, and ΔU_H under the same budgets.
2. **Trigger sensitivity:** inject synthetic ridge/residual spikes and cross‑seed disagreements; verify timely demotion and STOP containment.
3. **Budget impact:** measure compute saved by L‑ARPC and the overhead of monitors; tune thresholds via AutoCost/BwK without fixing numbers here.


---

## 26) AutoSubspace — Best 2‑D Subspace per +4 Group (Requires Testing)
**User directive:** Subspace choice is determined by best testing results. This policy selects, per tick and per +4 group, the 2‑D projection that maximizes downstream stability and holistic utility, without fixing numbers.

### 26.1 Candidate families (enabled by context)
- **Supervised:** LDA (when labels or binary outcomes exist), supervised contrastive 2‑D, SIR/SAVE.
- **Paired‑view:** CCA or PLS (when two view sets are naturally paired within the +4 group).
- **Unsupervised:** PCA (rand‑SVD), Kernel PCA/RFF‑KPCA, Isomap/UMAP‑style locality (as a toggle; default off in safety‑critical domains).
- **Domain‑meta:** linear projection from curated meta‑features (handcrafted invariants) with optional shrinkage.

### 26.2 Subspace Quality Score (SQS)
Composite, monotone score in [0,1] combining:
- **Residual gain:** Δ reduction in E8/Leech quantization residuals.
- **Sensitivity relief:** Δ reduction in ridge/flip rate under micro‑perturbations.
- **Consensus:** improvement in NTC neighborhood overlap across layouts.
- **Calibration:** Δ improvement in per‑view and joint calibration error.
- **Bridging:** Δ bridge density (valid stable bridges only) without rollback spikes.
- **Closure:** Δ in closure score and glyph coverage.
- **Cost:** penalty for compute/memory (AutoCost); favor lower cost when SQS ties.

(Weights and thresholds are learned in replay; none fixed here.)

### 26.3 Selection procedure (per +4 group, per tick)
1. **Generate candidates** allowed by context (labels/paired views/domain mode).
2. **Fit on seed‑split folds** (cross‑seed CV); compute SQS on held‑out.
3. **Pick argmax SQS**; if difference < ε, choose cheaper candidate.
4. **Freeze subspace for this tick**; record provenance in SNAP; expose to E8 synthesis.
5. **Fallback:** if selected subspace causes safety or stability regressions online (Review Cue), revert to last stable.

### 26.4 Safety & governance
- Strict domains disable manifold learners by default; supervised/linear options prioritized.
- ARPC runs before promotion of placements produced with a new subspace.

### 26.5 SNAP additions
```
subspace: {
  group_id: "+4:gX",
  method: "lda|cca|pls|pca|kpca|sir|save|meta",
  context: { labels: true|false, paired_views: true|false, domain: ... },
  sqs: { residual: ..., sensitivity: ..., consensus: ..., calibration: ..., bridges: ..., closure: ..., cost: ... },
  chosen: true,
  fallback_to: null|{ method, tick }
}
```

### 26.6 API stub
```
AutoSubspacePolicy {
  enabled_methods: [lda, cca, pls, pca, kpca, sir, save, meta],
  safety_mode: { strict_domains_disable_nonlinear: true },
  selection: { cross_seed_cv: true, tie_break: "min_cost" },
  fallbacks: { enable_revert_on_review_cue: true }
}
```

### 26.7 Tests (no invented constants)
- **Replay grid:** compare methods per +4 group across shells; measure SQS components and downstream ΔU_H.
- **Safety ablations:** strict vs tolerant domain modes; ensure no increase in STOP storms.
- **Cost trade‑off:** measure compute vs stability; validate tie‑break policy.


---

## 27) AutoSeed — 8‑Seed Cohorts (Fixed Floor + Adaptive Surplus, Requires Testing)
**User directive:** Start with **sets of 8 seeds**. Permit **cross‑set comparison on request**; allow all sets to inform each other **if needed** but **not by default**. Treat this as a fixed exploration floor with adaptive surplus and test to learn the best split.

### 27.1 Cohorts & lanes
- **Cohort:** an 8‑seed set `C_j = {S0..S7}` with diversified lanes: views, mirrors, tiebreaks, layout jitter, arbiter, porter (Section 3). Cohorts are the atomic exploration unit.
- **Active floor:** at least one cohort active each tick (fixed floor). Additional cohorts are added adaptively if budgets permit.
- **Isolation by default:** inter‑cohort influence is **off**; cross‑set signals are consulted only when explicitly requested or when triggers fire (27.3).

### 27.2 Allocation (per tick)
- **Floor:** 1 cohort devoted to exploration (diversity‑max). 
- **Surplus:** remaining budget assigns extra cohorts/seeds to exploitation or targeted probes via AutoAcquire/BwK, respecting AutoCost.
- **Diversity guards:** Sobol/LDS for mirrors, stratified tiebreak sampling, and MI(seed‑lane→outcome) caps; reseed lanes that dominate.

### 27.3 Cross‑set comparison policy
- **Off by default.** Enable only on **request** or when triggers occur: persistent layout disagreement, sensitivity ridges, residual anomalies, STOP storms, or closure stagnation.
- **When enabled:** run **paired‑cohort A/B** with matched budgets and log consensus cell rate, disagreement entropy, and multiplicity realized. Results may update global thresholds/configs; otherwise influence remains local.

### 27.4 Exploit vs explore split (no fixed numbers)
- **Fixed floor + adaptive surplus:** maintain a minimum exploration cohort; allocate surplus via BwK to maximize ΔU_H per cost subject to diversity and safety constraints.
- **Starvation avoidance:** each class (gray resolution, governance audits, calibration refresh, layout consensus, option exploration) receives minimum quanta per Section 15.

### 27.5 Lineage, retirement, and reseeding
- **Lineage:** record `{vector, parent, generation}` (Section 3). 
- **Retire** seeds that are dominated on the Pareto frontier (quality, risk, cost) or that breach MI caps.
- **Reseed** with jittered lanes; preserve cohort identity for auditing.

### 27.6 SNAP logging & dashboards
```
seed_cohort: {
  cohort_id: "C:j",
  seeds: [S0..S7],
  role: exploration|exploitation|probe,
  isolation: true|false,
  triggers: ["layout_disagree", "ridge", ...],
  metrics: { consensus_rate, entropy_map_ref, multiplicity }
}
```
Dashboards: cohort consensus, entropy heatmaps, multiplicity profiles, MI per lane, retire/reseed events.

### 27.7 Tests (no invented constants)
- **Floor ablation:** 8‑seed floor vs smaller/larger floors; compare ΔU_H, safety ceilings, and info‑per‑cost.
- **Cross‑set A/B:** on‑request vs always‑on; measure benefit vs overhead.
- **Diversity guards:** MI caps and reseeding frequency vs outcome stability.
- **Budget sensitivity:** surplus allocation under varying B_main; ensure no class starvation.

