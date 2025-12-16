# Project Board — AGRM/SNAP Build Plan

**Scope:** Convert our backlog into an actionable board with acceptance tests and milestones. This reflects everything explained so far. Implementation happens only when you say so.

---

## Milestones

- **M0 — Workspace & Scaffolding**  
  Repo layout, configs, CI smoke, deterministic seeding, logging.
- **M1 — Navigator: Golden‑Ratio/Weyl Ordering**  
  Implement GR spiral stepper; later E8‑aware projections.
- **M1.5 — E8‑DoF Navigator Core**  
  Add 8‑DoF state (x,y,z,r,p,y,σ,φ), φ‑stagger scheduler, gated integrator hooks; wire into Navigator as the sweep state machine.
  - **Acceptance (v0.4, 2025‑08‑13):**  
    - SAFE configuration has ≤ AGGR thrash at equal ticks.  
    - φ‑stagger reduces stalls vs no‑φ baseline at same ticks.  
    - Inverse‑lane reports ROI (avg inverse score − avg main score) ≥ 0 on ≥ 50% of seeds.
- **M2 — Patching Engine**  
  Missed‑node detection & legal insertion (single node & short chains).
- **M3 — MDHG MVP**  
  Hot‑zone hash table, velocity metrics, neighbor cache; wire into Navigator/Builder.
- **M4 — Salesman Portfolio**  
  2‑opt/3‑opt/LK‑light; flag analytics.
- **M5 — Metrics & Audit→Auto‑Tuning**  
  Baselines (NN, MST preorder), expanded Audit rules; SNAP of runs.
- **M6 — Tests & Fixtures**  
  Unit, property, adversarial geometries, seeded regressions.
- **M7 — SNAP Schema + Archivist/Repository**  
  Minimal schema, ingestion, compaction, replay.
- **M8 — SFBB Bridge (Seeds/Prodigals/Bridges)**  
  Port ProdigalManager & laminate stats; φ‑bridge hints.
- **M9 — SAP Envelope & Safe Cube Hooks**  
  Sentinel/Arbiter/Porter stubs; cube construction & expansion policy.
- **M10 — Domain RAG Parser**  
  Floors/Rooms/E8 segmentation; elevator seeding.


- **M0 — Workspace & Scaffolding**  
  Repo layout, configs, CI smoke, deterministic seeding, logging.
- **M1 — Navigator: Golden‑Ratio/Weyl Ordering**  
  Implement GR spiral stepper; later E8‑aware projections.
- **M1.5 — E8‑DoF Navigator Core**  
  Add 8‑DoF state (x,y,z,r,p,y,σ,φ), φ‑stagger scheduler, gated integrator hooks; wire into Navigator as the sweep state machine.
- **M2 — Patching Engine**  
  Missed‑node detection & legal insertion (single node & short chains).
- **M3 — MDHG MVP**  
  Hot‑zone hash table, velocity metrics, neighbor cache; wire into Navigator/Builder.
- **M4 — Salesman Portfolio**  
  2‑opt/3‑opt/LK‑light; flag analytics.
- **M5 — Metrics & Audit→Auto‑Tuning**  
  Baselines (NN, MST preorder), expanded Audit rules; SNAP of runs.
- **M6 — Tests & Fixtures**  
  Unit, property, adversarial geometries, seeded regressions.
- **M7 — SNAP Schema + Archivist/Repository**  
  Minimal schema, ingestion, compaction, replay.
- **M8 — SFBB Bridge (Seeds/Prodigals/Bridges)**  
  Port ProdigalManager & laminate stats; φ‑bridge hints.
- **M9 — SAP Envelope & Safe Cube Hooks**  
  Sentinel/Arbiter/Porter stubs; cube construction & expansion policy.
- **M10 — Domain RAG Parser**  
  Floors/Rooms/E8 segmentation; elevator seeding.

---

## Work Items & Acceptance Tests

### M1 — Navigator GR/Weyl Ordering (P1)
- **Implement:** GR stepper with φ‑related angle increment; shell‑aware ordering; hook for future E8 projections.
- **Acceptance:** On random Euclidean TSP (N∈{100,300,500}), mean tour length improves ≥5% vs. nearest‑neighbor at equal step budgets; deterministic under fixed seeds.

### M2 — Patching Engine (P1)
- **Implement:** Missed‑node finder; legal insertion via curvature/shell/sector validator; short‑chain splice.
- **Acceptance:** Zero missed nodes on ≥95% of random instances (N≤200) within time cap; no constraint violations (proof via validator logs).

### M3 — MDHG MVP (P1)
- **Implement:** `MDHGHashTable` with hotness (EWMA), velocity, neighbor lists; `get_cache(name)` in StateBus; query from Navigator/Builder.
- **Acceptance:** ≥30% cache hit‑rate on repeated seeds; ≥15% runtime drop on N≥500 vs. no‑MDHG runs.

### M4 — Salesman Portfolio (P1)
- **Implement:** 2‑opt/3‑opt + LK‑light; segment flags by shell/sector; feed Audit.
- **Acceptance:** ≥3% mean tour reduction post‑builder on N∈[100,1000] without exceeding 1.2× time budget.

### M5 — Metrics & Auto‑Tuning (P1)
- **Implement:** Baselines (NN, MST preorder); metrics (length ratio, merge distance, patch count, stall density, salesmen delta); Audit rules that recommend parameter tweaks.
- **Acceptance:** Second run improves length or time in ≥70% of cases; all metrics logged per run SNAP.

### M6 — Tests & Fixtures (P1)
- **Implement:** Pytest + Hypothesis; fixtures: ring, two‑cluster, near‑colinear, hub‑spoke; seeded regressions.
- **Acceptance:** CI green on unit/property tests; regressions stable across 10 seeds.

### M7 — SNAP + Archivist (P1)
- **Implement:** Minimal SNAP schema (config, seeds, lattice cut, path, flags, recs); Archivist append‑only store with compaction; replay tool.
- **Acceptance:** Runs are restorable bit‑for‑bit from SNAP; compaction reduces store size ≥40% on 50 runs.

### M8 — SFBB Bridge (P2)
- **Implement:** Import prodigal/laminate stats; φ‑guided bridge hints; seed‑map registry.
- **Acceptance:** Ablation shows φ‑bridge hints reduce imperfect transitions or length by statistically significant margin on n=6/7 benchmarks.

### M9 — SAP & Safe Cube (P1)
- **Implement:** Sentinel (policy checks), Arbiter (quarantine/restore), Porter (signed moves); Safe Cube constructor & expansion protocol.
- **Acceptance:** Violations trigger quarantine within 1 tick; restore to last good SNAP; cube expansions logged and reproducible.

### M10 — Domain RAG Parser (P1)
- **Implement:** Floors/Rooms segmentation; entity linking; elevator seeding; glyph triggers at N=5.
- **Acceptance:** Retrieval quality +15% recall@k vs. baseline; faithfulness (grounding rate) ≥95%.

---

## Definition of Done (DoD)
- Deterministic seeds logged; reproducible via SNAP replay.
- Metrics emitted; Audit recommendations round‑trip into next run.
- Unit/property tests for new code; docs updated.

## Risks & Mitigations
- Sprawl → decay + quotas + Archivist compaction.
- Governance deadlocks → time‑boxed Arbiter decisions; fallback tiers.
- Symmetry overreach → treat Weyl as prior; ablations mandatory.

---

## Tracking
Each item maps to the Running To‑Do Log IDs (T‑###) we’ve been maintaining; completion requires passing the associated acceptance tests above.


## Navigator φ-hysteresis & inverse lane (added now)
- T-135 — φ-hysteresis gating with cooldown (SAFE vs AGGR)
  - **Why:** Quantify safety vs. churn by introducing open/close thresholds and flip cooldowns.
  - **Action:** Implement per-sector gating state, θ_open/θ_close, cooldown; track thrash rate.
  - **Priority:** P1

- T-136 — Inverse-lane sampling budget
  - **Why:** Safely probe alternate paths; compare yield vs. cost.
  - **Action:** Add per-tick inverse budget, offset sector target (φ+½), and metrics.
  - **Priority:** P1

- T-137 — Acceptance tests for M1.5
  - **Why:** Ensure φ-stagger reduces stalls; SAFE < AGGR thrash at equal ticks.
  - **Action:** Unit/property tests; seeded regressions; metrics wiring.
  - **Priority:** P1


---

# Next Sprint — Detailed Worklist (versioned deliverables)

## M2 — Patching Engine (insertion/repair)
- **T-200 — Legal ops catalog**  
  **Why:** Formalize what “patching” means.  
  **Action:** Spec single-node insert, 2-node chain, short-chain splice, edge swap; define invariants.  
  **Deliverables:** `core/agrm/patching_spec_v0_1_2025_08_13.md`.  
  **Status:** **SHIPPED v0.1 (2025-08-13)**.
- **T-201 — Patch heuristics v1**  
  **Action:** Score = Δlength + coverage gain + tension (shell/sector) − compute budget.  
  **Deliverables:** `core/agrm/patching_v0_1_2025_08_13.py` with `propose()`, `apply()`, `revert()`.  
  **Status:** **SHIPPED v0.1 (2025-08-13)** (SNI only, TNI/SCS/ESW stubs next).
- **T-202 — Chain detection**  
  **Action:** Identify missed runs via sector continuity + VWS edge frequency gaps.  
  **Deliverables:** `core/agrm/chain_detect_v0_1_2025_08_13.py`.  
  **Status:** **SHIPPED v0.1 (2025-08-13)** (gap finder + seed pairs).
- **T-203 — Safety gates & logs**  
  **Action:** SAP hooks, rollback SNAP on failed patch.  
  **Deliverables:** integrated in controller; log to `snapshots/patch_runs/`.  
  **Status:** **PARTIAL v0.1 (2025-08-13)** (policy stubs only, wiring pending in controller v0.5).
- **T-204 — A/B harness**  
  **Action:** Compare NO-PATCH vs PATCH on suite; export CSV.  
  **Deliverables:** `experiments/ablation_patching_v0_1_2025_08_13.py`.  
  **Status:** **SHIPPED v0.1 (2025-08-13)** (greedy-completion vs patch-completion from same partial tour; CSV export next).
  **Action:** Compare NO-PATCH vs PATCH on suite; export CSV.  
  **Deliverables:** `experiments/ablation_patching_v0_1_2025_08_13.py`.

## M3 — MDHG MVP (hot/edge maps)
- **T-210 — Schema & API**  
  **Action:** `insert`, `bump_heat`, `decay`, `k_nn`, `edges`, SNAP export.  
  **Deliverables:** `core/agrm/mdhg_v0_1_2025_08_13.py`.  
  **Status:** **SHIPPED v0.1 (2025-08-13)** — MVP with exact kNN; snapshot/restore; co-hit edges.
- **T-211 — Buildings/Floors/Rooms mapping**  
  **Action:** Namespace hot zones; per-floor caches; room-level vector sets.  
  **Deliverables:** `core/agrm/mdhg_v0_2_2025_08_13.py`.  
  **Status:** **SHIPPED v0.2 (2025-08-13)** — building/floor/room routing + scoped kNN + per-building edge views.
- **T-212 — Decay + promotion**  
  **Action:** Background decay; promote edges on repeated hits.  
  **Status:** **SHIPPED v0.1** (simple λ-decay + co-hit promotion). Stability test next.
- **T-213 — VWS seeding**  
  **Action:** Seed Navigator warm-start from MDHG (top edges + kNN ring).  
  **Deliverables:** `core/agrm/vws_bridge_v0_1_2025_08_13.py`; controller wiring in `controller_v0_5_2025_08_13.py`.  
  **Status:** **SHIPPED v0.1 (2025-08-13)**.
- **T-214 — Snapshot & restore**  
  **Action:** `save_snp()`, `load_snp()` compatible with Archivist.  
  **Status:** **SHIPPED v0.1** (JSON snapshot/restore in module; wire to Archivist later).

## M4 — Salesman Portfolio (local improve) (local improve)
- **T-220 — 2‑opt baseline**  
  **Deliverables:** `core/agrm/salesman_2opt_v0_1_2025_08_13.py`.  
  **Status:** **SHIPPED v0.1 (2025-08-13)** — looped 2‑opt with negative‑Δ convergence; demo shows Δ≈−2.7 on n=200 probe.
- **T-221 — 3‑opt & LK‑light hooks**  
  **Deliverables:** `core/agrm/salesman_3opt_v0_1_2025_08_13.py`, `salesman_lk_light_v0_1_2025_08_13.py`.  
  **Status:** **PENDING**.
- **T-222 — Portfolio chooser**  
  **Action:** Pick heuristic based on shell/sector entropy + MDHG heat.  
  **Deliverables:** `core/agrm/salesman_portfolio_v0_1_2025_08_13.py`.  
  **Status:** **PENDING**.

## M5 — Metrics, Audit, Auto‑Tuning
- **T-229 — Auditor v1**  
  **Action:** JSON-safe run recorder (hash, version, cfg, stats).  
  **Deliverables:** `core/agrm/audit_v0_1_2025_08_13.py`.  
  **Status:** **SHIPPED v0.1 (2025-08-13)** — robust serializer handles sets/ndarrays.
- **T-230 — Auto‑tune stub**  
  **Action:** Lightweight grid over gate thresholds & cooldown; objective = coverage − penalties(thrash, stalls).  
  **Deliverables:** `experiments/autotune_v0_1_2025_08_13.py`.  
  **Status:** **SHIPPED v0.1 (2025-08-13)** — prints best cfg; logs each probe to snapshots.

**Next (suggested):**
- **T-231 — CSV+Parquet export** for experiments to feed dashboards.
- **T-232 — SAP wiring in controller v0.6** (quarantine, rollback SNAP on anomaly).
- **T-233 — Portfolio 3‑opt/LK‑light** to boost route quality for TSP benchmark.
- **T-234 — MDHG stability tests** under decay for churny workloads.

