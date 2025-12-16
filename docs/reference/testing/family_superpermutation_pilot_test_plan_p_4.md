# Family — Superpermutation Pilot & Test Plan (Package #4)

**Linkage:** Running Log Reconstruction (Fresh Pass v3) → Package #4 (Superpermutation pilot).  
**Char range (from main canvas):** [460093–… ) *(open; mini‑hash on close)*

---

## Scope (doc‑grounded)
- Capture the **greedy pilot** experiments at **n=3..6** and their metrics.  
- Record the **heavy test plan**: measures, acceptance criteria, and failure taxonomy.  
- Bind fixtures (Egan sequences), artifacts (Universal Superpermutation doc; cross‑field framework), and outputs (logs, tables, charts) to the package manifest.  
- Feed **policy notes** back to **AGRM/MDHG** (review‑only; no behavior change in this phase).

## Artifacts & fixtures
- **Docs:** `universal_superpermutation_principle.pdf` (or similarly named) — conceptual basis.  
- **Code:** `cross_field_framework` — evaluation harness (names as seen in main log).  
- **Fixtures:** `Egan n=7`, `Egan n=8` sequences — reference data for overlap/coverage checks.  
- **Pilot outputs:** per‑n logs/tables (n=3..6), overlap histograms, coverage summaries.

## Pilot methodology (n=3..6)
- **Generator:** greedy builder with tie‑break rules (as‑logged).  
- **Measures per n:**
  - **Imperfect join rate** (joins requiring backtrack/patch).  
  - **Overlap histogram** (k‑overlap counts; shape vs expected).  
  - **Coverage metric** (% permutations seen; unique path coverage).  
  - **Path length** (total sequence length; delta vs naive bounds).  
  - **Runtime × memory** (bounded estimate; per‑n trend).
- **Recording:** write a **pilot SNAP** per n with metrics + lineage to fixtures/artifacts.

## Heavy test plan (acceptance criteria)
- **Targets:** expand to **n=7** baseline with Egan fixtures; dry‑run **n=8** feasibility checks.  
- **Acceptance (per run):**
  1. Overlap histogram present; no missing bins; expected shape heuristics satisfied.  
  2. Coverage ≥ configured threshold (e.g., 99.9% for pilot; 100% for confirmed).  
  3. Path length ≤ policy cap (configurable), with delta vs baseline recorded.  
  4. Imperfect join rate trend non‑explosive vs n (documented slope/CI).  
  5. Trails include **failure taxonomy** entries for all non‑pass cases.
- **Outputs:** `iter_report.html`, `pilot_metrics.csv`, `overlap_hist.png`, `coverage.json`, per‑n **pilot SNAPs**.

## Failure taxonomy (first pass)
- **Type A:** Overlap under‑coverage (missing overlap bins).  
- **Type B:** Coverage shortfall (unique permutations < threshold).  
- **Type C:** Path inflation (length exceeds cap).  
- **Type D:** Instability (variance across seeds too high).  
- **Type E:** Resource breach (runtime/memory guard hit).

## AGRM/MDHG policy feedback (review‑only)
- Record notes on **greedy bias** and **laminate merges** to inform sampler weights and elevator thresholds; do not alter policies in this review phase.  
- Attach notes as `policy_feedback::superperm` SNAPs.

## Weyl coverage (package overlay)
- For cross‑system integration: record **8×2** coverage counters once superperm outputs are mapped to lattice directions (pending mapping step, not performed here).

## TODOs (family doc)
- Extract pilot logs/tables; populate `pilot_metrics.csv` with per‑n rows.  
- Generate `overlap_hist.png` and `coverage.json` from recorded runs.  
- Draft `acceptance_criteria.md` from doc text; link from plan.  
- Emit `policy_feedback::superperm` notes with citations back to main log.

