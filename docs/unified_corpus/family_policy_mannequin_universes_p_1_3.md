# Family — Policy, Mannequin & Universes (Packages #1–#3)

**Linkage:** Running Log Reconstruction (Fresh Pass v3) → Packages #1–#3.

---

## Scope (doc‑grounded)
Unify the **governance path** (PolicyBus → Sentinel/Safe‑Cube) with **Mannequin v2** (SAP‑gated slice streaming, Porter custody, budget guard) and **Universe** surfaces (profiles, materializers, overlays/diffs). No invention; strictly what appears in the log.

---

## Components

### 1) PolicyBus & Profiles
- Central gate used by Mannequin, Controller, Promotion.
- Default: **allow unless policy denies**; Safe‑Cube requires tags when enabled.
- **Profiles:** Universe‑scoped YAML, `apply_policy(...)` loads and pins a `policy_hash` used in manifests and gates.

### 2) Sentinel (Safe‑Cube)
- Enforces policy at Mannequin gate; denies when required tags absent.
- Allow path returns `sentinel_pass` with budgets (surface prediction + 125% guard noted in the stream).
- Gate decisions and rationale recorded to slice **SNAPs**.

### 3) Mannequin v2 — SAP‑gated streaming lattice
- States: DORMANT → WARM → ACTIVE_SLICE; **Porter custody** for each wave.
- **Budgets:** predicted nodes with **125% guard**; per‑wave **slice SNAP** with lineage.
- Criteria activate selective materialization (only touched regions wake up).

### 4) Universe surfaces
- **Profiles** (policy YAML), **materializers**, **overlays/diffs**.
- Materializers: `to_points(universe, criteria, wave_size, max_nodes)` → 8‑D coords; `to_graph(...)` → neighbor graph + elevator candidates (quota, k/buckets/backends).
- **Overlays/diffs:** `attach_mdhg_overlay(...)` and `write_universe_diff(...)` journal state transitions (before/after hashes).

---

## Interfaces & SNAPs (as per log)
- **Gates:** `PolicyBus.gate(payload)` → `{allow, reason, predicted, budget}`; `sentinel_pass` on allow.
- **Slice SNAPs:** per wave with custody + budgets + policy reason.
- **Manifests:** run/slice/promotion manifests include seeds, policies (`policy_hash`), toggles, counts.
- **Overlays:** `universe_diff::overlay` SNAP with pointers to before/after.

---

## Acceptance & Tests (from log)
- Safe‑Cube demo: deny without tags; allow with tags (budgets surfaced).
- Materializers round‑trip on small universes; quotas respected; elevators emitted when enabled.
- Overlays/diffs idempotence checks; policy hash pinned in manifests.

---

## Risks & Mitigations
- **Policy drift / mis‑pinning:** enforce `policy_hash` stamping and validation in every manifest.
- **Budget underestimate:** 125% guard + Porter custody + abort on overflow.
- **Slice thrash:** hysteresis (Mannequin warm/cold budgeting); cache slice predictions in Trails.

---

## TODOs (family doc)
- Extract example **policy YAML** and `policy_hash` stamps from log snippets; add to `examples/policies/`.
- Capture a sample **slice SNAP** and **gate decision** for Safe‑Cube deny vs allow.
- Collate a **universe overlay/diff** pair (before/after pointers) with hashes for the MDHG overlay attach flow.

---

## Package #4 — pipeline modules & batch runs (append)
**Implemented modules (v0.31.1):** `uniqueness.py` (canonicalization + fingerprints + registry), `caches.py` (step + negative), `laminates.py` (strict builder + backtracking + envelope gates), `promotion.py` (FS2 records + uniqueness), `batch_runner.py` (multi‑seed orchestration). **Bundle:** `snaplat_pkg_v0_31_1` (zip).  
**Run:** 3‑seed strict batch (n=7) using 5906 envelope; step cache/neg cache; uniqueness gating.  
**Artifacts:** tables: *Batch results — n=7 strict (v0.31.1)*, *Promotions — n=7 (v0.31.1)*; bundle: `snaplat_v0_31_1_package_and_workspace.zip`.  
**Scaled batch (v0.31.2):** 8 seeds × 90k iters, **per‑seed prefill** from 5906 (≈400 early hints), strict envelope + laminate backtracking + caches + uniqueness registry. **Bundle:** `snaplat_v0_31_2_scaled_batch.zip`.

**Policy gates active:** FS2 promotion = **perfect+novel only** (envelope ≤5906 each step, final 5906; canonical hash unique; secondary k‑gram/overlap if needed). Step Cache hints only recorded after passing envelope+coverage; Negative Cache skips patterns after threshold strikes.

**Validation (adds):**
- **V3‑SPM‑009:** v0.31.1 bundle present; tables list step/neg cache stats and uniqueness outcomes.  
- **V3‑SPM‑010:** v0.31.2 scaled batch present; prefill doc noted; caches show increased hit/use rates.

**TODOs (adds):**
- Prefill safety note: confirm all prefilled hints are envelope‑safe (proof via attached envelope checkpoints).  
- Emit *cache_delta.snap* per promotion (what new hints were learned).  
- Add *neg_cache_thresholds.json* (dynamic schedule: lenient→strict).

---

## Package #4 — baselines & claims (clarifications)
- **Claimed result (v0.30.23):** AGRM/MDHG‑guided n=7 length **5913** with 0 imperfections; artifacts at `/mnt/data/snaplat_experiments_v0_30_23/superperm_n7_ag/...`. Treat as **silver** in registry; **gold** remains 5906.  
- **Baseline (v0.30.22):** naive n=7 length **35277** (coverage ok); used only to validate ingestion and plotting.

**Validation (adds):**
- **V3‑SPM‑011:** Registry lists 5906 (gold), 5913 (silver), and any auto‑scanned successes; canonicalization is relabel+rotation+reversal invariant; pairwise checks recorded.

---
---

## Package #4 — scaled batches & modules (consolidated)
**Scaled Batch v0.31.2:** 8 seeds × 90k iters; per‑seed prefill (~400 early hints) from 5906; strict 5906 envelope; laminate backtracking; step + negative caches; uniqueness registry; bundle: `snaplat_v0_31_2_scaled_batch.zip`.

**Scaled Batch v0.31.3:** 10 seeds × 120k; dynamic negative thresholds (lenient→strict); **trail‑keyed** step cache (context disambiguation); trail‑aware negative keys; FS2 artifacts include cache deltas; doc‑intake smoke; bundle: `snaplat_v0_31_3_scaled_batch_and_intake.zip`.

**Operational modules (v0.31.1+):** `uniqueness.py` (canonicalization + fingerprints + registry), `caches.py` (step + negative), `laminates.py` (strict builder + backtracking + envelope gates), `promotion.py` (FS2 records + uniqueness), `batch_runner.py` (multi‑seed orchestration). Bundle: `snaplat_v0_31_1_package_and_workspace.zip`.

**Gates (active in P#4):**
- **Envelope rule:** n=7 gold=5906; silver=5908; n=8 reference=46205.
- **Promotion:** perfect + **novel** only (canonical SHA invariant to relabel/rotation/reversal → registry; k‑gram/overlap as secondary checks).
- **Caches:** Step Cache stores only envelope/coverage‑passing moves; Negative Cache rejects normalized anti‑patterns after threshold; trail‑keying reduces mis‑hints.

**Artifacts (P#4 index):** pilot CSV/JSON (n=3..6); baselines bundles (5908; 5906 & 46205); `snaplat_uniqueness_registry_n7_v0_30_29.json`; `step_cache.json`; `neg_cache.json`; Egan fixtures.

**Metrics schema (per‑run):** `n, seed, builder_mode, length, coverage_pct, imperfect_join_rate, overlap_hist(json), runtime_ms, mem_mb, envelope_ok, uniqueness_ok, promoted, trail_id`.

**Trails:** one Trail per run with lineage to fixtures, envelopes, uniqueness registry, caches, and the package manifest; **delta Trail** for 5906 vs 5908.

**Doc‑intake (smoke):** intake skeleton + W5H on *Universal Superpermutation* doc; planned Phase D: SNAP taxonomy tagging + plotting validator.

**Pivot status:** still within superperm theme; next pivot closes P#4 when a non‑superperm artifact/scope appears.

## Validation — P#4 (roll‑up)
- **V3‑SPM‑001:** Pilot CSV/JSON attached to manifest.  
- **V3‑SPM‑002:** Egan baselines (5906, 46205) ingested with envelopes; 5908 packaged as silver.  
- **V3‑SPM‑005:** Uniqueness registry present; promotion rule recorded.  
- **V3‑SPM‑006:** Step/Negative cache files present; cache hit/use metrics logged.  
- **V3‑SPM‑007:** Every pilot row has a Trail id.  
- **V3‑SPM‑008:** Delta Trail for 5906 vs 5908 exists.  
- **V3‑SPM‑009/010:** v0.31.1/0.31.2 bundles present; scaled results logged; prefill doc noted.  
- **V3‑SPM‑011:** Registry lists 5906 (gold), 5913 (silver), auto‑scanned successes; canonical invariances enforced.

## TODOs — P#4 (open)
- Attach pilot CSV/JSON + scaled results to manifest; generate `pilot_metrics.csv` roll‑up.  
- Emit `delta_5906_vs_5908.json` + `delta_report.html`; wire Delta Trail.  
- Add dynamic neg schedule config (`neg_cache_thresholds.json`); expose start/end thresholds.  
- Persist cache deltas alongside promotions (FS2); ensure rollback hooks.  
- Finish doc‑intake Phase D wiring; run multi‑doc ingest test under SNAP taxonomy.

