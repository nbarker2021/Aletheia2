# Family — TPG Bridges & Proposals (Telemetry)

**Linkage:** Running Log Reconstruction (Fresh Pass v3) → Package #5.

---

## Scope (doc‑grounded)
- Capture **TPG** integration into the iterate loop (telemetry‑only), emission of **Bridge SNAPs** and **Proposal SNAPs**, and Ops report/A‑B harness artifacts.
- Bind to **Policy/Beacons** (neg‑beacon penalties), **Trails** (events, drill‑downs), and **MDHG** (future soft hints).

---

## Components & Evidence
- **Modules:** `surgery.py` (2‑opt), `run.py` (one‑shot runner), orchestrator hook (`agrm/orchestrator/iterate.py` → `tpg.enabled=true`), docs/`TPG.md`. fileciteturn28file2L1-L8 fileciteturn28file2L9-L17 fileciteturn28file2L19-L23 fileciteturn28file2L25-L33
- **Telemetry‑only wiring:** Trail events `{event:"tpg", …}` per iteration; no promotion changes. fileciteturn28file2L20-L23
- **SNAPs:** `bridge::tpg_bridge` for low‑overlap edges; `proposal::tpg_order` (ordered room IDs). fileciteturn28file0L30-L35
- **Reports:** Ops report JSON/CSV from v0.28.1 demo; pivot parameters (max_nodes, beam, imperf_threshold, two_opt). fileciteturn28file0L22-L26 fileciteturn28file0L49-L58
- **A/B & Golden Runner:** FAST and expanded batteries; Trail drill‑down and colorized deltas; neg‑beacon penalties reliable. fileciteturn28file13L49-L57 fileciteturn28file12L60-L74

---

## Interfaces (as in log)
- `tpg` config in iterate: `{enabled,max_nodes,beam,imperf_threshold,use_two_opt}`. fileciteturn28file4L18-L25
- **Trail event schema:** `{event:"tpg", iter, imperfections, cost, complete}`. fileciteturn28file2L20-L23
- **SNAP overlays:** `universe.overlays["bridges"|"proposals"]`. fileciteturn28file4L28-L31

---

## Acceptance & Tests
- Telemetry present: Trail events on every TPG pass; Ops report rows filled for each universe. fileciteturn28file0L47-L55
- Bridge/Proposal SNAP emission verified; counts recorded in Trails. fileciteturn28file0L41-L45
- A/B deltas computed; Trail anchors clickable; neg‑beacon penalties honored. fileciteturn28file12L68-L76

---

## Risks & Mitigations
- **Leakage/contamination across families:** integrate **neg‑beacons** in edge costs; manifest‑driven overrides. fileciteturn28file12L70-L76
- **Telemetry drift:** pin `policy_hash` and scenario seeds in manifests (Golden Runner discipline). fileciteturn28file13L27-L33

---

## TODOs (family doc)
- Backfill sample **Bridge SNAP** and **Proposal SNAP** records with Trail pointers.  
- Add a tiny JS **Trail viewer** embed to the Ops report (expand rows inline). fileciteturn28file12L62-L69  
- Prepare canary config for **proposal‑as‑hint** (soft), with strict telemetry & rollback.


---

## Package #5 — Golden Runner & A/B artifacts (append)
**Golden Runner (FAST v0.29.0):** scenarios: Gov_bimodal_s1, Fin_line_s2, Doc_ring_s3 (N≈220, max_nodes=18, beam=24). Artifacts: `golden_runner_main_results_FAST_v0_29_0_2025_08_14.csv`, `golden_runner_metamorphic_FAST_v0_29_0_2025_08_14.csv`; TPG < greedy/chrono on cost & imperfections; relabel invariance passed.  
**Golden Runner (v0.29.1):** adds time‑shift stability (+/−3%), leakage metrics with neg‑beacons on/off, superperm sanity (n=3–6; n=5 overlap heatmap). Artifacts: `golden_runner_main_results_v0_29_1_2025_08_14.csv`, `golden_runner_meta_results_v0_29_1_2025_08_14.csv`, `golden_runner_leakage_results_v0_29_1_2025_08_14.csv`, `superperm_sanity_v0_29_1_2025_08_14.csv`, `n5_overlap_heatmap.png`.

**Manifest Runner (v0.30.1→0.30.2):** replay enforced via `policy_hash`; HTML report with Archivist counts; stable Trails. Artifacts: `/mnt/data/snaplat_manifest_v0_30_1_2025_08_14/manifest_results.csv`, `/mnt/data/snaplat_manifest_v0_30_1_2025_08_14/manifest_report.html`; later `/mnt/data/snaplat_manifest_v0_30_2e_2025_08_14/manifest_report.html`, `manifest_results.csv`, `manifest_trails.csv`.  
**2‑opt telemetry (v0.30.2):** bounded 2‑opt deltas exported as `tpg_2opt_delta_cost`, `tpg_2opt_delta_imperf` (telemetry‑only).

**A/B harness (v0.30.3 → v0.30.6):**
- **v0.30.3:** Direct vs Playbook; HTML `ab_report.html`; CSVs `ab_results.csv`, `ab_pivot.csv`.
- **v0.30.5:** Neg‑beacon penalty wired (`agrm/config/neg_beacons.py`); per‑scenario playbook profiles; Trails JSON; CSVs `ab_results.csv`, `ab_deltas.csv`; HTML `ab_report.html`.
- **v0.30.6:** Trail viewer anchors in HTML; colorized deltas; neg‑beacon penalties made reliable; bundles for 0.30.5/0.30.6 recorded.

**TPG config & events:** orchestrator passes `{enabled,max_nodes,beam,imperf_threshold,use_two_opt}`; iterates log Trail event `{event:"tpg", iter, imperfections, cost, complete}`; Bridge/Proposal SNAPs collected under `universe.overlays["bridges"|"proposals"]` (telemetry‑only).

## Validation — P#5 (adds)
- **V3‑TPG‑006:** Golden Runner FAST artifacts present; invariants (relabel) pass reported.  
- **V3‑TPG‑007:** v0.29.1 leakage deltas reported when neg‑beacons on; time‑shift stability included.  
- **V3‑TPG‑008:** Manifest runner enforces replay via `policy_hash`; Trails stable; HTML links present.  
- **V3‑TPG‑009:** v0.30.3 A/B artifacts present; pivot table shows per‑universe deltas.  
- **V3‑TPG‑010:** v0.30.5 A/B includes neg‑beacon penalty + per‑scenario profiles; `ab_deltas.csv` present.  
- **V3‑TPG‑011:** v0.30.6 Trail viewer anchors + colorized deltas; neg‑beacon penalties honored via config + overrides.

## TODOs — P#5 (adds)
- Backfill char‑range for P#5 start/end; stamp mini‑hash on close.  
- Attach Golden Runner (v0.29.0/0.29.1) CSVs + heatmap to package manifest.  
- Link v0.30.x A/B HTML/CSVs + Trails JSON; verify anchors.  
- Add 2‑opt telemetry columns to schema; ensure reported in results.  
- Expand neg‑beacon matrix; expose scenario overrides; record leakage deltas.

---

## Package #5 — Ops & Reporting details (append)
**Ops report schemas (doc‑grounded):**
- **`golden_runner_main_results*.csv`**: `scenario,universe,mode,seed,cost,imperfections,coverage_pct,time_ms,mem_mb,relabel_ok,trail_id`.
- **`golden_runner_metamorphic*.csv`**: `scenario,universe,metamorph(relabel|time_shift|seed_flip),delta_cost,delta_imperf,delta_coverage,pass,trail_id`.
- **`ab_results.csv`**: `scenario,universe,arm(direct|playbook),cost,imperfections,leakage,mrr,coverage_pct,time_ms,trail_id`.
- **`ab_pivot.csv`**: pivoted per‑universe stats with arm columns (`cost_*,imperf_*,leakage_*`) + computed deltas.
- **`ab_deltas.csv`**: `scenario,universe,delta_cost,delta_imperf,delta_leakage,delta_mrr,delta_time_ms,significance,trail_anchor`.
- **`manifest_trails.csv`**: `manifest_id,policy_hash,seed,scenario,artifact,trail_id`.

**HTML surfaces:**
- **`manifest_report.html`**: policy hash pin, replay params, Archivist counts, Trail table (links).
- **`ab_report.html`**: per‑scenario sections, colorized deltas, **Trail viewer anchors** (expand row in place), neg‑beacon config summary.

**Neg‑beacon matrix (config path):**
- `agrm/config/neg_beacons.py` provides defaults; per‑scenario overrides via manifest; exposure in A/B HTML and CSV.
- Leakage deltas computed with neg‑beacons ON vs OFF; reported in `ab_deltas.csv` and highlighted in HTML.

**Validation — P#5 (adds):**
- **V3‑TPG‑012:** `relabel_ok` present in Golden Runner CSVs; all FAST rows set `true`; metamorphic relabel tests pass.
- **V3‑TPG‑013:** Neg‑beacon matrix surfaced in A/B (HTML + CSV); leakage delta columns present; overrides honored by manifest.
- **V3‑TPG‑014:** All HTML tables include Trail anchors; random sample of anchors resolve to Trails.

**TODOs — P#5 (adds):**
- Attach CSV/HTML artifacts to package manifest; verify `policy_hash` consistency across Manifest/Golden/A‑B.
- Sample 10 Trail anchors from A/B and Golden Runner; log any broken links to E‑DBSU.
- Export neg‑beacon override table per scenario (`neg_beacon_overrides.csv`); bind to A/B report.

---

## Package #5 — Playbooks & Kernel Priors (append)
**Scope (seeded from P#5 text):** playbook profiles per scenario, kernel priors referenced in A/B configs, and policy‑gated parameterization (telemetry‑only in this phase).  
**Artifacts (seed):** `playbooks/*.yaml`, `kernel_priors/*.yaml` (manifest‑loaded), per‑scenario overrides rendered in A/B HTML.

**Config path (observed in P#5 context):** manifests surface `playbook_profile` and optional `kernel_priors` → orchestrator consumes → A/B reports show active profile + overrides.

**Acceptance (to verify):**
- Playbook name/profile appears in `ab_results.csv` and `ab_report.html` headers.
- Kernel‑prior fields pinned in manifest Trail; replay with same `policy_hash` reproduces.
- Overrides reflected in per‑scenario sections; deltas computed against **Direct**.

**Validation (adds):**
- **V3‑TPG‑015:** `playbook_profile` shown in A/B tables + HTML.  
- **V3‑TPG‑016:** `kernel_priors` snapshot attached to manifest Trail; replay consistent.  
- **V3‑TPG‑017:** Per‑scenario overrides present and honored (HTML + CSV).

**TODOs (adds):**
- Attach first discovered `playbooks/*.yaml` and `kernel_priors/*.yaml` to the P#5 manifest.  
- Export an index table of scenarios → (playbook, kernel‑priors) for A/B appendix.

---

## Next Pass Plan (active)
1) Finish P#5 ingestion: attach Golden/Manifest/A‑B artifacts; verify Trail anchors; lock V3‑TPG‑012..017.  
2) Detect next pivot; on first non‑TPG/A‑B scope, close P#5 (mini‑hash) and **open P#6** with seeded ledger and validations.  
3) If dense clusters appear (e.g., Playbooks & Priors internals), open a **family canvas** only if it speeds deep review (no slowdown to main scan).

