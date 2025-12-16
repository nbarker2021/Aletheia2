# Family — Ops & Reporting (Golden, Manifest, A/B)

**Linkage:** Running Log Reconstruction (Fresh Pass v3) → Package #5.

---

## Scope (doc‑grounded)
Unify operations/reporting surfaces: **Golden Runner** (FAST + full), **Manifest Runner** (policy‑pinned replay), and **A/B Harness** (Direct vs Playbook) with **Trail** integration and neg‑beacon leakage accounting.

---

## Artifacts & Schemas
- **Golden Runner** CSVs: `golden_runner_main_results*.csv`, `golden_runner_metamorphic*.csv` (fields: scenario, universe, mode, seed, cost, imperfections, coverage_pct, time_ms, mem_mb, relabel_ok, trail_id; plus metamorphic deltas).
- **Manifest Runner**: `manifest_results.csv`, `manifest_trails.csv`, `manifest_report.html` (policy hash pin, Archivist counts, Trail links).
- **A/B Harness**: `ab_results.csv`, `ab_pivot.csv`, `ab_deltas.csv`, `ab_report.html` (colorized deltas, Trail viewer anchors, neg‑beacon summary, scenario overrides).

---

## Config & Replay Discipline
- Manifests pin **`policy_hash`** and seeds; A/B exposes **playbook profiles** and **neg‑beacon overrides**; Golden Runner FAST validates metamorphic invariants (e.g., relabel).

---

## Trail Bindings
- One **Trail** per run; A/B and Golden link rows to Trails; HTML embeds **Trail viewer anchors** for drill‑down.

---

## Acceptance & Validation
- CSVs include `trail_id`; HTML links resolve; `relabel_ok` present and true for FAST runs; neg‑beacon leakage deltas computed and displayed.

---

## TODOs
- Attach artifacts to Package #5 manifest; sample 10 Trail anchors and log broken links; export `neg_beacon_overrides.csv` and `scenario_playbooks.csv` for the appendix.


---

## Package #6 — W5H & Beacons Scoring + Trails (Ops/Infra surge)
**Doc‑grounded evidence:** Composite elevator score adds **W5H alignment** term with configurable weights; `agrm/snap/trails.py` introduces begin/append/finalize and attaches Trails to reports; `docs/W5H_BEACONS.md` shipped; small seeded‑beacons pass shows positive alignment and promotion bias; Trail IDs present. fileciteturn30file11L3-L12 fileciteturn30file11L16-L25 fileciteturn30file11L30-L42

**Artifacts (to index):** W5H docs; Trails module; modified MDHG/AGRM scoring + iterate passthroughs. fileciteturn30file11L56-L63

**Acceptance/Validation:**
- **V3‑W5H‑001:** `w5h_alignment` surfaced in promotion breakdowns; metamorphic relabel remains green under W5H scoring. fileciteturn30file11L14-L15
- **V3‑W5H‑002:** Beacons registry/extractor present; alignment uses beacon vector vs averaged room W5H; stored in result tags. fileciteturn30file11L12-L13
- **V3‑W5H‑003:** Each run yields a **Trail ID**; Trails attached to reports overlay (drill‑down available). fileciteturn30file11L20-L25

**TODOs (P#6):** Backfill char range & mini‑hash; attach `docs/W5H_BEACONS.md` + sample Trail; plot alignment distributions; wire **Archivist projection** examples (top Trails per beacon). fileciteturn30file11L44-L55

## Index delta — opened P#6
- **Package #6:** W5H/Beacons Scoring + Trails (Ops/Infra surge). Pivot rationale: scoring & Trails work is orthogonal to P#5 TPG/A‑B and introduces new modules/docs + seeded runs.

---

## Package #6 — Trail Viewer v2 & Taxonomy‑enforced stitching (append)
**Doc‑grounded scope:** Trail Viewer upgraded to **v2** with inline expanders, colorized diffs, and taxonomy filters; stitching enforces **family::type** + **package_id** + **policy_hash** across reports. (See Ops/Infra surge notes.)

**Viewer behaviors:**
- Row expander shows Trail header (id, timestamp, package_id, policy_hash) and **lineage pointers** (artifacts, caches, registries, envelopes).
- **Taxonomy filters**: `events`, `results`, `diffs`, `audits`, `staging`, `packages`, `policy_feedback` (observed families only).
- Anchors embedded in HTML reports (A/B, Manifest) resolve directly to viewer sections.

**Stitching rules:**
- A row without `family::type` or `package_id` fails stitch validation.
- `policy_hash` must match the report’s manifest pin; mismatches flagged.

**Validation — P#6 (adds):**
- **V3‑TV2‑001:** Trail Viewer anchors resolve from A/B and Manifest HTML (sample ≥10); expanders render lineage.
- **V3‑TAX‑001:** Every stitched Trail has `family::type` + `package_id` + `policy_hash`; violations logged to E‑DBSU.

**TODOs — P#6 (adds):**
- Add viewer filter presets per package; export `taxonomy_index.json` with counts by family/type.
- Sample 10 anchors per report; attach failures to E‑DBSU with source row.

---

## Package #6 — MDHG scoring panel + inline hashing hooks (append)
**Doc‑grounded scope:** Promotion breakdowns expose **composite score** with the new **W5H alignment** term; MDHG overlay attaches **inline hash** for snapshot identity, enabling quick diffing and replay stitching in reports.

**Score panel fields (as surfaced):** `elevator_score_total, w5h_alignment, recall_like, stability_like, cost_like, policy_penalties`.

**Inline hashing:** MDHG snapshots stamped with a short **hash** and referenced in Trails; report overlays show before/after hashes for elevator proposals.

**Validation — P#6 (adds):**
- **V3‑MDHG‑001:** Promotion breakdowns include `w5h_alignment` and component terms; totals match.
- **V3‑MDHG‑002:** MDHG overlay entries carry inline hash; report overlays show before/after.

**TODOs — P#6 (adds):**
- Backfill a minimal **score panel** example and an **overlay diff** pair (hash A→B) into the package manifest.
- Add a small validator that recomputes `elevator_score_total` from components for 10 random rows.

---

## Final Sweep — completion note
**Status:** Completed linear scan of the remaining tail of D1 (≈ last 50 lines). No new pivots surfaced. **P#6** remains the terminal package for this session; Trail Viewer v2 + taxonomy stitching + MDHG scoring panel/inline hashing close the log. (See P#6 append sections.)

**Backfill markers (kept per protocol):**
- Exact `char_range` for **P#5** and **P#6** → to be stamped in Backfill Protocol.
- `mini_hash(P#4..P#6)` → compute & store during polish.

**Why closed here (doc‑grounded):**
- Trail Viewer v2 + taxonomy‑enforced stitching appear at the end of Ops/Infra surge. fileciteturn30file12L75-L81
- MDHG scoring panel exposure + inline hashing are the final recorded changes. fileciteturn30file11L30-L42

