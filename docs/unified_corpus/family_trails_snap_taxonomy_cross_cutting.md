# Family — Trails & SNAP Taxonomy (Cross‑cutting)

**Linkage:** Running Log Reconstruction (Fresh Pass v3) → Packages #1–#4.

---

## Scope (doc‑grounded)
- Centralize the **SNAP family/type taxonomy** observed in the log and the **Trails** primitives used for lineage, replay, and audits.
- No invention: enumerate only families/types explicitly used; expand as new ones appear in later packages.

---

## SNAP Taxonomy (observed so far)
- **Events:** `mdhg_event::elevator`
- **Results:** `result::elevator`
- **Diffs:** `universe_diff::overlay`
- **Audits/Reports:** `freshstart::…`, `iter_report::…`, `promotion_summary::…`
- **Staging:** `staging::candidate`
- **Packages:** `package::idea` (Idea Package manifests)
- **Policy Feedback:** `policy_feedback::superperm` (planned; created when notes are attached)

> This list grows only when the running log shows new families/types.

---

## Trails — model
- **Trail** = immutable record with lineage pointers to artifacts (files/bundles), policies (`policy_hash`), caches, registries, and package manifests.
- **Delta Trail** compares artifacts/runs (e.g., 5906 vs 5908 envelopes) and stores metrics deltas.
- Trails are referenced in reports and used by Ops for quick drill‑downs.

### Required fields (per Trail)
- `id`, `timestamp`, `package_id`, `family::type`, `policy_hash`, `artifacts[]`, `metrics{}`, `parents[]`, `notes`.

---

## Bindings (as in log)
- Pilot runs (P#4) → one Trail per run; a **delta Trail** for 5906 vs 5908.
- FreshStart/Iterative → Trails carrying run parameters and outcomes.
- Promotion → Trails with uniqueness decisions and cache deltas (planned in P#4 TODOs).

---

## Acceptance & Tests
- Every reported table row references a Trail id (validators in P#4).
- Delta Trails include at least length/coverage/overlap deltas and sources.
- Trails survive repo resets (FreshStart) via re‑attach process.

---

## Risks & Mitigations
- **Trail sprawl:** use package‑scoped manifests and roll‑ups; archive cold Trails.
- **Policy drift:** enforce `policy_hash` pinning in Trails and reports.

---

## TODOs (family doc)
- Backfill examples for each observed family/type from the log with Trail ids.
- Add a `trail_schema.json` example file and a `delta_trail_schema.json` example.
- Cross‑link to Ops reports and the Package #4 appendix once generated.


---

# Package #5 — TPG Bridges & Proposals + Golden Runner (telemetry‑only)
**Char range:** *(pending; set during backfill)*  
**Mini‑hash:** *(pending)*  
**Pivot (in):** TPG orchestrator enabled; **Bridge SNAPs** + **Proposal SNAP** emitted; Ops reports + A/B harness online; negative‑beacon penalties validated; Trail viewer anchors.  
**Topic anchors (auto):** tpg, bridge_snaps, proposal_snaps, telemetry_only, ops_report, ab_runner, golden_runner, neg_beacons, trail_viewer  
**Weyl coverage:** pos=0/8, inv=0/8 (init)

**Doc‑grounded evidence:**
- TPG integration + telemetry only; config block; Trail event logging; Bridge/Proposal SNAPs; demo params & artifacts. fileciteturn28file2L9-L23 fileciteturn28file0L30-L41 fileciteturn28file0L47-L55
- v0.28.1 Bridges+Proposals demo: patch + ops reports (JSON/CSV). fileciteturn28file0L16-L26
- Golden Runner FAST/A/B and expanded v0.29.1 batteries; artifacts & results. fileciteturn28file13L49-L71 fileciteturn28file8L15-L24
- A/B report with Trail drill‑down, deltas, neg‑beacon penalties made reliable (v0.30.5→0.30.6). fileciteturn28file12L7-L16 fileciteturn28file12L70-L76

**Artifacts (P#5 index):**
- `snaplat_patch_v0_28_1_2025_08_14.zip`; `snaplat_ops_report_v0_28_1_tpg_bridges_demo_2025_08_14.(json|csv)`; `snaplat_full_v0_29_1_2025_08_14.zip`; A/B bundles `snaplat_ab_v0_30_5_2025_08_14/*`, `snaplat_ab_v0_30_6_2025_08_14/*`. fileciteturn28file0L18-L26 fileciteturn28file8L27-L35 fileciteturn28file12L7-L15

**SNAP families/types (observed):** `bridge::tpg_bridge`, `proposal::tpg_order` (telemetry only); `iter_report::ab_results`, `iter_report::ab_deltas`. fileciteturn28file0L30-L35 fileciteturn28file12L11-L16

**Validation — P#5**
- **V3‑TPG‑001:** When `tpg.enabled=true`, each iterate logs a Trail event (`event: tpg`, cost, imperfections). fileciteturn28file2L20-L23
- **V3‑TPG‑002:** Bridge SNAPs emitted for low‑overlap edges; Proposal SNAP saved with ordered room IDs (telemetry‑only). fileciteturn28file0L30-L35
- **V3‑TPG‑003:** Ops report artifacts present for v0.28.1 demo (JSON/CSV). fileciteturn28file0L22-L26
- **V3‑TPG‑004:** Golden Runner A/B artifacts present; Trails JSON links resolve; deltas computed. fileciteturn28file12L7-L16
- **V3‑TPG‑005:** Neg‑beacon penalties wired and validated in A/B; config honored through builder. fileciteturn28file12L70-L76

**TODOs — P#5**
- Backfill char range + mini‑hash; attach `tpg_config` snapshots to package manifest.  
- Emit **Bridge‑aware QA** (telemetry): attach rationale templates + evidence to each Bridge SNAP; Trail pointer. fileciteturn28file0L65-L73  
- Canary **proposal‑as‑hint** (soft) path and log promotion deltas (still policy‑gated). fileciteturn28file4L36-L42  
- Expand **neg‑beacon** matrix; sweep penalties and record leakage deltas. fileciteturn28file8L68-L74

