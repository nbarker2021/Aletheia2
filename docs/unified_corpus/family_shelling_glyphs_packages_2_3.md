# Family — Shelling & Glyphs (Packages #2–#3)

**Linkage:** Running Log Reconstruction (Fresh Pass v3) → Packages #2–#3.

---

## Scope (doc‑grounded)
- Consolidate the **shelling engine** (N‑layer decomposition up to N=5) and **glyph codec** (3‑word base, compress/inflate/invert) as delivered in early work.
- Capture constraints around **inverse variants**, **Weyl 8‑direction context**, and interactions with **Safe‑Cube** and **AGRM/MDHG**.
- Centralize artifacts/docs referenced in the review (buildfiles + Drive docs) for future deep analysis.

---

## Artifacts & Docs (from Packages #2–#3)
- **Docs:** `core_shelling_methodology.md`, `The Complete Shelling Guide.md`, `complete_shelling_guide_master_reference.md`, `advanced_shelling_operations_integration.md`, `glyph_systems_analysis.md`.
- **Bundles:** `buildfiles-…zip` (contains the above and related references).

---

## Components & Evidence (summarized from log text)

### 1) Shelling Engine
- API surface: `shell(meta, max_n=5)` → emits nested shells up to N=5; supports ephemeral expansion for active work.
- Behavior notes: initial iterations prioritize **space creation and bounds**; one‑shot ops only when **full N=4 context** exists.
- Ephemeral lattices are allowed during learning and are later consolidated via **Archivist/SNAP Ops**.

### 2) Glyph Codec
- **3‑word base rule** (ASCII); `compress(...)`, `inflate(...)`, `invert(...)` round‑trip invariants.
- Inverse hooks: every glyph carries positive + inverse semantics for **bidirectional mapping**.
- Registry intent: tie 3‑word glyphs to bridge semantics for pre‑expansion opportunities (deferred to E8/Weyl family).

### 3) Interactions
- **Safe‑Cube:** temp safe region requires tags + policy; glyphs/shell tags travel with SNAPs for gating.
- **AGRM/MDHG:** shell/glyph tags flow into room metadata and elevator candidate enrichment; inverse glyphs appear in enriched elevator payloads.

---

## Invariants & Acceptance (from log)
- **Glyph rule:** Exactly 3 words (ASCII) at N=5 compression; ambiguity beyond 3 flags a lower‑N knowledge gap to resolve before promotion.
- **Round‑trip tests:** `compress→inflate` equality; `invert` sets/clears inverse markers correctly.
- **Dual‑mapping:** Positive and inverse variants must be mapped and tested before marking a package “complete”.

---

## Risks & Considerations
- Ambiguity at N≥5 without sufficient N≤4 context → requires additional shelling or scout passes.
- Glyph collision risks across domains → quarantine to E‑DBSU and demand dual confirmation before promotion.
- Over‑eager expansion of lattices without Safe‑Cube tagging → mannequin Sentinel should deny.

---

## TODOs (family doc)
- Extract concrete examples of shell outputs for N=1..5 from the reviewed docs; attach as appendix.
- Enumerate canonical `compress/inflate/invert` examples with expected outputs; add unit‑style tables.
- Propose a minimal **glyph registry schema** (3‑word base + lineage) synchronized with the E8/Weyl family doc.
- Link shell/glyph tags to MDHG room metadata examples observed in Packages #2–#3.


---

## Package #4 — doc-grounded artifacts & gates (append)
**Artifacts (from D1 evidence):**
- Pilot outputs: `superperm_pilot_greedy_results_v0_28_0_2025_08_14.csv` and `.json` (n=3..6). 
- Baselines bundles: `snaplat_n7_baselines_v0_30_27.zip` (n=7=5908 envelope + stats); `snaplat_baselines_v0_30_28.zip` (n=7=5906 & n=8=46205 envelopes + SNAPs).
- Uniqueness: `snaplat_uniqueness_registry_n7_v0_30_29.json` (canonicalization + structural fingerprints; promotion rules captured).
- Cache: `step_cache.json` (SNAPDNA Step Cache; prefix‑trie hints across strict runs).
- Fixtures: `egans superperm work in full.txt`, `n7egan.txt`, `n8egan.txt` (source sequences; write‑up linkage).

**Gates now reflected in P#4 manifest:**
- **Envelope rule:** n=7 → 5906 **gold** envelope; 5908 **silver**; n=8 → 46205 reference guardrail.
- **Promotion:** perfect + **novel** only (canonical SHA invariant to relabel/rotation/reversal; k‑gram/overlap secondary checks).
- **Step Cache:** deterministic hints only after passing envelope+coverage at a step; cache persisted across seeds.

**Validation updates (P#4):**
- **V3‑SPM‑001:** Pilot CSV/JSON present (n=3..6) — *to verify attach in archivist*.
- **V3‑SPM‑002:** Egan n=7/n=8 envelopes present; coverage/length verified — *attach report refs*.
- **V3‑SPM‑005 (new):** Uniqueness registry file present; promotion rule recorded.
- **V3‑SPM‑006 (new):** Step Cache file present; cache‑hit metrics logged.

**TODOs (delta):**
- Link each artifact above to its package manifest entry and Trails.
- Add a mini “delta report” 5906 vs 5908 (visitation order, overlap hist deltas, envelope deltas) as a Package #4 appendix.

---

## Package #4 — metrics schema & Trails bindings (append)
**Pilot metrics table (per‑run, per‑n):**
- `n`, `seed`, `builder_mode` (greedy|beam|sss), `length`, `coverage_pct`, `imperfect_join_rate`, `overlap_hist` (json), `runtime_ms`, `mem_mb`, `envelope_ok` (bool), `uniqueness_ok` (bool), `promoted` (bool), `trail_id`.

**Trails bindings:**
- Each pilot/export writes a **Trail** with lineage to: fixtures (Egan), envelope bundle, uniqueness registry, step cache, and the package manifest.
- A **delta Trail** captures 5906 vs 5908 comparisons (length, coverage, overlap deltas; visitation order sketch).

**Validation (adds):**
- **V3‑SPM‑007:** Every pilot row maps to a saved Trail id; missing Trail ⇒ validation fail.
- **V3‑SPM‑008:** Delta Trail present for 5906 vs 5908 bundles; includes at least length/coverage/overlap deltas.

**TODOs (adds):**
- Generate `pilot_metrics.csv` from logs and attach to Package #4 manifest.
- Emit `delta_5906_vs_5908.json` + `delta_report.html` under Package #4 appendix.

---

## Cross‑links
- **Family canvases:**
  - *Superpermutation Pilot & Test Plan (P#4)* — detailed methodology, acceptance, failure taxonomy.
  - *MDHG & AGRM (P#1–#3)* — governance spine + materializers + controller/MDHG integration.
  - *Shelling & Glyphs (P#2–#3)* — N‑layer + glyph codec + invariants.

