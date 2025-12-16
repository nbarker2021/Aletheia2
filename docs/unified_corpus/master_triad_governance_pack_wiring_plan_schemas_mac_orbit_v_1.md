# Master Triad + Governance Pack — Wiring Plan & Schemas (MAC‑ORBIT v1.1)

A deterministic two‑tier system with a **Master Triad** of CSVs and an optional **Governance Pack**. Orbit remains the working shell; promotion into Master is explicit and auditable. All pieces respect BW‑32 windows, Base‑320 packets, LAEC, Buffer Gate, and glyph discipline. This doc wires your refinements into the prior MAC‑ORBIT design.

---

## 0) Topology Overview
- **Master Triad** (persisted, non‑ephemeral)
  1. **MASTER_GPT_TRUTH.csv** — GPT‑determined truths to context/IRL law governance
  2. **MASTER_USER_TRUTH.csv** — user‑defined truths to context/beliefs; alignment flags vs IRL law/theory
  3. **MASTER_DOCUMENT_INTAKE.csv** — canonicalized content of any external document (web/user uploads); not native internal calls

- **Orbit (working shells)**
  - `orbit.csv` — all provisional atom cards (meaning + embeddings) with promotion metadata.

- **Governance Pack (optional, switchable)**
  - **LAWS_VALIDATION.csv** — all laws, validations/invalidations, observation tools
  - **GREY_EVIDENCE.csv** — accepted but not fully validated reference materials (strong but non‑decisive)
  - **VALID_MULTI_D_BRIDGE_LOG.csv** — cross‑step (≥4) valid bridges
  - **GLYPHIC_MASTER_LOG.csv** — glyph registry & fast bridge links
  - **SYSTEM_OPS_GUIDE.csv** — processes, math specs, examples for proofing
  - **TESTBED.csv** — candidate datasets or cases to test unclear/underperforming parts
  - **TEST_RESULTS.csv** — 5W+H logs for every test; joins to TESTBED

- **Master Detail Shell (optional)**
  - `MASTER_DETAIL_SHELL.csv` — lower, permanent shell biased toward granular details of master inclusions (for high‑precision recall)

- **Master Glyph Language (MGL)**
  - `MGL_REGISTRY.csv` — reserved glyph items/digits usable *only* by Master data; third‑party glyphs that reuse any MGL item trigger overlap checks.

---

## 1) Shared Identity & Hashing (all files)
Deterministic, audit‑grade IDs:
- `run_id` — UUIDv5 over sorted input file names + config hash
- `atom_id` — UUIDv5 over canonical atom payload (minus IDs/hashes)
- `doc_id, chunk_id, span_start, span_end` — provenance to text
- `glyph_id` — SHA256 over canonicalized triad items + spec version
- `context_hash` — SHA256 over context source text/law
- `ledger_hash` — rolling step hash (append‑only); optional `merkle_root` per window
- `window_id, packet_id` — BW‑32 and Base‑320 mapping

Unicode NFC normalization and whitespace trimming applied before hashing. All rows append‑only.

---

## 2) Atom Card Model (used in Master & Orbit)
Common fields (superset; unused may be blank):
- **Provenance**: `run_id, atom_id, doc_id, chunk_id, span_start, span_end, created_at, operator_id, config_hash`
- **Type**: `card_type ∈ {meaning, embedding}`
- **Semantics**: `primitive, meta_bucket, parity_set`
- **Context Truth**: `context_id, context_label, truth_value ∈ {T,F,U}, context_source ∈ {native, uploaded, IRL}, context_hash`
- **Glyph Triad**: `glyph_a, glyph_b, glyph_c, glyph_id, glyph_spec_version`
- **Meaning Payload**: `claim_text, quote_text, confidence ∈ {high,med,low}`
- **Embedding Payload**: `embed_space_id, embed_dim, embed_encoding ∈ {float32_csv,sparse_json,b64pack}, embed_values|vector_id, embed_spec_hash`
- **Governance**: `laec_steps, s_acc, r_back, b_exp, e_br, d_dup, tlattice_id, stabilized, buffer_Hc, constraints_applied, ledger_hash, step_index, window_id, packet_id`

---

## 3) Master Triad Schemas

### 3.1 MASTER_GPT_TRUTH.csv
Adds to base atom:
- `master_state ∈ {accepted, deprecated}`
- `consensus_level ∈ {doc, corpus, global}`
- `evidence_strength ∈ [0,1]`
- `stability_score ∈ [0,1]`
- `agreement_rate ∈ [0,1]`
- `origin_orbit_ids (json list)`
- `last_reviewed_at, reviewer_id`

**Rules**
- Only **accepted** promotions from Orbit with PS ≥ θ_accept.
- If `context_source=IRL`, `context_hash` must point to a law/definition row in LAWS_VALIDATION.

### 3.2 MASTER_USER_TRUTH.csv
- Inherits atom fields; adds:
- `user_id`
- `alignment ∈ {aligned, misaligned, unknown}` (vs IRL/theory)
- `alignment_basis` (law IDs or theory IDs)
- `challenge_status ∈ {open, resolved}`

**Rules**
- User truths never override IRL automatically; promotion requires adjudication (PS with human_signal weight).

### 3.3 MASTER_DOCUMENT_INTAKE.csv
- Minimal per‑chunk canonicalized text registry:
- `doc_id, chunk_id, offset_start, offset_end, text_hash, text_excerpt (<= 240 chars), source_uri|upload_id, ingest_time`
- `shell, bucket_id, glyph_triad, ledger_hash`

**Rules**
- Only external sources (no native internals). Text itself may live outside (hash‑addressed) to keep CSV light.

---

## 4) Orbit Working File — orbit.csv
Extends base atom with promotion/conflict:
- `orbit_id, shell, bucket_id, octet_id`
- `promotion_score (PS) ∈ [0,1]`
- `promotion_status ∈ {proposed, accepted_to_master, rework, retired}`
- `conflict_set (json list of atom_ids)`
- `recency_rank (int)`  

**PS Formula**
`PS = w_e*evidence_strength + w_s*stability + w_c*consistency + w_r*recency + w_h*human + w_g*agreement − w_k*cost`
(Weights & thresholds in ops.yml.)

---

## 5) Governance Pack Schemas (optional)

### 5.1 LAWS_VALIDATION.csv
- `law_id, title, jurisdiction|domain, text, source_ref, observation_tools (json), validity ∈ {valid, invalid, contested}`
- `applies_to (contexts list), last_checked_at, checker_id`

### 5.2 GREY_EVIDENCE.csv
- `evidence_id, description, source_ref, strength ∈ {low, med, high}, relevance_tags, notes`

### 5.3 VALID_MULTI_D_BRIDGE_LOG.csv
- `bridge_id, start_atom_id, end_atom_id, hop_count (>=4), path (json list of atom_ids), legality ∈ {valid, invalid}, saved_steps (int), saved_windows (int)`
- `justification, created_at`

### 5.4 GLYPHIC_MASTER_LOG.csv
- `glyph_id, glyph_a, glyph_b, glyph_c, spec_version, scope (primitive|meta_bucket|context), collisions (json), related_bridges (json)`

### 5.5 SYSTEM_OPS_GUIDE.csv
- `op_id, process_name, rationale, math_spec_ref, example_ref, version`

### 5.6 TESTBED.csv
- `case_id, hypothesis, input_refs, expected_behavior, priority, owner`

### 5.7 TEST_RESULTS.csv
- `case_id, when, who, where, what, how, outcome, artifacts_ref, notes`

### 5.8 MASTER_DETAIL_SHELL.csv (optional)
- `detail_id, master_atom_id, fine_grain_primitive, micro_context, glyph_ref, notes`

### 5.9 MGL_REGISTRY.csv
- `mgl_id, item_token, class (word|symbol|notation), semantic_scope, reserved_by (master_atom_id or primitive), added_at`
- **Policy**: any Orbit atom using an `item_token` present here triggers an **overlap audit**; if compatible, it can inherit semantics, else flagged.

---

## 6) Bit‑Window & Packet Mapping
- **P1 (Native)**: write `MASTER_DOCUMENT_INTAKE.csv` rows; create Orbit atoms (TLattice if outliers; LAEC accrual).
- **P2 (RAG)**: build meaning cards; evidence strength; update Orbit.
- **P3 (ALT)**: explore variants; compute stability/consistency; update PS.
- **P4 (Output)**: promote Orbit→Master where `PS ≥ θ_accept`; write deltas; emit operator summaries & test prompts.
- Ten windows = **one Base‑320 packet**; log `packet_id` across all changed rows.

---

## 7) Promotion Pathways
1) **Orbit → MASTER_GPT_TRUTH** (auto if PS high, laws satisfied)
2) **Orbit → MASTER_USER_TRUTH** (user assertions; PS includes human_signal)
3) **MASTER_USER_TRUTH → MASTER_GPT_TRUTH** (after adjudication; creates a deprecated record in USER_TRUTH and a new accepted record in GPT_TRUTH, linking both ways)

Demotions create `deprecated` flips with ledger entries; no deletion.

---

## 8) Prompt‑Cursor Recall
- Maintain `cursor_id = sha256(latest_prompt)`
- Rank Orbit atoms by recency match to cursor primitives/keywords → `recency_rank`
- Retrieval order: `Orbit@cursor → Master → Orbit@rest`
- On promotion, store `cursor_id` in `origin_orbit_ids` for traceability.

---

## 9) Master Glyph Language (MGL) — Rules
- Canonical triad‑of‑items; each item 1–3 tokens; NFC normalized; separators escaped.
- Reserved items live in `MGL_REGISTRY.csv`; only Master data may introduce new MGL items.
- Overlap policy: if Orbit uses an MGL item, run **overlap audit** — inherit semantics if consistent; else block promotion and log conflict in `GLYPHIC_MASTER_LOG.csv`.

---

## 10) Ops.yml Toggles (deltas)
```yaml
master:
  triad: {gpt_truth: true, user_truth: true, document_intake: true}
  detail_shell: {enable: true}
  mgl: {enforce: true}

governance_pack:
  enable: true
  logs: {laws: true, grey: true, bridges: true, glyphs: true, ops: true, testbed: true, results: true}

promotion:
  ps_weights: {evidence: 0.25, stability: 0.2, consistency: 0.15, recency: 0.1, human: 0.1, agreement: 0.1, cost: 0.1}
  theta_accept: 0.65
  theta_retire: 0.35

mgl:
  overlap_audit: strict   # strict | warn | off
```

---

## 11) Script Sketch (single‑file flow)
```python
# mac_orbit_master.py (sketch)
# 1) ingest external docs -> MASTER_DOCUMENT_INTAKE + orbit atoms
# 2) build meaning cards -> orbit updates
# 3) compute stability/consistency -> PS
# 4) promote to MASTER_GPT_TRUTH / MASTER_USER_TRUTH
# 5) update governance pack (laws, bridges, tests)
# 6) emit operator summary (RAG cards + promotions + LAEC + windows)
```

Each step is deterministic, appends rows, updates `ledger_hash`, and (optionally) a Merkle root per window.

---

## 12) Why this matches your intent
- Master is the **only** non‑ephemeral store; Orbit remains flexible and exploratory.
- The latest prompt becomes the strongest entrance to context recall without overriding IRL law or Master consensus.
- MGL reserves stable glyph semantics while still letting Orbit invent new glyphs—safely.
- The Governance Pack externalizes laws, grey materials, test scaffolding, bridges, and ops math so every promotion is explainable and reversible.

