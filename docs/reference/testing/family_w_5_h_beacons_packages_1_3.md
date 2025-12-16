# Family — W5H & Beacons (Packages #1–#3)

**Linkage:** Running Log Reconstruction (Fresh Pass v3) → Packages #1–#3.

---

## Scope (doc‑grounded)
- Consolidate **Beacon**/**Sub‑Beacon** semantics and the **W5H model** (Who/What/When/Where/Why/How) as integrated in early work.
- Capture the **registry**, **room‑level tagging**, **scoring term**, and **Trails** anchoring described in the log.
- Bind Beacons to **AGRM/MDHG** (room metadata, elevator candidate enrichment) and to **Ops** surfaces (reports, A/B, Iterative).

---

## Components & Behavior

### 1) Beacon Registry
- Central index of **Beacon** and **Sub‑Beacon** definitions; supports namespace (e.g., `docs/*`, `work/*`, `user/*`).
- Emits **registration SNAPs**; collisions quarantined to E‑DBSU.

### 2) W5H Extraction & Tagging
- Extract W5H attributes from candidate items; attach as **room tags** and **Trail facets**.
- `w5h_total` style aggregate used by policy (e.g., DNA policy minimums).

### 3) Scoring Term & MDHG Integration
- W5H contributes a **scoring feature** in Ops evaluations and MDHG **elevator** selection.
- Negative beacon signals (e.g., `neg_beacons`) are captured and enforced in policy.

### 4) Trails & Ops Surfaces
- Every Beacon/Sub‑Beacon tag is written to **Trails**, enabling replay and auditing.
- Ops reports (A/B, Iterative) surface W5H coverage, Beacon hit‑rates, and negative Beacon blocks.

---

## Interfaces (as in log)
- `beacons.register(...)`, `beacons.lookup(...)`
- `w5h.extract(item) -> tags`, `w5h.score(tags) -> float`
- Policy hooks: `dna.min_w5h_total`, `neg_beacons` lists.

---

## Acceptance & Tests
- Registry accepts/denies with SNAP evidence; collision path tested (E‑DBSU).
- Extraction round‑trips across known docs; W5H totals match expected.
- Ops reports include W5H/Beacon sections with counts and deltas.

---

## Risks & Mitigations
- **Over‑tagging / drift:** throttle taggers; require evidence Trails.
- **False negatives in `neg_beacons`:** shadow‑mode first, then enforce.

---

## TODOs (family doc)
- Pull concrete registry entries and example `w5h_total` thresholds from the log; add to `examples/beacons/`.
- Attach sample Ops report slices showing W5H coverage and neg‑beacon actions.
- Cross‑link to DNA policy and controller flags where W5H/Beacons are referenced.

