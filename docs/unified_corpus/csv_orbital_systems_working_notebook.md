# CSV Orbital Systems — Working Notebook (v0)

## Scope & Source
You said the **Csv orbital system** zip contains 20 text files that together form one “book”: **“New csv methods”** is the preface; all others are chapters to be used together. Additional zips (**Geosys data**, **New geometry based ideas**, **logsinorder**) provide context, experiments, and logs. We’ll treat the four zips as the active corpus.

## Index (from files)
*I parsed the zips and discovered a clean A–S chapter set with a few uppercase/typo variations.*

- Preface — Final Rule & Governance Pack v0.1 — Internal‑Ops Draft
- Doc A — Concepts, Goals, and System Shape (Deep Guide)
- Doc B — Formal Objects & Naming (Field‑Level Spec)
- Doc C — CSV Artifacts & FK Maps (Full Tables + Indexing)
- Doc D — Legality & Deterministic Acceptance (Guardrails & Proofs)
- Doc E — Geometry & Octadic Checks (Metrics, Clocks, Scoreboards)
- Doc F — Futures, Scoring, and Controllers (DFSS Minimax)
- Doc G — Graph, Orbitals, and Seeds (Nodes/Edges, Orbits, Seeds)
- Doc H — LAEC, TLattice, and Buffer Gate (Energy, Lattices)
- Doc I — Evidence, Audit, and Traceability (RAG + Trails)
- Doc J — Observer & Role Model (Modes, Watchers, Reviewers)
- Doc K — Master vs Orbit; MGL; Prompt‑Cursor Recall (States, Glyph Governance, Recency)
- Doc L — Runbooks & Change Control (Procedures, Watchers, Rollback)
- Doc M — System Boundaries & Stability Windows (Limits, Windows)
- Doc N — Implementation Cheat‑Sheets & Repo Starter Spec (CSV‑First Runtime)
- Doc O — “Live” CSV Runtime Protocol (IO, Ticks, Buffers)
- Doc P — IRL Systems Mapped to Our Process (2024–2025 Sources)
- Doc Q — Adapters & Bridges (Ingress/Egress, Normalizers)
- Doc R — Risk & Failure Modes (Stress, Timeouts, Traps)
- Doc S — Technical Support Addendum (Base‑320, Bits, Braid/Helix)

> Supporting corpora surface superpermutation/geometry themes: Dual‑Octad, braiding, octadic clocks, NGA program, etc. These clearly inform Docs E/F/K and the runtime expectations in N/O.

## Initial Observations
- **Structure is coherent**: preface + 19 chapters; internal cross‑references are dense. Docs **L**, **N**, **P**, **K**, **S** act as hubs.
- **Naming drift**: one file uses `Cdv doc S` instead of `Csv` and a few letters are uppercase. We can normalize to **Doc A…S** without renaming the originals.
- **Runtime posture**: CSV‑first runtime + observers/watchers + DFSS minimax controller; implies deterministic replayability and auditable trails (Docs F, I, L, N, O).
- **Geometry layer**: octadic metrics/scoreboards, energy/lattice primitives, seed/orbit mechanics (Docs E/G/H/K) line up with your Dual‑Octad SOI approach.

## Plan of Attack
1. Stitch a master “Book” (concatenated, neutral order) with section headers set from each document’s internal title — **no canonical order assumed**.
2. Per‑document deep read & synopsis (objective, contracts, invariants, interfaces, examples, TODOs), guided by each doc’s own title/operation.
3. Cross‑ref map across **all four zips** (chat logs + summaries + specs). Build a directed graph of references and trace chains.
4. Canonical object model (from recurring fields across docs); emit CSV + JSON Schema; flag naming drift.
5. Geometry spec: formalize octadic metrics/scoreboards/clocks using evidence from *New geometry based ideas* and *Geosys data*.
6. Runtime spec: consolidate protocols, ticks, buffers, observers/watchers; derive an executable CSV‑first harness.
7. Adapters & bridges: enumerate sources/targets; produce adapter templates.
8. Assurance: acceptance proofs, audit trails, failure modes → test harnesses + checklists.
9. Ops kit: runbooks, rollback, change‑control gates; cheat‑sheets.
10. Finalize v0.1 spec + glossary + changelog.

## Open Decisions (please confirm)
1. **Canonical order**: OK to enforce Preface, A→S as the official order?
2. **Normalization**: OK to standardize titles (e.g., fix `Cdv doc S` → `Doc S`) while preserving originals?
3. **Output shape**: prefer a single consolidated Markdown/PDF spec, or keep per‑chapter files + an index? (I can generate both.)
4. **Implementation target**: which environment should the CSV‑first runtime examples assume? (e.g., Python CLI + plain CSV on disk.)
5. **Priority**: if we have to go deep first, do you want me to start with **Docs L/N/O** (runtime & ops) or **E/F/K** (geometry & scoring)?

## Artifacts Generated (v0.3)
- **Consolidated book (draft)** — concatenation of the CSV orbital system texts with titles preserved. Not an authoritative order.
  - [Download](sandbox:/mnt/data/csv_orbital_system_book_draft.md)
- **Corpus index (CSV)** — every file across all zips with title/kind/stack bucket.
  - [Download](sandbox:/mnt/data/corpus_index.csv)
- **Cross‑references — authoritative (CSV/JSON/DOT)** extracted from each document’s **footer linkage** (tail scan of “Document X — …” entries). This supersedes the naive token approach.
  - [CSV](sandbox:/mnt/data/crossrefs_authoritative_tailscan.csv)
  - [JSON](sandbox:/mnt/data/crossrefs_authoritative_tailscan.json)
  - [GraphViz DOT](sandbox:/mnt/data/crossrefs_authoritative_tailscan.dot)
- **Operational stack map (CSV)** — suggested buckets per file (navigation aid only).
  - [Download](sandbox:/mnt/data/operational_stack_map.csv)

## Audit & Trace
- Updated the pipeline to **prioritize the in‑document linkage footers**; we now parse the tail sections and extract explicit `Document [A–S]` references. Original files remain untouched; derived artifacts note their method.
- Deprecated the earlier naive token cross‑ref for analysis sequencing (kept only for debugging).

## Next Steps
- Start per‑document deep reads; I’ll flag when a document’s *title‑implied operation* suggests a different stack placement.
- Upgrade cross‑ref graph to typed links ("builds on", "adapts", "audits", etc.) and generate path traces from chat logs → summaries → specs.
- Synthesize the canonical schema + harness skeleton, then back‑fill with examples from the logs.

