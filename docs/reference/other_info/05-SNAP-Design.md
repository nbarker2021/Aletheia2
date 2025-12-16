# SNAP — Persistent Typed State & Personas

## Core
- `SNAPCore`: register kinds, validate, save/load/query; JSON source of truth.
- `FileStorage`: key→file mapping for transparent inspection and versioning.

## High-Impact Kinds
- **ArXivPaper**, **PaperDigest**, **LLMProfile**, **AlgorithmCard**
- **CodeUnit (SNAPBIT)** with tests & deps
- **ErrorTrace** and **PlaybookStep**
- **AnyFile** for arbitrary single-file capture

## Personas
- **SNAPDNA** (domain/level, sources, weights, routing_tags, tools, score)
- **HybridSNAPDNA** (weighted composition)
- Score updates via EMA from downstream outcomes (sweeps/evals).

## Routing (MoE v0)
- Route by tag/tool overlap + persona score; returns top‑K persona IDs.

## Stitcher (v0)
- Merge matching CodeUnits; de‑dup imports; aggregate tests; return a stitched CodeUnit.
- v1 roadmap: AST-based compose, import resolver, style/lint, test-runner.

## Playbooks
- Match ErrorTrace via boolean `when` expressions;
- v0 annotates; v1 will execute sandboxed actions and run `verify` checks.

## Ingestion (stub)
- `web_ingest.py` demonstrates ArXiv fetch → `ArXivPaper` SNAP → persona seeding.
