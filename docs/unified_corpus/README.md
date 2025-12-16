
# Universal State Format — Master Codex, Lexicon, Beacons (v0.1)

This package defines three JSON Schemas and helpers that plug directly into E8-AS and SNAP v2:

- `master_codex_v0.1.json`: live terms/defs/functions with E8-AS binding
- `lexicon_v0.1.json`: Python-compatible labels with metadata tags
- `beacon_v0.1.json` + `w5h_vector_v0.1.json`: top/sub-level checkpoints with W5H weighting

**Integration**

- Each object carries `e8` (E8-AS) and `axes` fields to align with MDHG identity and SNAP routing.
- When persisted, wrap the object in a SNAP v2 manifest with `kind` one of: `MicroOp`, `PolicyMatrix`, `PersonaDNA`, `Dataset`, etc.
- Beacons emit scores via `lib/w5h_scorer.py`, which you can record in `manifest.metrics`.

**Next steps** (Phase 1):
1. Enforce these schemas in the service write paths.
2. Emit a `Master Codex` SNAP each turn with terms/defs learned/updated.
3. Attach `Beacon` evaluations to promotion decisions and persona training sessions.
