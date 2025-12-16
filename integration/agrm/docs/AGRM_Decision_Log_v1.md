# AGRM — Selective Adoption & Reimplementation (v1)

## Final-build modules (Adopt)
- `unified/integration/agrm_unified_adapter.py` — **ADOPT** (reimplemented cleanly to remove truncations; deterministic, no heavy deps)
- `unified/agrm_unified/runner.py` — **ADOPT** (extended with `run_batch` helper; deterministic by seed)

## Franken modules
- `src/unified/franken/agrm_franken.py` — **REFERENCE ONLY** (referenced in docs; not present in tree)
- Other `franken/*` — **REFERENCE ONLY** unless materializes in tree with intact code

## Legacy “bestof AGRM”
- `_bestof_unpack/bestof_agrm/agrm/AGRM.py` (large, partial truncations, heavy deps) — **IDEA-ONLY REIMPLEMENTATION**
  - Use for design/algorithm reference only
  - Do **not** import; port minimal functions when specifically needed

## Why this selection
- We prioritize **final-build** cohesion and maintainability.
- Franken is treated as a **parts bin**; only intact, low-risk segments would be eligible (none present for AGRM).
- Legacy bestof contains useful logic but carries truncation and dep risk; safer to **reimplement** minimal pieces matching current interfaces.

## Changes made in this pass
- Replaced truncated adapter with a clean implementation that expands `(family × k × count)` and builds snapshots via `runner.run_once`.
- Added `run_batch(params_list, seed)` to runner for deterministic multi-run behavior.

## Next
- If AGRM requires advanced routing/scoring, identify the minimal algorithmic fragment in `bestof_agrm/agrm/AGRM.py` and **reimplement** it as a focused, tested function under `unified/agrm_unified/`.
