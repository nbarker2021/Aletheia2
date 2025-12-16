# Operations Manual

## Session Setup
1. Clone or place `code/` and `snap_system/` (if used) on a writable machine.
2. Ensure Python 3.9+ available.
3. For SNAP-only flows, run `snap_system/run_demo.py` to validate env.

## Typical Workflows
### A) Persona Learning
- Ingest sources (start with ArXiv & top labs).
- Save `ArXivPaper` + `PaperDigest` + `AlgorithmCard`.
- Create `SNAPDNA` personas; compose hybrids if task spans domains.
- Route tasks via MoE; update persona score based on outcomes.

### B) AGRM Sweeps
- Set `AGRMConfig`; run `AGRMController.run_sweeps()`.
- Review `stats_*`, `trace_*`, `best_config.json` in snapshots.
- Promote best config to next run.

### C) Error Handling
- On failure, snapshot `ErrorTrace` with minimal repro.
- Apply `PlaybookStep`; verify fix; append to FunctionLifecycle.

### D) Code Reuse
- Save minimal `CodeUnit` for utilities you write.
- Stitch to build larger blocks; add tests before saving.

## Backup / Restore
- ZIP `snap_store/` and `docs/` periodically.
- Export/import AGRM states with `save_state` / `load_state`.

## Safety & Constraints
- Keep playbooks purely declarative in v0 (no arbitrary code exec).
- Validate external inputs before saving as SNAP states.
