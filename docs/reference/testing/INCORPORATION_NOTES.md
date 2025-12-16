# CQE Incorporation — Reference (Strong) Items

This bundle contains per-item **kernel specs** (`kernels/kernel_{id}.json`), a **benchmark pack** (`benchmarks.jsonl`),
and a **strong_ingestion_plan.csv** that assigns owner teams and tracks status.

## How to incorporate
1. **Load kernel JSON** for each id and register it in the CQE harness.
2. **Replay OPE→FCE** using the seed recorded in `benchmarks.jsonl`. Confirm thresholds.
3. **Emit overlays** (parity/octet/16h) and attach receipts to the ledger.
4. If a `transfer_rail` is present, run the **cross-rail replay** and log results.
5. Flip `status` in `strong_ingestion_plan.csv` from `TODO` → `IN_PROGRESS` → `DONE`.

## Rails → Teams
- QD → Photonics & Quantum Control
- LC → Optics & Metasurfaces
- BIO → Bio/Coherence
- COSMOS → Sky & Fourier-Lensing

— Generated automatically.
