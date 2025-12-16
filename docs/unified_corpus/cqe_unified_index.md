# CQE Session Unified Pack (v1)
Generated: 2025-10-10T20:50:26Z

## Files
- cqe_unified_manifest.json
- cqe_unified_harness.yaml
- cqe_bucket_nav.json
- cqe_session_ledger.jsonl

## Slice Registry
- **MORSR_GAUGE_POSE** — spec_available (deps: POSE, GOV) — PEPS/IPS/PaG; ledger-first; pose-aware cache.
- **DYNATOKE** — active (deps: GOV, LATT, FRAC, CRT) — lanes, ΔΦ gating, shell ×10, token conservation.
- **LATT** — active (deps: SACNUM, CRT, FRAC) — E8/Leech, Construction A–D, Hamming gating, Conway lifts.
- **FRAC** — active (deps: LATT, SACNUM, GOV) — Julia/Mandelbrot charts, μ-banding, K=16 bins.
- **SACNUM** — active (deps: GOV, LATT) — Digital-root, PD80/PS160, frequency labels.
- **CRT** — active (deps: SACNUM, LATT, GOV) — Residue frames, tile4096, Garner proofs.
- **RAMANUJAN** — available (deps: CRT, LATT, FRAC) — q-series, partitions, eta/theta, mock-theta shadows.
- **GAUSS** — available (deps: CRT, FRAC, LATT) — Reciprocity gates, Dirichlet χ, Gauss sums, BQF/ℤ[i].
- **TRANSLATE_RT** — available (deps: LATT, FRAC, GOV) — E8→Λ24→E8', pack-constrained realization.
- **ACOUSTIC** — available (deps: TRANSLATE_RT, GOV) — GGwave-like PHY, S0–S2 shells.
- **OPS_CSV** — active (deps: GOV, LATT, FRAC, CRT) — Delimiter/Type/Insert repairs.
- **OPS_JSON** — active (deps: GOV, LATT, FRAC, CRT) — Brace/Type/Schema repairs.
- **TORUS** — test (deps: GOV, LATT, FRAC, CRT) — C(k) closure; k* detection.
- **POSE** — enforced (deps: GOV) — Global-once; pose-aware cache keys.
- **GOV** — enforced (deps: ) — ΔΦ, CBC, escrow, ethics, channel entropy.
