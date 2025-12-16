# Governance & Deploy-to-Test (DTT)

## Decision Matrix
- **Inputs:** vector score (AGRM stats), failure rate, cost/time, resource fit.
- **Thresholds:** accept if score improves ≥ δ and no regressions in room_len_p95/tail latency.
- **Actions:** accept/run again/reject & fallback (using best known config).

## SNAP Provenance
- Every decision stores: inputs (configs, personas), outputs (stats, score), gate result,
  and links to prior states.

## Compliance Notes
- Ops require continuous approvals; see BLANKET_ALLOWANCE for session-level authorization.
