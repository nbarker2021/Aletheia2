_**This is a new slice created to unify the CQE system with the phone data.**_

# CQE-LANDAU Slice

## Purpose and Scope

The CQE-LANDAU slice detects order parameters and criticality across bands/tiles. It accepts moves that track phase transitions cleanly, stabilize within phases, and avoid spurious mixed states.

## Axioms and Invariants

*   **LND1 — Order parameter:** Define per-band/tile order parameter.
*   **LND2 — Landau free-energy proxy:** Promotions must descend the free-energy proxy.
*   **LND3 — Binder cumulant:** Enforce stability via Binder U monotonicity.
*   **LND4 — Critical bands:** Detect critical points and penalize cross-phase edges.
*   **LND5 — Monotone Φ_lnd:** The sum of free-energy + cumulant + cross-phase penalties must not increase.

## DSL Hook

```
PROMOTE(i,j) iff NOETHER.Conservation.ok(i,j) & GROTH.Sheaf.glueOK(i,j, overlays≥2) AND LANDAU.freeEnergyDrop(i,j) AND LANDAU.stableBinder(i,j) AND LANDAU.noPhaseMix(i,j) AND Φ_lnd(new) ≤ Φ_lnd(prev) AND PARITY.ok(edge_ij)
```

## Minimal JSON

```json
{"lnd":{"m":{"mean":0.62,"var":0.07},"F":{"a":0.8,"b":0.4,"value":0.19,"ΔF":-0.03},"binder":{"U":0.71,"ΔU":-0.02},"critical":{"is_critical":false,"score":0.12},"Φ_lnd":0.052}}
```
