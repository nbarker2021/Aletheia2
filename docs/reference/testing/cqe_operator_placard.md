# CQE Operator Placard (One Page)

## Inputs
- **Pre**, **Metric**, **Stations: Measure → Tag → Resume**
- Tokens: pick 16; select an **Octet** (V1..V8) that spans Stations + Views.

## Procedure
1) **Octet**: Assign tokens to V1..V8.
2) **Mirror**: Check (V1,V8), (V2,V7), (V3,V6), (V4,V5). Write a one-line rationale for each pair.
3) **Strict**: Impose **Strict Bounds**; forbid **glue**; require **witness**.
4) **Δ-lift**: Make a single admissible lift consistent with bounds + witness.
5) **Ledger**: Mark **Views exercised (1..8)**, update **bounds (old→new)**, write **4-bit receipt**, page **hash/status**.

## Receipts
- **4-bit**: four-bit sticker/field (binary). Deterministic commit allowed if no coins:
  `commit = first-nibble(SHA256(seed))`.

## Observers (J/Q/K)
- J: metrics; Q: view priority; K: strictness.

**Mantras**: No Glue • Witness Everything • Tighten Bounds • Commit or Don’t Ship
