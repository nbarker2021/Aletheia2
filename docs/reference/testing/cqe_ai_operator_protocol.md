# CQE — Self-Defining Operator Protocol (derived from materials)

This protocol is inferred exclusively from the CQE documents by harvesting nearby instructions around core concepts.

## 1) Inputs
- Select tokens from Stations/Views.
- Use **Pre**, **Metric**, and **Measure/Tag/Resume** to establish the measurement frame.
- Dock a card; write token values; attach sticky labels.

## 2) Process
- Form the **Octet (V1..V8)**; overlay if specified by Loom/Overlay.
- Run **Mirror** on pairs (V1,V8), (V2,V7), (V3,V6), (V4,V5).
- Apply **Strict**: tighten thresholds; forbid **glue**; require **witness**.
- Perform **Δ-lift** only with admissible steps under bounds.

## 3) Outputs / Evidence
- Record **Mirror evidence** (metrics, plots, checksums).
- Update **Strict Bounds** (old→new) when ratchets occur.
- Mark **Views exercised: 1..8**.

## 4) Ledger / Receipts
- Write **4-bit commit**, page **hash**, and **status (WORKING/NON-WORKING/PROVISIONAL)**.
- Include observers (J/Q/K) to set metrics, priority, and strictness.
