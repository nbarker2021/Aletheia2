# Superpermutation Solver — SFBB Alignment

## Strategy Mix
- **Bouncing Batch**: partition n! permutations into 2^n cells; evaluate hypotheticals per cell,
  then bounce laminate knowledge to neighbors.
- **Hypothetical generation**: n−1 seeding, prodigal combination → Mega‑Hypotheticals (GR‑proportioned),
  De Bruijn‑guided, constrained random, mutation, formulas.
- **Completion**: fill missing permutations efficiently using winners/losers and laminate compatibility.
- **Reconfiguration**: shorten complete sequences post‑hoc without dropping coverage.

## Knowledge Artifacts
- **Prodigal Result**: efficient subsequence; tracks breakpoints, overlap rate, extensibility.
- **Mega‑Winners/Losers**: long runs of favored/disfavored transitions.
- **Laminates & Anti‑Laminates**: allowed/disallowed k‑mer transitions; merged across cells.
- **Constraint laminate**: from best known solution to cap length.
- **Winners/Losers**: weighted k‑mer preferences from analysis.

## Graph Backbone
- **De Bruijn graphs** weighted by winners/losers; search for high‑weight paths.
- Analyze density & connectivity to switch strategies dynamically.

## Multiprocessing & DTT
- Workers test complete hypotheticals; return validity, length, deltas to laminate/winners/losers.
- Aggregator merges, updates constraints; DTT gate controls adoption of shorter strings.

## Failure & Recovery
- **Dead zones**: laminate density low → switch to mutation or prodigal recombination.
- **Over‑constraint**: relax anti‑laminates, increase formula variance.
