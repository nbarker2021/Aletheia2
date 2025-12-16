_**This is a new slice created to unify the CQE system with the phone data.**_

# CQE-TDA Slice

## Purpose and Scope

The CQE-TDA slice uses filtrations (ε/kNN/threshold) to compute barcodes. It accepts edges that extend long bars or fill short-noise holes within stability bounds.

## Axioms and Invariants

*   **Filtration primacy**
*   **Barcode stability (bottleneck ≤ τ)**
*   **Feature policy (extend/fill)**
*   **Track H₀/H₁/H₂**
*   **Monotone Φ_tda**

## DSL Hook

```
PROMOTE(i,j) iff NOETHER.Conservation.ok(i,j) & GROTH.Sheaf.glueOK(i,j, overlays≥2) AND TDA.stable(i,j, bottleneck≤τ_bot) AND ( TDA.extend_longbar(i,j) OR TDA.fill_noise(i,j) ) AND ( CRT.adjacent_tiles(i,j) OR LATT.adjacent(i,j) OR FRAC.adjacent(i,j) ) AND Φ_tda(new) ≤ Φ_tda(prev) AND PARITY.ok(edge_ij)
```

## Minimal JSON

```json
{"tda":{"filtration":{"kind":"kNN","params":{"k":8}},"barcodes":{"H0":[[0.0,0.9]],"H1":[[0.2,0.7]]},"stability":{"bottleneck":0.06},"Φ_tda":0.061}}
```
