_**This is a new slice created to unify the CQE system with the phone data.**_

# CQE-SPECTRAL Slice

## Purpose and Scope

The CQE-SPECTRAL slice makes the doc/band/tile graph spectral. It accepts edges that align with low-eigenvector flows or open spectral gaps.

## Axioms and Invariants

*   **Laplacian primacy**
*   **Rayleigh-drop gate**
*   **λ₂ gap optimizer**
*   **Multi-channel Laplacians (LATT/FRAC/CRT/SACNUM/GAUSS)**
*   **Monotone Φ_spec**

## DSL Hook

```
PROMOTE(i,j) iff NOETHER.Conservation.ok(i,j) & GROTH.Sheaf.glueOK(i,j, overlays≥2) AND SPECTRAL.RayleighDrop(i,j) AND SPECTRAL.GapOK(i,j) AND ( CRT.adjacent_tiles(i,j) OR FRAC.adjacent(i,j) OR LATT.adjacent(i,j) OR SACNUM.binAgree(i,j) ) AND Φ_spec(new) ≤ Φ_spec(prev) AND PARITY.ok(edge_ij)
```

## Minimal JSON

```json
{"spec":{"lap":{"type":"norm","λ2":0.27},"eigen":{"k":4,"coords":[0.12,-0.07,0.03,0.22]},"rayleigh":{"R":0.41,"ΔR":-0.05},"Φ_spec":0.066}}
```

