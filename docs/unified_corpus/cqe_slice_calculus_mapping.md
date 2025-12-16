# Slice Calculus ↔ Current CQE/ARC360 Mapping

## Already covered (implemented or partially implemented)
- **Angle space, lattice W** → ARC360 uses 360° with H/M/S ticks; cascade grids (×2, ×10) act as multi-resolution lattices.
- **Rotation (pose)** → ARC360 tick snapping and pass cascades (states pass1–pass3).
- **Enforce without demand** → All our moves are suggestions/receipts; no mutation.
- **Residue corridors / prime-coprime** → PRIME_COPRIME module: closure scores per prime, CRT weaves, recycle routes.
- **Bridging/tunneling** → ARC360 bridging: seam ring-bridges, cascade-bridge suggestions, residue bridges, prime tunnels.
- **Torus / periodicity** → TORUS_magic_sizes and spectral periodicity proxies (FRAC/TORUS flags).

## Gaps to close (what we’ll add)
1. **Observable generators on S^1_W**  
   - Build `extreme_vertex`, `chord_histogram`, `quadrant_occupancy` samplers for a chosen viewer set (hexagon/decagon/octagon).  
   - Source braid events from `permutation` transitions or from `cqe_braiding_metrics.csv` when present.

2. **Slice calculus primitives**  
   - Compute `Dθ`, `Δθ`, `∫`, `E(f)` per observable; emit equalizing pose by minimizing target energies (e.g., quadrant variance).

3. **Equivalence checks**  
   - Rotation/reflection-invariant L2 on a stack of observables; emit an **equivalence key** (pose key per observable, concatenated + canonicalized).

4. **Multi-resolution monotonicity**  
   - Verify equivalence holds from W=80→120→600; flag scale-dependent faces.

5. **Braid current / topological receipts**  
   - Build `J_B[m]` and braid windows; produce writhe and crossing counts as slice integrals.


## Data we have vs need
- **Have:** ARC360 angles (pass states), residue maps (32/64/128), prime recycle routes, braiding metrics (global).  
- **Need (per-face sampling):** viewer face point sets or a canonical generator, to evaluate observables on S^1_W for each token. If absent, we can synthesize a lawful N-gon viewer family (no semantics) and run purely geometric calculus.
