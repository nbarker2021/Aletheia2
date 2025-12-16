
# Cartan Quadratic Equivalence (CQE) — Operational Theorem (Draft)

**Claim (operational form).**  
Let \(X\subset \mathbb R^d\) be a dataset equipped with a symmetry group \(G\le O(d)\) and a *viewer* \(V\) that maps \(X\) to a canonical chart \(V(X)\) (CSA-like snap: idempotent, equivariant, energy-nonincreasing).  
There exist **four** orthogonal “quadratic” reparameterizations \(Q_1..Q_4\in O(d)\) and **two** parity counter-forms \(P_0=\mathrm{Id}, P_1\in O(d)\) such that, for **two** independent “meaning” summaries \(M_1, M_2\) (class functions on \(G\)-orbits),
\[
M_j\big(P_a Q_i\, V(X)\big) = M_j\big(V(X)\big)\quad \text{for all } i\in\{1..4\}, a\in\{0,1\}, j\in\{1,2\},
\]
up to small numerical tolerance. Hence an **8-lane** equivalence bundle (4×2) preserves meaning across quadratic views and parity flips.

**Instantiations used here.**
- \(M_1\): Top-3 masses of the inner-product histogram (pairwise metric shape).
- \(M_2\): Top-10 eigenvalues of the covariance (variance/energy spectrum).
- Viewers \(V\): (i) CSA-like canonicalization for lattices; (ii) PCA whitening for real-form data; (iii) PCA+chamber snap for transformer heads (previous harness).
- Quadratic views \(Q_i\): block permutations, Householder reflections, block-orthogonal rotations, and (for lattices) a root reflection in the first block.
- Parity \(P_1\): fixed coordinate sign-flip.

**Empirical verdicts.**
- Lattices: \(E_8^4\) (32D), \(E_8^8\) (64D) — PASS (previous), and **Construction A (24D)** — PASS.
- Real forms: **compact (block rotations, 16D)** — PASS; **split (block boosts, 16D) with whitening viewer** — PASS.
- Transformer head surrogate (64D) — PASS (previous).

**Why this matters.**  
CQE formalizes an 8-lane invariance structure that (a) justifies “do it once, reuse forever” ledgers across CSA choices and parity; (b) explains why octets recur as organizing media; and (c) gives a concrete, auditable criterion for viewer quality (snap must be energy-nonincreasing and equivariant; meaning must be invariant across 8 lanes).

**Roadmap to theorem.**
1. **Hypothesis class.** Finite-dimensional datasets whose distributions are mixtures of orbits under compact subgroups of \(O(d)\) or lattices from Construction A/C/D; viewers that are \(G\)-equivariant canonicalizers.
2. **Quadratic bundle.** Exhibit four generators \(Q_i\) in a Weyl-like/orthogonal subgroup preserving \(M_1,M_2\) (class functions).
3. **Parity.** Provide an orientation-flip \(P_1\) with the same invariants.
4. **Stability.** Show small noise/finite sampling perturbations keep errors below fixed tolerances.
5. **Extensions.** Real forms via complexification/whitening viewers; code-lattices via Construction A; Leech/E8 families via known Weyl groups.

**Reproducibility.**  
This notebook emits tables for each domain with per-view errors and a summary table with PASS/FAIL under thresholds (hist ≤ 1.5e−2; spec ≤ 1e−2).
