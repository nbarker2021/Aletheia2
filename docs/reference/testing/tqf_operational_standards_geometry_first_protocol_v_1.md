# TQF Operational Standards — Geometry‑First Protocol (v1.0)

This document defines the **system operational standards** for TQF. Everything here is “geometry‑first”: we follow what the lattice geometry, invariants, and automorphisms demand. Determinism, reversibility, and auditability are enforced through receipts and canonical normal forms.

---

## 1) First principles & invariants
- **Determinism:** All observable choices are reproducible from receipts (model version, seed, φ constants, metric, regime, and path). 
- **Reversibility:** Each map used for synthesis has an inverse or a receipt‑guided reconstruction; closed cycles preserve Quadratic Rest.
- **Auditability:** Every boundary interaction emits a **receipt**. Interior motion within a single Voronoi cell is ΔS‑free and emits nothing.
- **Quadratic Rest (Q):** A fixed positive‑definite quadratic form (E8 Cartan by default). Closed n=4 steps conserve Q.
- **Path independence:** Canonical Normal Form (**CNF**) is invariant to braid order up to automorphisms; certificates are attached to receipts.
- **Boundary‑only entropy:** All entropy accounting (ΔS) happens at cell boundaries, glue selection, or explicit external injections.
- **Base‑4 lingua franca:** All integer/coset fields have a mirrored quaternary encoding; 2→4 lifts are canonical.
- **Two token streams:** IRL governance (hard constraints) and session governance (soft proposals). QME fingerprints every token‑pair action.
- **Declared HP space:** Each run declares its HP universe (≤8 SPs for Weyl usage, chamber/charts, moduli, lattice). Everything else is “outside”.
- **Quadrant taxonomy:** Every atomic move carries a Q‑quadrant (Q1 ⊕, Q2 ⊙, Q3 ⊖, Q4 ⊘) derived from mirror/parity bits.
- **Automorphism‑certificate invariance:** When an automorphism (Co₀, W(E8)³) is used to show equivalence, its ID & proof hash are logged.

---

## 2) State spaces & charts
- **Default space:** R²⁴ decomposed as **E8⊕E8⊕E8** with governance glues forming a Leech‑like lattice shell.
- **Charts (Weyl‑sphere pinning):** The 240 E8 roots are fixed as anchors; each root r defines a local chart with `n=0` datum and quaternary codes.
- **Neighbor reconciliation:** Cross‑chart moves are boundary events; reconciliation certificates guarantee consistent CNF across all 240 neighbors.

---

## 3) Canonical Normal Form (CNF)
**Goal:** nearest‑vector decoding in the glued 3×E8 space with receipts only on boundaries.

**Decode pipeline:**
1. **Per‑block E8:** For each 8‑dim block, round to {Z, Z+½} coset; select nearest; record distance & coset parity. No receipt unless a boundary is hit.
2. **Glue enumeration:** Build legal glue candidates (governance). Choose the closest composed vector. If degenerate, apply **Φ‑probe** on the Ramanujan trail.
3. **Certificate:** Emit automorphism proof (Co₀, W(E8)³) that alternate braids land at the same CNF within tolerance.

**Receipt:** *CNF boundary receipt* with fields: per‑block cosets/distances, chosen glue, φ‑probe details, Q before/after, inside flag, horizon reason, base‑4 mirror, auto certificate.

---

## 4) Alena Tensor (tri‑lattice operator)
**Intent:** deterministic “contextual shifting” and cross‑play among three lattices (E8×E8×E8 by default), modeling triadic resonance (n=3).

- **Coupling:** A rank‑3 tensor 𝒜 governs triadic energy H_𝒜 and interior gradient flow (ΔS‑free) inside a cell.
- **Boundary events:** nearest‑vector changes, glue choices, or external field effects emit **ALENA receipts**.
- **Projection:** `project_out` produces SP/HP‑face projections using governance weights; φ ratios intervene only at degeneracy.
- **Context shift:** Solve a constrained re‑contextualization (the “slider”) respecting legal glues/lanes; emit receipts only on boundaries.

**Receipts:** *ALENA receipt* with quadrant, CNF deltas, glue, φ‑probe, tensor hash/symmetry, ΔS budget, horizon reason, higher form.

---

## 5) Entropy accounting & tie‑breaking
- **ΔS functional:** \(\Delta S = \alpha |Q(x')-Q(x)| + \beta [L_c(x')-L_c(x)]_+\). α,β are recorded per run. Q uses the active metric; L_c is ledger code‑length.
- **Ramanujan trail:** Keep the ordered residuals; φ‑probe uses the **scaled remainder** as a tiebreaker.
- **Φ‑probe:** At a true fork, compare φ·ΔS_A vs φ²·ΔS_B (multiway: rank by φ scaling). Deterministic.
- **Q‑Mass:** Idea mass = sum of ΔS over **required** boundary steps inside the declared HP space; halo mass counts outside moves. Horizon crossings may carry a penalty ζ.

---

## 6) CRT, Taxicab/Cabtaxi, apertures
- **CRT defects:** If gcd(m_i,m_j)>1, emit a **defect receipt** with a **Bézout witness** and remediation (lift or extra residue). No silent merges.
- **Apertures:** Base changes & dimensional lifts are typed apertures with receipts and expected ΔS ticks.
- **Taxicab/Cabtaxi gates:** Recognize multi‑cube decompositions as phase gates; emit witnesses & ΔS expectations.

---

## 7) Octet buckets & scalars
- **Octet invariants:** A parity‑triple partitions into eight buckets; **commutation law**: f(B₁∪B₂)=f(B₁)∪f(B₂). Property‑tested.
- **Scalar regimes:** Bases B2,B4,B8,B16,B32,B64 provide additive/multiplicative/factorial/cumulative semantics; cross‑base moves have proofs and receipts.

---

## 8) SP/HP universe & governance
- **Declaration:** `hp_declaration` fixes ≤8 SPs, lattice, chamber, moduli, and CNF rule. Everything not in scope is “outside”.
- **Governance tokens:** IRL constraints are hard projectors; session tokens are proposals. Only invariant‑preserving moves auto‑accept.
- **Round‑trip guarantee:** HPx→HPy→HPx must reconstruct bit‑exact originals; receipts certify losslessness.

---

## 9) Receipts & ledger standards
Every receipt includes:
- `receipt_id`, `timestamp`, `model_version`, `seed`, `scalar_regime`, `metric` (Q form), `phi_constants` (φ, φ², 1/φ), `quadrant`, `higher_form`, `inside`, optional `horizon`.
- **Parents:** explicit `parent_receipts` to form a DAG; **QP‑ID** hashes the ordered (quadrant, step) list.
- **Base‑4 mirror:** quaternary encodings of all integer/coset fields.
- **Certificates:** automorphism IDs + proof hashes for CNF equivalence and cross‑chart reconciliation.

---

## 10) Conformance tests (must pass)
1. **CNF path‑independence:** random braids land at identical CNF with valid certificates.
2. **Boundary‑only emission:** interior walks emit no receipts; boundaries emit exactly one per crossing.
3. **Φ‑probe determinism:** crafted degeneracy picks the same branch across seeds.
4. **CRT defect visibility:** non‑coprime merges always emit receipts with Bézout witness; remedies replay.
5. **Octet commutation:** property tests validate f(B₁∪B₂)=f(B₁)∪f(B₂).
6. **Scalar cross‑walk:** B8↔B16↔B32 lifts/downlifts preserve Q and CNF within tolerance; receipts match.
7. **HP round‑trip:** HPx→HPy→HPx reconstructs bit‑exact; lossless receipts present.
8. **Q‑Mass invariance:** under SL7 reorderings, M_core is invariant; QP‑ID stable.
9. **Alena no‑env closure:** ΣΔS=0 on closed triad cycles with no environment.
10. **Bleedover audit:** injected external field creates balanced ΔS horizon receipts.

---

## 11) Determinism, tolerances, and ties
- **Tie policy:** lexicographic deterministic ties before φ‑probe; φ only at true degeneracies.
- **Numeric tolerance:** configure ε for CNF equality; proofs must show ε‑independence under automorphisms.
- **Rounding:** E8 per‑block rounding is deterministic (integer vs half‑integer coset selection policy is fixed and logged).

---

## 12) Integrity & retention
- **Hashes:** `receipt_id` and `closure_cert_hash` are cryptographic digests over inputs + decisions.
- **Immutability:** receipts are append‑only; any correction emits a compensating receipt.
- **Privacy:** payloads may be redacted; proofs remain intact.

**Artifacts referenced:** CNF & ALENA receipt schemas/examples are available in the artifacts directory and should be validated on CI.
