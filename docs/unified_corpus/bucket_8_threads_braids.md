## Bucket 8 — Threads & Multi‑Body Braids

### 1. Definition of Threads
- **Thread**: The atomic decomposition of a user request or state transition into one 8‑digit window.
  - Each window is lifted into 80D space and checked through gates (PAL/MIRROR) and rests (W4, W80, Wexp).
  - Threads act as independent “lanes” of processing, analogous to single‑particle worldlines.

### 2. Definition of Braids
- **Braid**: The composite formed when multiple threads are merged via CRT (Chinese Remainder Theorem) compatibility.
  - If residues agree: braid is direct.
  - If residues clash: expansion detour (7/72, φ‑lane) is attempted, then re‑merge.
  - If irreconcilable: failure is logged (BRAID_FAIL).

### 3. Multi‑Body Problem Framing
- Claim: any 3+ body relation **must be braided**, since pairwise overlaps cannot capture global consistency.
- Threads = local constraints for each body.
- Braids = global consistency conditions.
- This connects directly to the n‑body physics analogy (stable orbits emerge only when forces are reconciled across all bodies).

### 4. Algebraic Formulation
- Threads: {t₁, t₂, …, tₙ}, each with residue vector rᵢ.
- Braid operation: CRT({rᵢ}) → R if gcd‑compatible.
- If not: find expansion lane E such that lifted residues {rᵢ′} are gcd‑compatible.
- Governance kernel check: R ∈ K₍Monster∩E8₎.
- Log entropy: ΔS = ΣSᵢ(braid) – ΣSᵢ(thread).

### 5. Entropy Role in Braids
- Entropy accumulates across threads.
- Successful braid = entropy neutral or ΔS ≥ 0 (reconciliation adds capacity).
- Failed braid = ΔS < 0, logged in ledger with Crooks suppression.
- Palindromic bias increases likelihood of successful braids (due to symmetry compatibility).

### 6. Taxicab Number Analogy
- First taxicab number (1729) = smallest integer expressible as sum of two cubes in two distinct ways.
- Analogy: braids must reconcile multiple representations of the same total.
- Suggests taxicab numbers mark “free embedding points” for braids—natural resonance states where multiple threads can merge legally.

### 7. Open Questions
1. **Minimal braid unit**: Is it always 3 threads (triadic), or can dyadic braids exist that later demand triadic closure?
2. **5‑braid structure**: Are all 5‑braids built from 5×(3‑thread primitives), as suggested in earlier theory?
3. **Golden ratio modulation**: How does φ bias affect braid stability? Likely acts as a resonance filter for triadic vs pentadic clusters.
4. **Cache channels**: Do expansions (7/72, φ‑lane) always act as caches, holding entropy until a braid path opens?

### 8. Global Test Plan
- **CRT Consistency Tests**: Run synthetic thread sets with controlled residues; measure braid success rates.
- **Expansion Cache Tests**: Force incompatible residues, then measure if 7/72 detour allows reconciliation.
- **Entropy Tracking**: Compare ΔS across braids; check if monotonic reconciliation holds.
- **Taxicab Embedding**: Attempt to braid thread sets where totals equal known taxicab numbers; see if pass rates spike.
- **Golden Ratio Bias**: Add φ modulation to weights; test whether it increases braid stability.

### 9. Claim
Bucket 8 asserts: *All multi‑thread systems must reconcile into braids, and braids represent the minimal lawful unit of multi‑body state evolution. Success or failure of a braid is governed by modular compatibility, expansion caches, and entropy monotonicity.*

