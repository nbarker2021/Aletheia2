# Bucket 7 — Windows, Cycles, and Braid Structure

## 1. Core Definitions

**Windows (W):** Anchoring steps that all routes must pass:
- **W4**: Σv ≡ 0 mod 4 (basic parity rest).
- **W80**: Σv ≡ 0 mod 8 and Q(v) ≡ 0 mod 4. Two flavors:
  - Global: one quadratic check.
  - Strict: each 8-block must pass.
- **Expansion windows**: e.g. 7/72 (Σv ≡ 0 mod 9 & Q(v) ≡ 0 mod 7) or φ-lane 11/40.

**Cycles:** Repeated passes through windows that define evolution:
- Range: 1–760 steps (default harness cycle range).
- Action vs rest trend:
  - Odd cycles: action-biased (movement, expansion, or repair).
  - Even cycles: organization-biased (stabilization, rest, or braid closure).

**Braids:** Multi-thread routes stitched together:
- Threads: each facet of the input mapped to its own 8-window sequence.
- Braiding: CRT merge across threads.
- Failures: clashing residues require expansion, re-braid, or failure log.

---

## 2. How Windows, Cycles, and Braids Interlock

1. **Windows** act as *gates of legality*. A route is only valid if it begins, ends, and reconciles at a rest.

2. **Cycles** define the tempo: odd/even alternation ensures entropy is released in action bursts, then reconciled in stabilization pauses.

3. **Braids** enforce multi-body consistency: separate threads must merge without residue conflicts, or else expand (via 7/72 or φ-lane) before re-merging.

Together, this triad ensures routes are **finite, auditable, and non-lossy**.

---

## 3. Algebraic Framing

- **Windows as constraints:**
  - W4: Σv mod 4 = 0.
  - W80: Σv mod 8 = 0 and Q(v) mod 4 = 0.
  - Wexp(p,ν): Σv mod ν = 0 and Q(v) mod p = 0.

- **Cycle parity:** Odd = enforce expansion/repair; Even = enforce closure/organization.

- **Braiding as CRT:** Let residues rᵢ across threads share a common modulus M. Braiding is possible iff ∃r such that r ≡ rᵢ (mod Mᵢ) for all i. Otherwise, route must expand.

---

## 4. Conceptual Interpretation

- Windows = **rest points of legality** (like quantum measurement projectors).
- Cycles = **temporal rhythm**: action vs reconciliation.
- Braids = **multi-body entanglement**: all threads converge on a consistent state.

This makes the system look like:
- A **lattice walk** with parity/isotropy checks (windows).
- A **temporal alternator** (cycles).
- A **braid group operator** (threads merging under CRT).

---

## 5. Open Questions

1. **Cycle ranges:** Why 760 as the default? Is this linked to lcm of windows (e.g. 4, 8, 9, 7, 11, 24)?

2. **Odd/even bias:** Is the action/rest alternation provable as a universal? Or just emergent in simulations?

3. **Braiding depth:** Are 5-braids always decomposable into 3-triad sets (as hypothesized), or are higher braids primitive in their own right?

4. **Window overlaps:** Do palindromic superpermutations always guarantee braid legality? Or only in some modulus families?

5. **Taxicab embeddings:** Do numbers like 1729 (taxicab #1) define natural braid closure checkpoints for multiple threads?

---

## 6. Global Test Plan

1. **Window integrity test:** Confirm that every closed route ends in W80 legality.
2. **Cycle parity test:** Run 1000+ simulated cycles and track odd/even entropy balance.
3. **Braid legality test:** Generate threads, attempt CRT merges, log all residue clashes.
4. **Palindrome bias test:** Compare pass rate of palindromic vs random threads.
5. **Taxicab embedding test:** Check whether sums-of-cubes identities act as braid closures in UVIBS embedding.

---

## 7. Provisional Claim

> **Claim:** UVIBS routes are braid-invariant cyclic walks in D=80 lattice space, where legality is defined by closure at rests (W4, W80, Wexp). Odd/even cycles alternate action and organization, while braids enforce multi-thread consistency via CRT. Palindromic sequences and taxicab embeddings bias routes toward legality by minimizing entropy spillover.

This is testable through algebraic simulation, entropy tracking, and braid merge experiments.

