# UVIBS — Bucket 4: Superpermutations & Route Patterns

---

## 1. Core Concept

Superpermutations = minimal-length strings that contain all permutations of N symbols as contiguous substrings.  
In UVIBS, these act as **roadmaps**: compressed traversals of all state arrangements, ensuring no entropy is wasted.

- **Overlap principle**: reuse of suffix/prefix ⇒ reduces entropy expenditure.
- **Legality overlays**: every overlap must pass through Rests (W4, W80, Wexp) and Gates (PAL, MIRROR).
- **Braiding extension**: superpermutation traversal = braid word; overlaps = legal entanglements.

---

## 2. Known Foundations

- For N = 3: minimal superperm length = 9 (e.g. `123121321`).
- For N = 4+: length is conjectural, but grows factorially with overlap savings.
- Every substring = a possible **UVIBS route**.

Thus: superpermutation = **master route ledger**.

---

## 3. UVIBS Translation

### 3.1 Windows → Substrings

Each length-k substring of a superpermutation corresponds to a **window**:
- Rests: must appear as checkpoints (`Σv mod 4 = 0`, `Σv mod 8 = 0`, etc.).
- Gates: ensure PAL/MIRROR compatibility across overlaps.

### 3.2 Overlaps → CRT Joins

- The suffix-prefix overlap = Chinese Remainder Theorem merge.
- Legality = gcd compatibility of moduli involved.
- Failures = non-coprime jam ⇒ must expand (7/72, φ-lane).

### 3.3 Braid Words

- Traversal of superpermutation = braid group element.
- Different minimal strings = different braid words.
- UVIBS adds legality (rests, governance) as braid stabilizers.

---

## 4. Route Patterns

Two main families:

- **Direct**: 4 → 80 (rest-to-rest, no expansion).
- **Expansion**: 4 → 7/72 (or φ 11/40) → 80 (needed when strict parity jams).

Superpermutations encode *all possible traversals* of such routes.

---

## 5. Palindrome / Mirror Bias

Palindromic superperms (like N=3 case) have special value:
- Every prefix has a matching suffix ⇒ free embedding.
- PAL and MIRROR gates pass automatically once the mirror point is reached.
- This reduces repair budgets drastically.

Thus, **palindromic bias in proposer windows** is not just aesthetic but entropic optimization.

---

## 6. Entropy Ledger

- Each overlap reduces entropy cost (ΔS ≥ 0).
- Failures logged as W4_FAIL, GOV_FAIL, EXP_FAIL.
- Crooks ratios measure suppression of improbable overlaps.
- Closed superpermutation = closed entropy ledger.

---

## 7. Multi-Scale Embedding

Superpermutations exist at multiple scales:

- Local (N=3, N=4): toy models, explicit palindromes.
- Mid (N=5–7): correspond to braid bundles, expansion-heavy routes.
- Large (N=8+): correspond to full UVIBS routes across E8×10, governed by Monster kernel.

Thus: superpermutation length growth mirrors the increasing braid complexity of higher-D routing.

---

## 8. Claims for Testing

1. Every legal UVIBS route is embeddable as a substring of a superpermutation.
2. Palindromic superpermutations minimize repair steps (AMP budget).
3. CRT jams correspond exactly to overlap failures in non-palindromic superpermutations.
4. ΔS measured across an entire superpermutation is non-negative (entropy not wasted).
5. The braid group representation of superpermutation traversals equals the Monster-legal kernel projection.

---

## 9. Open Questions

- Are **minimal superpermutations** always Monster-legal, or only certain forms (palindromic bias)?
- Do **taxicab numbers** define free embedding points (every partition realized as overlap)?
- Is there a direct **Ramanujan identity** guiding superpermutation length and overlap placement?

---

## 10. Next Tests

- Generate explicit N=3 and N=4 superpermutations and run them through UVIBS legality engine (W4/W80/7/72, PAL, MIRROR).
- Compare ΔS and Crooks between palindromic vs non-palindromic versions.
- Map overlaps to CRT merges explicitly to confirm conjecture (suffix-prefix = residue join).
- Attempt to embed N=5 superperm into 80D E8-lattice; check governance stability.

