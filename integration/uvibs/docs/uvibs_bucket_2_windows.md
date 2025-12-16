Uvibs — Bucket 2: Windows & Rests (draft 1)

---

## Scope
This bucket formalizes the **resting checkpoints** and **window operators** that anchor movement through UVIBS space. They are the algebraic gates where entropy is reconciled, legality is checked, and local moves are embedded into global flows.

---

## 1. Core Rests

### W4 — Parity Rest
- **Definition:** \( \sum v \equiv 0 \ (\text{mod } 4) \).
- **Role:** First primitive rest; ensures closure of triadic/binary moves.
- **Operator interpretation:** 1 (primitive) + 3 (triad) → rest at 4.
- **Physics parallel:** Local conservation law (like charge parity).
- **Tests:**
  - Any sequence of moves must eventually hit W4.
  - Verify ΔS ≥ 0 on closure at W4.

### W80 — Codec Rest
- **Definition:**
  1. \( \sum v \equiv 0 \ (\text{mod } 8) \) (octadic neutrality).
  2. Quadratic form \( Q(v) = v^T G v \equiv 0 \ (\text{mod } 4) \).
- **Modes:**
  - Global: total parity checked once.
  - Strict: each E8 block checked individually.
- **Role:** Main reconciliation point after expansion or direct codec.
- **Tests:**
  - Direct 4 → 80 closure (global).
  - Strict failures → expansion needed.

### Expansion Windows Wexp(p,ν)
- **General form:** \( Q(v) \equiv 0 \ (\text{mod } p), \ \sum v \equiv 0 \ (\text{mod } \nu) \).
- **Examples:**
  - (7,9|72): isotropy mod 7 + triadic neutrality mod 9.
  - (11,10|40): φ-lane, isotropy mod 11 + neutrality mod 10.
- **Role:** Reopen routes when strict W80 jams.
- **Tests:**
  - Show ΔS > 0 after re-entry.
  - Verify closure back into W80 within finite steps.

---

## 2. Windows as Legal Handshakes

- **Concept:** Every window is both a gate (legal check) and a ledger step (entropy snapshot).
- **Forcing rules:**
  - W4 must precede all higher rests.
  - W80 is mandatory for closure.
  - Expansion windows are optional, but must reconcile.

- **Pal/Mirror Interaction:**
  - PAL ensures natural symmetry for closure.
  - MIRROR ensures reversibility of entry/exit.
  - Together they bias acceptance probabilities at windows.

---

## 3. Algebraic Composition

- **Rest equations:**
  - W4: \( x_1 + x_2 + x_3 + x_4 \equiv 0 \ (\text{mod } 4) \).
  - W80: \( \sum_{i=1}^{80} x_i \equiv 0 \ (\text{mod } 8), \ Q(x) \equiv 0 \ (\text{mod } 4) \).
  - Wexp(p,ν): defined as above.

- **Monotonicity:**
  - At each rest, enforce ΔS ≥ 0.
  - Crooks ratio = \( P_{rev}/P_{fwd} = e^{-ΔS} \).

---

## 4. Example (N=3 superpermutation → rests)

Sequence 123121321:
- At ‘1’: W4 rest (parity close).
- At ‘3’: Wexp(7,9) expansion open.
- At ‘2’: codec dual lane toward W80.
- End ‘321’: close at W80.

Ledger shows ΔS path monotone, with expansion only once.

---

## 5. Open Questions

- **Cache vs. Catching:** Are expansion lanes (7/72, φ-lane) pure detours, or caching channels for entropy reuse in later windows?
- **Golden ratio modulation:** Where do φ-based expansions naturally stabilize W80 closure?
- **Composite windows:** Can Wexp compose (e.g., W7/72 + W11/40) into higher embeddings without loss of legality?
- **Taxicab parallels:** Does the taxicab sum structure (two cubes = one number) act as a natural expansion rest?

---

## 6. Tests

1. Verify ΔS monotonic closure for random paths through 4 → 80.
2. Introduce strict jam at W80, route through 7/72, check reconciliation.
3. Force dual PAL+MIRROR prefilter, track acceptance probability boost.
4. Simulate φ-lane entry, measure repair budget vs. 7/72.
5. Attempt composite expansion, check entropy ledger.

---

**Claim to test:** All legal UVIBS trajectories must begin with W4, optionally pass through expansion windows, and must close at W80, with ΔS ≥ 0 on closure and Crooks-consistent reversibility. Expansion windows act not only as detours but as entropy caches, biasing future closure success.

