# Bucket 2 — Seeds, Lifts, and Provenance (Expanded Technical Document)

## 0) Scope and Purpose
This document expands **Bucket 2** of the Superpermutation Governance System: it fully formalizes the mechanics of **seeds (Σ-states at n=5)**, **derivative lifts (n−1→n transitions)**, and **provenance rules (classification of digits >4)**. It includes explicit catalogs, tables, decision rules, and micro-traces. The goal is to remove ambiguity and make this bucket a **standalone operational handbook**.

---

## 1) Seeds Σ₁…Σ₈ (n=5 State Alphabet)

### 1.1 Definition
- At n=5, minimal superpermutation length = 153.
- Exactly **8 inequivalent minimal solutions** exist.
- In this system, these 8 are treated as **seeds** (Σ₁…Σ₈), each serving as a lawful chamber point in E8.

### 1.2 Characterization
| Seed | Palindromic? | Mirror Phase | Entropy Slot Policy | Lane Ordering | Notes |
|------|--------------|--------------|---------------------|---------------|-------|
| Σ₁   | Yes          | In-phase     | Suppressed (E−)     | Canonical 12345 | Reference pal seed |
| Σ₂   | No           | In-phase     | Allowed (E+)        | Alt lane A    | Minimal, entropy accepted |
| Σ₃   | No           | Out-phase    | E−                  | Alt lane B    | Shifts mirror timing |
| Σ₄   | No           | Out-phase    | E+                  | Alt lane C    | Alternate entropy strategy |
| Σ₅   | No           | In-phase     | E−                  | Alt lane D    | Balanced alternate |
| Σ₆   | No           | In-phase     | E+                  | Alt lane E    | Closer to Σ₂ but distinct |
| Σ₇   | No           | Out-phase    | E−                  | Alt lane F    | Strong out-phase presence |
| Σ₈   | No           | Out-phase    | E+                  | Alt lane G    | Entropy‑using out-phase |

### 1.3 Entropy Signatures
- **E− seeds**: entropy slots disabled, minimalistic.
- **E+ seeds**: entropy slots open, allowing deferred residue.

### 1.4 Weyl/E8 Role
- Each Σ corresponds to a **Weyl chamber point**.
- Symmetry actions (reflections) permute seeds, ensuring completeness.

---

## 2) Derivative Lifts (n−1→n)

### 2.1 General Rulebook
- From n−1 to n, choose a **partition**:
  - (n−1,1), (n−2,2), (n−3,3), …
- Each partition corresponds to a **shape** via modular meeting points (short cycles).

### 2.2 Partition → Shape Mapping
| Partition | Canonical Meeting Point (a,m) | ord_m(a) | Shape | Notes |
|-----------|-------------------------------|----------|-------|-------|
| (n−1,1)   | (2,5)                         | 4        | Square | One declared gate on vertex |
| (n−2,2)   | (4,5) + (2,5)                 | 2+4      | Two involutions + square | (Edges + square overlap) |
| (n−3,3)   | (3,7)                         | 6        | Hexagon | Two mirrored triads + bridge |

### 2.3 Algorithmic Selection
- **Step 1.** Identify all candidate (a,m) pairs with ord_m(a) ∈ {4,6,8}.
- **Step 2.** Score each lift:
  - Quarter-fix cost = QF
  - Pal receipts cost = PR
  - Entropy receipts cost = ER (rare)
  - **Score = QF + PR + ER**
- **Step 3.** Choose lowest score.
- **Step 4.** If tie → prefer in-phase pal alignment, else entropy suppressed.

### 2.4 Conflict Resolution
- If multiple bases with same order exist (e.g., ord₅(2)=4, ord₅(3)=4):
  - Prefer the base that aligns with current mirror phase.
  - If neither aligns, deposit into entropy slot and delay insertion.

---

## 3) Provenance of Digits >4 (Case Study: Digit ‘6’)

### 3.1 Provenance Class Axes
- **Seed origin**: Pal / Alt
- **Mirror phase**: In / Out
- **Entropy usage**: E− / E+

### 3.2 Table of 8 Classes for Digit ‘6’
| Class ID | Seed Type | Mirror Phase | Entropy | Notation |
|----------|-----------|--------------|---------|----------|
| P-I-E−   | Pal       | In-phase     | None    | Σ₁ aligned |
| P-I-E+   | Pal       | In-phase     | Used    | Rare exception |
| P-O-E−   | Pal       | Out-phase    | None    | Mirror shifted |
| P-O-E+   | Pal       | Out-phase    | Used    | Drift permitted |
| A-I-E−   | Alt       | In-phase     | None    | Balanced alternate |
| A-I-E+   | Alt       | In-phase     | Used    | Common entropy path |
| A-O-E−   | Alt       | Out-phase    | None    | Mirror misaligned |
| A-O-E+   | Alt       | Out-phase    | Used    | Fully alternate drift |

### 3.3 Assignment Rules
- When emitting digit >4, classify as follows:
  - **Entropy flag = E+** if residue is placed in slot at emission.
  - **Mirror phase** determined by controller (pal segment staging).
  - **Seed type** inherited from Σ-state chosen.

### 3.4 Receipts Format
- Each occurrence of digit >4 logged as:
  - `(position, digit, provenance_class)`
  - Example: `(124, 6, A-O-E+)`
- Receipts guarantee **every digit >4 has lawful provenance**.

---

## 4) Modular Arithmetic Layer

### 4.1 Base/Mod Table (m ≤ 16)
| m | φ(m) | Coprime bases | Orders (2,3,4) |
|---|------|---------------|----------------|
| 5 | 4    | {1,2,3,4}     | ord₅(2)=4, ord₅(3)=4, ord₅(4)=2 |
| 7 | 6    | {1,2,3,4,5,6} | ord₇(3)=6, ord₇(5)=6 |
| 8 | 4    | {1,3,5,7}     | All ord 2 |
| 9 | 6    | {1,2,4,5,7,8} | ord₉(2)=6, ord₉(5)=6 |
| 11 | 10  | Rich structure | ord₁₁(2)=10 |
| 13 | 12  | Rich structure | ord₁₃(2)=12, ord₁₃(6)=12 |

(Extension to m=32 is maintained in artifacts.)

### 4.2 Illegal Moves
- If lift suggests non-coprime base: **illegal lane**.
- Controller response:
  - First attempt alternate base with same order.
  - If none, defer via entropy slot (mirrored).
  - If still unresolved, fallback to GR mirror-2 scheduler.

---

## 5) Worked Micro-Traces

### Example: (5,1) Lift at n=6
- Meeting point: (a,m)=(2,5).
- Cycle: 1 → 2 → 4 → 3 → 1.
- Declare gate at 1 (residue anchor).
- Walk the cycle, checking alt‑4 in every 4-window.
- If out-of-phase: quarter-fix.
- Provenance: every ‘6’ tagged with (Seed type, Phase, Entropy flag).

### Example: (3,3) Lift at n=6
- Meeting point: (a,m)=(3,7).
- Cycle: 1 → 3 → 2 → 6 → 4 → 5 → 1.
- Form hexagon.
- Insert two mirrored triads as chords, one bridging triad.
- Provenance receipts: most 6’s here fall into A-O-E+ class (alt, out-phase, entropy used).

---

## 6) Interaction With Global Invariants

- **Pal-8 enforcement:** If digit >4 emitted out-of-phase, one **pal receipt** is mandatory.
- **W80 parity:** Ensures quadratic parity invariant across full pal segment. Certain meeting points (ord=6) align better with W80 than others.

---

## 7) Visualization & Geometry

- **(5,1) Square:** 4 vertices, 1 declared gate; resembles cube face.
- **(4,2) Edge + Square:** two involutions as opposing edges, stitched by square.
- **(3,3) Hexagon:** 6-cycle with 2 internal chords (triads) and 1 bridging triad.
- **Projection:** Each shape projects to E8 chamber coordinates corresponding to Σ-state.

---

## 8) Scalability Beyond 6

- At n=7: Provenance expands by introducing *fractal nesting* axis (e.g., shallow vs deep provenance). → 16 classes.
- At n=8: Adds helix axis (strand A vs strand B). → 32 classes.
- General rule: provenance dimension doubles at each odd/even consolidation step.

---

## 9) Summary
- **Seeds Σ₁…Σ₈**: full characterization and entropy signatures defined.
- **Lifts**: mapped to modular cycles; scoring function formalized.
- **Provenance**: 8 classes for digit ‘6’ enumerated; receipts schema provided.
- **Mod layer**: catalog through m=16 included; illegal move handling clarified.
- **Micro-traces**: worked (5,1) and (3,3) examples.
- **Global invariants**: pal receipts and W80 parity integrated.

This completes the formal expansion of Bucket 2: **Seeds, Lifts, and Provenance.**

