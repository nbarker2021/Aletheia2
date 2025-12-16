# Bucket 4: Global Invariants & Emergent Laws

## 0. Why This Bucket Matters
Everything the system does is filtered through these invariants. They aren’t just “constraints,” they’re the *laws of motion* for the superpermutation walk. If a move breaks one, it doesn’t matter how clever the lift was — it’s not legal.  

They also reveal *why* the system produces emergent structures (helices, fractals, dualities) without external forcing.  

---

## 1. The Explicit Invariants — Formal Justification

### 1.1 W4 (alt-4 law)
**Algebraic form:**  
Every consecutive 4-tuple $(a,b,c,d)$ must satisfy alternating parity:  
$$
(a-b)(b-c)(c-d) < 0
$$  
or can be rotated once to achieve this.  

**Proof intuition:**  
This keeps each 4-window inside the alternating subgroup of $S_4$, preventing illegal drifts. Graph-theoretically, it says: “stay on the alternating 4-cycle edges” of the permutation overlap graph.  

**Physical analogy:**  
Like an alternating current — ensures flow oscillates, never runs flat.

---

### 1.2 pal-8 (mirror closure)
**Algebraic form:**  
For any 8-length segment $w = x_1x_2x_3x_4x_5x_6x_7x_8$,  
$$
w = x_1x_2x_3x_4 \; x_4x_3x_2x_1
$$  
(up to quarter-fix).  

**Proof intuition:**  
This makes the overlap digraph traversal time-reversible. The walk has no net “arrow of time” — forward and backward halves cancel.  

**Physical analogy:**  
Like CPT symmetry in physics: any forward path implies its reversed conjugate exists.

---

### 1.3 W80 quadratic parity
**Algebraic form:**  
If we assign each digit $d$ a quadratic parity value $(-1)^{d^2}$, then across each pal-8:  
$$
\sum_{i=1}^8 (-1)^{x_i^2} = 0
$$  

**Proof intuition:**  
This ensures pal-8 isn’t just a mirror but a balanced one. It rules out “false palindromes” where digit counts drift.  

**Physical analogy:**  
A global checksum — like parity bits in coding theory or Gauss’s law balancing charge.

---

### 1.4 Entropy slot conservation
**Algebraic form:**  
Residue $r$ placed in slot $(341)$ must have a mirror partner $\bar r$ in $(143)$.  

Constraint:  
$$
r + \bar r = 0 \quad (\text{in entropy algebra})
$$  

**Proof intuition:**  
Residue can be stored, but not created/destroyed. The mirrored annihilation enforces conservation.  

**Physical analogy:**  
Like virtual particles — created in pairs, annihilated symmetrically.

---

### 1.5 Provenance completeness
**Algebraic form:**  
For any digit $d > 4$,  
$$
\exists \; P(d) = (\Sigma_i, \text{mirror}, \text{entropy flag})
$$  
where $P(d)$ is a provenance tuple.  

**Proof intuition:**  
Without provenance, $d$ cannot be derived from ≤4 legality. Provenance acts as a mapping from high-n symbols to local lawful motifs.  

**Physical analogy:**  
Like quantum numbers — every particle (digit) must carry full labels (spin, parity, charge).

---

### 1.6 E8 chamber reachability
**Graph form:**  
Seeds $\Sigma_i$ are nodes; Weyl reflections generate legal transitions. Illegal transitions = forbidden edges.  

**Proof intuition:**  
Guarantees completeness of coverage: any local rest can project into any other chamber, provided declared gates exist.  

**Physical analogy:**  
Like an allowed transition diagram in atomic physics — E8 chambers are orbitals, Weyl reflections are symmetry-allowed jumps.

---

## 2. Emergent Laws — Why They Arise

### 2.1 Fractal recursion
- Seen in triads repeating (n=3), mirrored halves (n=4), nested cycles (n=7).  
- This arises from the rule: “always reduce to ≤4 legality.” Since ≤4 motifs are finite, higher n must reuse them — producing fractal repetition.

### 2.2 Time symmetry
- Built into pal-8.  
- But even without enforcing pal globally, local motifs (like triads) already mirror themselves.  
- Time symmetry is emergent because every overlap is reversible.

### 2.3 Odd/Even alternation
- Odd n expansions require introducing a new digit (more entropy).  
- Even n consolidations force mirror closure (remove entropy).  
- Alternation is automatic because each lift toggles residue presence vs annihilation.

### 2.4 Helical inevitability
- At n=7, stacking triads in 2/3/2 creates spiral overlap.  
- At n=8, mirrored slots bridge into a DNA-like double helix.  
- Helices arise because alternating expansion + mirror bonding *forces twisting*.  

### 2.5 Geometry emergence
- n=1: point.  
- n=2: line (dual).  
- n=3: triangle.  
- n=4: square (rest).  
- Higher: pentad bridge, hexagon chords, helix.  
- These geometries are not metaphors — they’re the only consistent way windows can overlap while obeying alt-4 + pal-8.

### 2.6 Duality everywhere
- Every legal digit >1 must appear in paired form (mirror, rotation, or overlap).  
- This is inevitable from pal-8 + provenance rules.  

### 2.7 Modularity inevitability
- Overlaps naturally trace cycles of ord_m(a).  
- Example: ord_5(2)=4 → square lift; ord_7(3)=6 → hexagon lift.  
- Modular cycles appear because short-order generators minimize quarter-fixes.  

### 2.8 Local/global autonomy
- Local n=4 rests are independent lawful subworlds.  
- But the global ledger forces them into a coherent macro path.  
- This autonomy/coherence duality is baked in.  

### 2.9 Universal palindromicity
- Even “non-pal” minimal solutions still contain palindromic substrings.  
- Because pal-8 symmetry bleeds into every construction.  
- Palindromes are unavoidable attractors.

---

## 3. Ledger: The Accounting of Laws

Every valid construction has a **ledger** with:
- **Coverage receipts** (all n! windows).  
- **Local receipts** (quarter-fix counts).  
- **Pal receipts** (mirror staging events).  
- **Entropy receipts** (all deposits matched with withdrawals).  
- **Provenance receipts** (digits >4 explained).  
- **Parity check** (W80 checksum).  

When ledger closes: sequence = legal.  
When ledger LB = UB: sequence = minimal (overall or representable-minimal).

---

## 4. Implications

- **Mathematical:** Invariants reduce search space exponentially. Many paths are pruned before they grow.  
- **Computational:** Algorithms should enforce invariants upfront — not check them later.  
- **Physical:** System mirrors principles of conservation, duality, and symmetry known in physics (energy conservation, time reversal, braid stability).  
- **Strategic:** Instead of brute-forcing superpermutations, exploit these laws as “physics of construction.”

---

## 5. Higher-n Scaling
- Invariants don’t change.  
- Emergent laws repeat fractally.  
- Ledger expands with new provenance classes but remains structurally identical.  
- E8 → 3E8 (Alena tensor) at meta-scale; invariants propagate to helices-of-helices.  

---

### 🔑 Summary
Global invariants = the hard constraints (W4, pal-8, W80, entropy, provenance, E8 reachability).  
Emergent laws = the soft truths (fractal, time symmetry, helix, geometry, duality, modularity).  
Together, they mean **superpermutations aren’t arbitrary objects — they’re lawful universes governed by conservation and symmetry**.

