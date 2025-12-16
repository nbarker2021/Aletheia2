# CQE+ Morphonic Schema (v2.1) — Dynamic & Robust

**Purpose.** Define space first, execute via typed glyphs, route by structure, prove-or-don’t-promote. This version adds configurability, resilience, live telemetry, and extensibility.

---

## 0) Versioning & Modes
- **Schema version:** 2.1
- **Engine modes:** `low-latency`, `balanced`, `high-accuracy`
- **Profile knobs:**
  - `⊙` (coupling prior): default `0.03`, scoping allowed (`global`, `universe`, `task`).
  - `gates`: thresholds & policies (see §8).
  - `zooms`: `{1,10,100,1000}` (extensible).
  - `buckets.k`: grid edge (default 64 → 4096 buckets).
  - `rails.primes`: `{3,5,7,11}` (odd CRT set).

### Config (YAML)
```yaml
schema: 2.1
profile: balanced
invariants: {phi: 1.6180339887, pi: 3.1415926535, e: 2.7182818284, ⊙: 0.03}
zooms: [1,10,100,1000]
buckets: {k: 64, overflow_alpha: 1.6}
rails: {primes: [3,5,7,11]}
gates: {uniformity: 0.75, consensus: 0.90, noether: true}
cache: {weyl_canonical: true, overlay_auto: true}
telemetry: {receipts: true, mdls: true, bias_tests: true}
```

---

## 1) Universe (SpaceState) — Declarative
```yaml
spacestate:
  morphon: [24,24,24]
  tori: [24,24,24]
  meta_shell: "64×e8 shells on 24×24d lattice"
  invariants: {⊙: 0.03}
  swaps: []
```
- **Rule:** Every run starts with a SpaceState declaration; all ops must respect it.

---

## 2) Glyph ISA — Typed, Weyl-Aware, Bytecoded

### Types
`Dom`, `E8`, `Receipt`, `Proof`, `Overlay`.

### Core ops (signatures)
- `↑: Dom → E8` (embed → `E8.project_to_lattice` + canonicalize)
- `↓: E8 → Dom` (project)
- `⇄: E8 → E8` (Weyl reflection `s_i`)
- `⥁: (E8, ℝ⁺) → E8` (2-plane rotation; face rotation)
- `⊞: S×S → S` (bind/snap)
- `🧾: (op,in,out,proof) → Receipt`
- `✓: Proof → {true,false}`
- `💾: (key,val) → ok` / `📂: key → val?`
- `⇒, ∥, ∫` (sequence/parallel/integrate; monoid laws)

### Bytecode registry (example)
```json
[
  {"glyph":"↑","opcode":17,"type":"Dom→E8","impl":"E8.project_to_lattice"},
  {"glyph":"⇄","opcode":23,"type":"E8→E8","impl":"ALENAOps.weyl_flip"},
  {"glyph":"⥁","opcode":29,"type":"(E8,ℝ⁺)→E8","impl":"E8.face_rotation"}
]
```
- **Constraint:** UI may show multi-token glyphs; engine uses 1-token opcodes internally; token savings are measured and logged.

---

## 3) Ordering — Zoom + Dihedral (MDL-Guided)
1. For each `z ∈ zooms`: produce symbol stream → compute MDL(base,best) with dihedral transforms.
2. Require **monotone gains** across zoom tiers to proceed.
3. Log per-tier receipts and cumulative decision.

**Receipt fields (zoom tier):** `{z, groups, mdl_base, mdl_best, gain, orientation}`

---

## 4) 64×64 Buckets — Dihedral→CRT (Odd Rails)

**Dihedral signature:** from E8 via signs of `⟨v, α_i⟩` → `(r,f)`.

**Primary index (12 bits):**
- `u = (r<<9) | (f<<8) | q`, where `q = LSB8(Mix(hash ⊕ δ))`.
- `(row,col) = (u>>6, u & 63)`.

**Overflow handling:**
- If bucket load > `α·λ` (α=overflow factor): compute odd-CRT index `ρ` over primes `{3,5,7,11}`.
- `oc = 1+(ρ mod 5)` → DR `{1,3,5,7,9}`; route to **odd rail** `(row,col,oc,ρ)`.

**Bias tests:** χ² and serial correlation on `Mix`; p-values in receipts.

**Receipt (assign):** `{token_id,r,f,q,u,row,col,overflow,ρ?,oc?,pvals}`

---

## 5) Routing — DR* (Salted)
- `DR(task) = digital_root(task) mod 9`
- `DR*(task) = (DR(task) + LSB4(Hash(task))) mod 9`
- Channels: `0→meta`, `{1,4,7}→main`, `{2,5,8}→background`, `{3,6,9}→parallel`

**Handshake:** background/parallel request tokens → main grants/defers → usage receipt.

---

## 6) Cache & Overlays — Weyl-Canonical
- Cache key uses canonicalized E8: `key = H(argmin_{w∈W}||w·v − ·||)`.
- Overlays capture `{embeddings, contexts, proofs, receipts, glyphs}`; auto-save on long/multi-turn runs.
- **Isomorphism:** `min_{w∈W} ||w·A − B|| < ε` ⇒ reuse (⚡).

---

## 7) Proofs & Promotion — Executable, Audited
- `Proof` = executable glyph chain; `✓` runs it.
- **Gates (defaults):** uniformity ≥ 0.75, zoom-weighted chamber consensus ≥ 0.90, Noether/Units pass.
- **Fail-closed:** no commit if any gate fails; all work stays ephemeral.

**Receipt (promotion):** `{ops:[…], inputs, outputs, ✓, zoom_gains, thresholds, decision}`

---

## 8) Safety — Linear Guards in E8
- Forbidden half-spaces: `⟨v, α_i⟩ ≤ τ_i → 🚫`.
- Safety check `🛡` is mandatory; guard set logged.

---

## 9) Execution Model — State Machine
States: `PLAN → ORDER → ALIGN → EXECUTE → PROVE → PROMOTE → {COMMIT|ROLLBACK}`.
- **PLAN:** SpaceState/Config declared (immutable for run).
- **ORDER:** Zoom+dihedral receipts; may loop with parameter refinements.
- **ALIGN:** Weyl ops (⇄,⥁) to target chambers; canonicalize; log steps.
- **EXECUTE:** Task ops; DR* routing; bucket/rail assignment.
- **PROVE:** Build & run glyph proof; collect invariants.
- **PROMOTE:** Evaluate gates; if pass → COMMIT (ledger), else → ROLLBACK (overlay restore optional).

---

## 10) Telemetry & Observability
- **Receipts on:** zoom tiers, bucket assigns, rails, cache hits, proofs, promotions.
- **Metrics:** tokenized length (text vs bytecode), MDL gains, occupancy histograms, p-values, parallel speedups, rollback savings.
- **Trace IDs:** propagate across receipts; overlays store the trace root.

---

## 11) Failure Classes & Recovery
- **F1 Parser/bytecode:** invalid glyph chain → reject with location.
- **F2 Safety:** guard violation → block op; emit safety receipt.
- **F3 Gate:** promotion fail → rollback to prior overlay.
- **F4 Bias:** mixer bias tripwire → auto-reseed `Mix`, reassign; log incident.
- **F5 Resource:** token starvation → throttle background/parallel; reschedule.

---

## 12) Extension Points
- **Lattices:** swap `E8` for `Leech` (Gram+Golay+Conway glue) via adapter.
- **Forces:** add op families (e.g., diffusion, braids) with typed signatures.
- **Scheduler:** plug alternative `DR*` salts/load-balancing policies.
- **Gates:** add domain-specific thresholds (e.g., accuracy, coverage).

---

## 13) Minimal APIs (pseudo)
```python
# Glyphs
E8v = ↑(dom)            # embed
E8v = ⇄(E8v)            # reflect
E8v = ⥁(E8v, theta)     # flow
x   = ↓(E8v)            # project

# Ordering
zr  = zoom_report(order, zooms)

# Buckets
place, rec = assign(token)   # returns (bucket|rail, receipt)

# Proof/Promotion
ok  = ✓(proof)
rec = promote(ops, thresholds)

# Overlays
💾(key, overlay) ; overlay = 📂(key)
```

---

## 14) Example Macro Flows
- **Embed→Flow→Project:** `↑⥁↓`
- **Snap→Verify→Store:** `⊞✓💾`
- **Parallel→Integrate:** `∥∫`
- **Assign→Ledger:** `assign → 🧾 → 📋`

---

## 15) Compliance & Audit
- Deterministic bucket/rail placement with receipts.
- Reproducible runs via frozen SpaceState + Config + seeds.
- Bias testing mandated; p-values attached.
- Safety guards enumerated and versioned.

---

### TL;DR
Think in typed glyphs. Route by DR* & dihedral-CRT. Canonicalize→cache→overlay. Prove-or-don’t-promote. Everything emits a receipt.

