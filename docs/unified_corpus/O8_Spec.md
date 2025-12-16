# O8 — a base‑8, shape‑pack programming language

**Core idea:** Everything is geometry-first. Code is a stream of **4‑bit shape packs** (strings over `{a,b}` with `a=1, b=0`) grouped into **octets**. Numbers default to **base‑8**. The runtime snaps work into E8‑caps and only **commits at boundaries**; interiors are simulated and logged, not applied.

## File extension
`.o8`

## Header
```
scene MarsLogistics
dim 20             # base‑8 by default (here “20” octal = 16 decimal)
gauge auto
sidecar {
  "units": "SI",
  "mode": "meaning-silent"
}
```

## Body
Blocks of 8 instructions:
```
octet {
  abba ROLE pilot;                     # set a role (side effect only in ledger)
  baab POSE;                           # choose gauge rotation maximizing pose alignment
  baaa TICKET;                         # mark boundary tickets (margin ≤ τ_w)
  baba ANNIHILATE;                     # drop ultra-thin tickets (≤ τ_annih)
  babb SNAP;                           # commit at caps (ledger only in the minimal runtime)
  bbab MIRROR;                         # palindromic check receipt
  bbaa RATCHET;                        # tighten thresholds by 10%
  abab EMIT "o8_receipts.json";        # write receipts
}
```

Shape packs are **4 letters** over `{a,b}`; they map to opcodes by symmetry (palindromic ones bias to parity ops):
- `bbbb` → `NOP`
- `bbba` → `DLIFT` (small geometric nudge)
- `bbab` → `MIRROR`
- `bbaa` → `RATCHET`
- `babb` → `SNAP`
- `baba` → `ANNIHILATE`
- `baab` → `POSE`
- `baaa` → `TICKET`
- `abbb` → `BIND key=value`
- `abba` → `ROLE name`
- `abab` → `EMIT "file"`
- `abaa` → `CALL lang "func" [json_args]`
- `aabb` → `MAP route`
- `aaba` → `FORK`
- `aaab` → `JOIN`
- `aaaa` → `ASSERT expr`

## Numbers (base‑8 primary)
Unprefixed integers are parsed as **octal**. Use `0x`, `0b`, or `0d` to force hex, binary, or decimal.

## Semantics (minimal runtime)
- **POSE** chooses a rotation from a fixed set (I, H₈, QR, signflip×H₈) that maximizes modal pose on the first 8D block.
- **TICKET** flags boundary points via E8 coset margin ≤ τ_w (OR across 8D blocks).
- **SNAP** logs a cap commit event (no destructive mutation in the minimal runtime).
- **ANNIHILATE** removes tickets with margin ≤ τ_annih (rails).
- **RATCHET** tightens τ_w, τ_annih by 10%.
- **EMIT** writes the current ledger (append-only receipts) to JSON.
- **CALL** delegates to language adapters (only `py` safe adapter included: `np`, `math`).

## Why this language?
- **Base‑8 tokens** align with octet machinery and CQE caps; **4‑bit packs** are the “verbs.”
- **Meaning lives in sidecars**; geometry decides execution. This keeps runs meaning‑silent until commit.
- **Delegation** is explicit: you can call the best tool/language for a subtask; O8 remains the governing lattice.

## CLI
```
python3 o8.py examples/hello.o8 --out run1
```
Artifacts land in `--out`:
- `ledger.jsonl` — append‑only receipts of every step
- `summary.json` — scene, gauge, dim, sidecar
- plus any `EMIT` files you wrote

