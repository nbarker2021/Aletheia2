# Intake Harness + Bit Window (RAG-Control Pack)

This pack wires your **glyphs + DataFrames** into a RAG-controlled output flow, adds a **bit-window** execution controller, and sets up a harness to ingest brand‑new reference docs while staying deterministic.

---

## 0) What you get
- **bit_window.yml** — defines the 32‑bit window (8 lanes × 4 phases) and ensures first/last bits are locked (Native/Output).
- **harness.py** — orchestrates intake → process (ODS‑320) → RAG cards → ALT bundles (when “through” mode triggers). All actions are gated by the bit window.
- **rag_cards.py** — builds RAG‑Card v1 CSV from chunks + primitives with auditable locations.

> Compatible with the previously provided `ops.yml`, `glyphs.py`, and `orchestrator.py`.

---

## 1) Bit Window — Spec (BW‑32)

**Layout:** 32 bits = **8 lanes (A–H)** × **4 phases (1–4)**

- **Phase 1 — Native Actions (ingest/process)**
- **Phase 2 — RAG Build (retrieve/align/ground/verbalize/verify)**
- **Phase 3 — ALT Exploration (drivers/policies/scoring variants + diffs)**
- **Phase 4 — Output Commit (user‑facing bundles & summaries)**

**Indexing:** `bit[p][l]` with `p∈{1..4}` and `l∈{A..H}`.

**Locked bits:**
- **bit[1][A] = 1** (must allow Native Actions to run minimal intake)
- **bit[4][H] = 1** (must allow Output Commit so results can be emitted)

This satisfies your constraint: *“first bit and last bit forced to 1: All native actions / output of results to user.”*

**Semantics per phase:**
- **P1 Native**: ingest, chunk, primitives, shell/bucket, glyphs, ledger, seeds, graph.
- **P2 RAG**: build RAG candidates, claim–evidence cards, process‑first summary, no human inference beyond quotes and structurally derived claims.
- **P3 ALT**: run curated alternatives (scheduler/policy/scoring), write diff report.
- **P4 Output**: package ODS‑320, rag_cards, explain_first, diffs; render user summary.

**Composition to Base‑320:** one **B320 packet = 10 consecutive BW‑32 windows**. Window headers hold: `run_id`, `window_id (0..9)`, `glyph_triad_head`, `glyph_triad_tail`, and the 32‑bit mask.

---

## 2) bit_window.yml (default)
```yaml
bit_window:
  # Phases 1..4, Lanes A..H
  # 1 = enabled, 0 = disabled; locked bits enforced at runtime
  P1:
    A: 1  # LOCKED (Native minimal)
    B: 1
    C: 1
    D: 1
    E: 1
    F: 1
    G: 1
    H: 1
  P2:
    A: 1  # Retrieve
    B: 1  # Align
    C: 1  # Ground
    D: 1  # Verbalize
    E: 1  # Verify
    F: 0  # spare
    G: 0  # spare
    H: 0  # spare
  P3:
    A: 1  # ALT scheduler
    B: 1  # ALT policy (strict-even)
    C: 1  # ALT scoring (stability)
    D: 0  # ALT glyph overlay
    E: 0  # ALT packet base
    F: 0
    G: 0
    H: 0
  P4:
    A: 1  # ODS-320 bundle
    B: 1  # RAG-Card CSV
    C: 1  # diff_report.md
    D: 1  # explain_first.md
    E: 1  # operator_summary.md
    F: 0
    G: 0
    H: 1  # LOCKED (Output commit)
```

---

## 3) rag_cards.py (RAG‑Card v1 builder)
```python
# rag_cards.py — build RAG-Card v1 from deterministic artifacts
from dataclasses import dataclass
from typing import List, Dict, Any
import re
import pandas as pd

@dataclass
class EvidenceSpan:
    doc_id: str
    para_idx: int
    char_start: int
    char_end: int
    text: str
    glyph_triad: str
    shell: int
    bucket_id: int

@dataclass
class RagCard:
    claim_id: str
    claim_text: str
    confidence: str  # 'high'|'med'|'low'
    evidence: List[EvidenceSpan]

# Minimal sentence splitter; replace with robust NLP if desired (kept deterministic)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def build_cards(doc_texts: Dict[str,str], df_chunks: pd.DataFrame, df_prims: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for doc_id, text in doc_texts.items():
        # paragraph index map (simple split)
        paras = [p for p in re.split(r"\n{2,}", text) if p.strip()]
        para_offsets = []
        offset = 0
        for p in paras:
            start = text.find(p, offset)
            para_offsets.append((p, start))
            offset = start + len(p)

        # candidate keywords from primitives (by frequency)
        prims = df_prims[df_prims['doc_id']==doc_id]['primitive'].value_counts()
        top = prims.index.tolist()[:8]
        if not top:
            continue

        # build claims from top primitives (simple templates)
        claim_no = 0
        for kw in top:
            claim_no += 1
            claim_id = f"{doc_id}:C{claim_no:02d}"
            claim_text = f"The document emphasizes '{kw}'."
            # evidence: sentences containing keyword
            ev_list: List[EvidenceSpan] = []
            for para_idx, (p, p_start) in enumerate(para_offsets):
                if kw.lower() in p.lower():
                    # find first sentence with hit
                    sents = SENT_SPLIT.split(p)
                    sent_offset = 0
                    for s in sents:
                        if kw.lower() in s.lower():
                            s_start = text.find(s, p_start + sent_offset)
                            s_end = s_start + len(s)
                            # map to nearest chunk by char range
                            dfc = df_chunks[(df_chunks['doc_id']==doc_id) &
                                             (df_chunks['offset_start']<=s_start) &
                                             (df_chunks['offset_end']>=s_end)]
                            if dfc.empty:
                                dfc = df_chunks[df_chunks['doc_id']==doc_id].head(1)
                            row = dfc.iloc[0]
                            ev_list.append(EvidenceSpan(
                                doc_id=doc_id, para_idx=para_idx,
                                char_start=int(s_start), char_end=int(s_end), text=s.strip(),
                                glyph_triad=row['glyph_triad'], shell=int(row['shell']), bucket_id=int(row['bucket_id'])
                            ))
                            break
            confidence = 'high' if len(ev_list)>=2 else ('med' if len(ev_list)==1 else 'low')
            for ev in ev_list:
                rows.append({
                    'doc_id': ev.doc_id,
                    'claim_id': claim_id,
                    'claim_text': claim_text,
                    'quote': ev.text,
                    'para_idx': ev.para_idx,
                    'char_start': ev.char_start,
                    'char_end': ev.char_end,
                    'glyph_triad': ev.glyph_triad,
                    'shell': ev.shell,
                    'bucket_id': ev.bucket_id,
                    'confidence': confidence,
                    'keyword': kw,
                })
    return pd.DataFrame(rows)
```

---

## 4) harness.py (intake controller + bit window + RAG output)
```python
# harness.py — ties bit window + orchestrator + rag cards into one controllable run
import argparse, json
from pathlib import Path
import yaml
import pandas as pd
from hashlib import sha256

from orchestrator import process_docs, run_alt_variants, write_diff_report
from rag_cards import build_cards

LOCK_FIRST = ('P1','A')
LOCK_LAST  = ('P4','H')


def load_yaml(p: Path):
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(p: Path, data: dict):
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def enforce_locks(bits: dict):
    bits['P1']['A'] = 1
    bits['P4']['H'] = 1
    return bits


def bits_enabled(bits: dict, phase: str, lane: str) -> bool:
    return bool(bits.get(phase, {}).get(lane, 0))


def read_texts(input_dir: Path):
    texts = {}
    for p in sorted(input_dir.glob('**/*')):
        if p.is_file() and p.suffix.lower() in {'.txt', '.md', '.docx'}:
            try:
                if p.suffix.lower()=='.docx':
                    import docx
                    d = docx.Document(str(p))
                    text = '\n\n'.join([para.text for para in d.paragraphs])
                else:
                    text = p.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                text = p.read_text(encoding='utf-8', errors='ignore')
            texts[p.stem] = text
    return texts


def main():
    ap = argparse.ArgumentParser(description='Bit Window Harness (RAG-Control)')
    ap.add_argument('--ops', type=Path, default=Path('ops.yml'))
    ap.add_argument('--bits', type=Path, default=Path('bit_window.yml'))
    ap.add_argument('--input', type=Path, default=Path('input'))
    ap.add_argument('--out', type=Path, default=Path('out'))
    ap.add_argument('--through', action='store_true', help='force through-mode ALT variants')
    args = ap.parse_args()

    cfg = load_yaml(args.ops)
    bits = enforce_locks(load_yaml(args.bits)['bit_window'])
    docs_text = read_texts(args.input)

    run_id = sha256(('harness:' + str(sorted(docs_text.keys()))).encode('utf-8')).hexdigest()[:8]

    out_root = args.out
    primary_out = out_root / 'bundle_primary'

    # Phase 1 — Native Actions (A..H lanes all map to core process_docs)
    if any(bits_enabled(bits,'P1',lane) for lane in 'ABCDEFGH'):
        primary = process_docs(cfg, docs_text, run_id, primary_out)
    else:
        raise SystemExit('P1 disabled — cannot proceed')

    # Phase 2 — RAG Build (needs at least A..E if you want full chain)
    if any(bits_enabled(bits,'P2',lane) for lane in 'ABCDEFGH'):
        # Build RAG cards only if P2 is enabled
        df_cards = build_cards(docs_text, primary['df_chunks'], primary['df_primitives'])
        df_cards.to_csv(out_root / 'bundle_primary' / 'rag_cards.csv', index=False)

    # Phase 3 — ALT Exploration
    variants = []
    if any(bits_enabled(bits,'P3',lane) for lane in 'ABCDEFGH') and (args.through or True):
        variants = run_alt_variants(cfg, docs_text, run_id, out_root)
        write_diff_report(primary, variants, out_root)

    # Phase 4 — Output Commit (must be enabled at least at H)
    if not any(bits_enabled(bits,'P4',lane) for lane in 'ABCDEFGH'):
        raise SystemExit('P4 disabled — output commit locked on H but off elsewhere')

    # Minimal operator summary
    lines = []
    lines.append('# operator_summary.md')
    lines.append(f'- Docs: {len(docs_text)}')
    lines.append(f'- Chunks: {len(primary["df_chunks"])}  |  Primitives: {len(primary["df_primitives"])}')
    if (out_root / 'bundle_primary' / 'rag_cards.csv').exists():
        dfc = pd.read_csv(out_root / 'bundle_primary' / 'rag_cards.csv')
        lines.append(f'- RAG cards: {len(dfc.claim_id.unique())} claims / {len(dfc)} evidence rows')
    if variants:
        lines.append(f'- ALT bundles: {len(variants)} (see diff_report.md)')
    (out_root / 'operator_summary.md').write_text('\n'.join(lines), encoding='utf-8')

    print('Harness done. See', str(out_root))

if __name__ == '__main__':
    main()
```

---

## 5) How to use the harness

1. Place your **reference docs** (new .txt/.md/.docx) in `input/`.
2. Keep or tweak `ops.yml` and `bit_window.yml`. (Locks are enforced; you can toggle other bits as desired.)
3. Run:
```bash
python harness.py --ops ops.yml --bits bit_window.yml --input input --out out --through
```
4. You’ll get:
   - `bundle_primary/` (ODS‑320 + rag_cards.csv + explain_first.md)
   - `bundle_alt_01..` (if through-mode or triggers)
   - `diff_report.md` and `operator_summary.md`

**Note:** The RAG cards are **process‑grounded**: each claim has evidence rows with `(doc_id, para_idx, char_start, char_end, glyph_triad, shell, bucket_id)` for audit.

---

## 6) Why this matches your constraints
- **First and last bits locked** ensure minimal Native intake and Output commit.
- **Bit-window windows → B320 packets**: pack 10 consecutive windows; headers hold glyph head/tail + bitmask, giving a compact, auditable commit unit.
- **Resource-light work windows**: yes—these substeps are light; the window model lets us do end‑to‑end results in one organized step without background tasks.

---

### Ready for your refs
Hand me the reference docs next. I’ll use the harness spec above to run **Explain First** and produce **RAG‑Card v1** outputs, with ALT bundles when useful, all governed by **BW‑32** and sealed into **Base‑320** packets for your audit trail.

