
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decoherence callbacks: recall and compare by receipt."""
from __future__ import annotations
import json, math
from typing import Dict, Any, List, Optional, Tuple
from coherence_metrics import composite_coherence, collapse_detector, embedding_alignment
from state_store import StateStore

def recall_pair_and_compare(store: StateStore, rid_a: str, rid_b: str) -> Dict[str,Any]:
    A = store.load(rid_a) or {}
    B = store.load(rid_b) or {}
    pts_a = A.get("points") or []
    pts_b = B.get("points") or []
    emb_a = A.get("embedding") or []
    emb_b = B.get("embedding") or []
    coh_a = composite_coherence(pts_a)
    coh_b = composite_coherence(pts_b)
    coll = collapse_detector(pts_a, pts_b)
    align = embedding_alignment(emb_a, emb_b) if emb_a and emb_b else None
    return {"A": {"receipt": rid_a, "coherence": coh_a},
            "B": {"receipt": rid_b, "coherence": coh_b},
            "collapse": coll, "embedding_alignment": align}

def timeline_metrics(store: StateStore, receipts: List[str]) -> Dict[str, Any]:
    series = []
    for rid in receipts:
        doc = store.load(rid)
        if not doc: continue
        coh = composite_coherence(doc.get("points") or [])
        series.append({"receipt": rid, "ts": doc.get("ts"), "score": coh["score"], "angular": coh["angular"], "radial": coh["radial"], "spectral_entropy": coh["spectral_entropy"]})
    series.sort(key=lambda r: r.get("ts", 0))
    return {"timeline": series}
