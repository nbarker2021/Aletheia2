"""
CQE Harness Core — deterministic, dependency-light.
Writes JSONL receipts to the given output directory.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, Tuple
import hashlib, json

@dataclass(slots=True)
class HarnessResult:
    stamp: str
    input: str
    pose: Dict[str, Any]
    metrics: Dict[str, Any]
    status: str = "ok"
    notes: str = ""

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def compute_pose_metrics(text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pose = {"kind": "placeholder", "hash": _sha256(text)}
    metrics = {"dphi": 0.0, "parity": [1]*8, "energy": 0.0}
    return pose, metrics

def legalize_update(pose: Dict[str, Any], metrics: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return pose, metrics

def write_receipt(outdir: Path, data: Dict[str, Any]) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    ledger = outdir / "ledger.jsonl"
    with ledger.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")
    return ledger

def run_harness(text: str, outdir: Path) -> Dict[str, Any]:
    stamp = datetime.now(UTC).isoformat()
    pose, metrics = compute_pose_metrics(text)
    pose, metrics = legalize_update(pose, metrics)
    result = HarnessResult(stamp=stamp, input=text, pose=pose, metrics=metrics,
                           notes="CQE reference harness; replace with production logic.")
    data = asdict(result)
    write_receipt(outdir, data)
    return data
