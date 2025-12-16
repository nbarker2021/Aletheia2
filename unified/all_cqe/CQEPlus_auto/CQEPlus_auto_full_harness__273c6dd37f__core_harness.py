"""CQE Harness Core
=====================

This module defines the **reference harness** entrypoint `run_harness` and related utilities.
The harness is **geometry-first**: it treats poses and parity as control variables and emits
ledgered receipts for every action. It is intentionally minimal here, so you can plug in your
real E8/Leech/Monster validators and MORSR/ALENA operators.
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple
import json, hashlib
from datetime import datetime

@dataclass
class HarnessResult:
    """Structured result record for one harness run."""
    stamp: str
    input: str
    pose: Dict[str, Any]
    metrics: Dict[str, Any]
    status: str = "ok"
    notes: str = ""

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def compute_pose_metrics(text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute pose and metrics for the given input (placeholder)."""
    pose = {"kind": "placeholder", "hash": _sha256(text)[:16]}
    metrics = {"dphi": -0.0, "parity": [1]*8, "energy": 0.0}
    return pose, metrics

def legalize_update(pose: Dict[str, Any], metrics: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply a legality repair step if needed (placeholder)."""
    return pose, metrics

def write_receipt(outdir: Path, data: Dict[str, Any]) -> None:
    """Append a JSONL receipt under `outdir/ledger.jsonl`. Creates directories if needed."""
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "ledger.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n" )

def run_harness(text: str, outdir: Path) -> Dict[str, Any]:
    """Run the CQE harness on `text` and write a ledger receipt in `outdir`."""
    stamp = datetime.utcnow().isoformat() + "Z"
    pose, metrics = compute_pose_metrics(text)
    pose2, metrics2 = legalize_update(pose, metrics)
    receipt = HarnessResult(
        stamp=stamp, input=text, pose=pose2, metrics=metrics2, status="ok",
        notes="CQE reference harness: plug in production MORSR/ALENA/E8 logic."
    )
    data = asdict(receipt)
    write_receipt(outdir, data)
    return data
