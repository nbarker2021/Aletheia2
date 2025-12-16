"""Ledger utilities for CQE (append-only JSONL receipts)."""
from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, Any

def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file. Creates parents if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
