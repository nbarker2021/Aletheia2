"""
Configuration and environment resolution for CQE.
Respects environment variables: CQE_HOME, CQE_RUNS, CQE_LOGS, CQE_DATA
Falls back to sensible defaults inside the current workspace.
"""
from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass

@dataclass(slots=True)
class Settings:
    home: Path
    runs_dir: Path
    logs_dir: Path
    data_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        base = Path(os.environ.get("CQE_HOME", Path.home() / ".cqe"))

        def resolve(name: str, default: str) -> Path:
            val = os.environ.get(name)
            if val:
                p = Path(val)
                return p if p.is_absolute() else Path.cwd() / p
            return Path.cwd() / default

        runs = resolve("CQE_RUNS", "runs")
        logs = resolve("CQE_LOGS", "logs")
        data = resolve("CQE_DATA", "data")
        return cls(base, runs, logs, data)
