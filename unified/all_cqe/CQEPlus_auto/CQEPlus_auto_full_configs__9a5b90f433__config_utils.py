"""Config helpers for CQE. Uses pydantic if present; falls back to env-only."""
from __future__ import annotations
import os
from pathlib import Path

try:
    from pydantic import BaseModel
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore

class Settings(BaseModel):  # type: ignore[misc]
    """Runtime settings for CQE components."""
    home: Path = Path.home() / ".cqe"
    runs_dir: Path = Path("runs")
    logs_dir: Path = Path("logs")
    data_dir: Path = Path("data")

    @classmethod
    def from_env(cls) -> "Settings":  # type: ignore[override]
        base = Path(os.environ.get("CQE_HOME", str(Path.home() / ".cqe")))
        return cls(
            home=base,
            runs_dir=Path(os.environ.get("CQE_RUNS", "runs")),
            logs_dir=Path(os.environ.get("CQE_LOGS", "logs")),
            data_dir=Path(os.environ.get("CQE_DATA", "data")),
        )
