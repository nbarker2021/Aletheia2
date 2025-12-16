
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List
import statistics

@dataclass
class IntegrityPanel:
    accepts: int = 0
    rejects: int = 0
    plateau_ticks: int = 0
    parity_fixes: int = 0
    parity_breaks: int = 0
    phi_series: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepts": self.accepts,
            "rejects": self.rejects,
            "plateau_ticks": self.plateau_ticks,
            "parity_fixes": self.parity_fixes,
            "parity_breaks": self.parity_breaks,
            "phi_mean": (statistics.mean(self.phi_series) if self.phi_series else None),
            "phi_last": (self.phi_series[-1] if self.phi_series else None),
        }
