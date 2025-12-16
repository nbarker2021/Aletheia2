
from dataclasses import dataclass
from typing import Dict, Any
import json, os

@dataclass
class Config:
    seed: int
    receipts_dir: str
    artifacts_dir: str
    enable_mandelbrot: bool
    enable_toroidal: bool
    phi_weights: Dict[str, float]
    validation_thresholds: Dict[str, float]

    @staticmethod
    def from_file(path: str) -> "Config":
        with open(path, "r") as f:
            d = json.load(f)
        return Config(**d)
