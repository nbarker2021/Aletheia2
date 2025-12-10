"""WorldForge – minimal manifold generator for Scene8.

Produces a World object with:
    • e8_state (np.ndarray shape (8,))
    • step(dt) → updates e8_state, returns thermodynamic snapshot
    • boundary_ok flag (dummy for now)
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any
import datetime, random

class World:
    def __init__(self, seed: int | None = None, worldtype: str = 'toroidal'):
        self.rng = np.random.default_rng(seed)
        self.e8_state = self.rng.standard_normal(8)
        self.worldtype = worldtype
        self.weyl_chamber = int(self.rng.integers(0, 248))
        self.free_energy = 0.0
        self.entropy = 0.0
        self.boundary_ok = True

    def step(self, dt: float = 1e-3) -> Dict[str, Any]:
        # random walk in E8 space (placeholder dynamics)
        delta = self.rng.standard_normal(8) * np.sqrt(dt)
        self.e8_state += delta
        # simple harmonic potential energy
        self.free_energy = float(0.5 * np.sum(self.e8_state ** 2))
        # entropy proxy: random small drift near zero
        self.entropy += float(self.rng.normal(0, 1e-4))
        # boundary check dummy
        self.boundary_ok = np.all(np.abs(self.e8_state) < 10)
        return {
            'free_energy': self.free_energy,
            'entropy': self.entropy,
            'boundary_ok': self.boundary_ok,
            'timestamp': datetime.datetime.utcnow().isoformat()
        }

def spawn_world(spec: Dict[str, Any] | None = None) -> World:
    spec = spec or {}
    return World(seed=spec.get('seed'), worldtype=spec.get('worldtype', 'toroidal'))
