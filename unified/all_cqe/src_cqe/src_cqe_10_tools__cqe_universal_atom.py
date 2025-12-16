"""
CQE Universal Atom - Multi-Slice Geometric Data Structure

The UniversalAtom is the core data structure that carries coordinates
across all 37 mathematical slices in the CQE system.
"""

import uuid
import hashlib
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
import numpy as np

@dataclass
class SliceData:
    """Base class for slice-specific data"""
    energy: float = 0.0
    parity_state: bytes = field(default_factory=lambda: b'\x00' * 8)
    bands: Dict[str, int] = field(default_factory=lambda: {"band8": 0, "band24": 0, "tile4096": 0})
    validated: bool = False
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

@dataclass  
class UniversalAtom:
    """
    Universal Atom with coordinates across all 37 mathematical slices

    Each atom represents a point in the high-dimensional mathematical space
    defined by the Cartesian product of all slice spaces, embedded in E8.
    """

    # Core identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sha256: str = field(default="")
    raw_data: Any = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # E8 Foundation coordinates
    e8_coordinates: List[float] = field(default_factory=lambda: [0.0] * 8)
    lattice_position: List[int] = field(default_factory=lambda: [0] * 8)

    # Slice data storage - all 37 slices
    slice_data: Dict[str, SliceData] = field(default_factory=dict)

    # Global governance
    governance_bands: Dict[str, int] = field(default_factory=lambda: {"band8": 0, "band24": 0, "tile4096": 0})
    parity_lanes: bytes = field(default_factory=lambda: b'\x00' * 8)
    merkle_path: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize atom after creation"""
        if not self.sha256:
            self.sha256 = self._compute_content_hash()

    def _compute_content_hash(self) -> str:
        """Compute SHA-256 hash of atom content"""
        content = {
            "raw_data": str(self.raw_data),
            "e8_coordinates": self.e8_coordinates,
            "created_at": self.created_at
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def set_slice_data(self, slice_name: str, data: SliceData):
        """Set data for a specific slice"""
        self.slice_data[slice_name] = data

    def get_slice_data(self, slice_name: str) -> Optional[SliceData]:
        """Get data for a specific slice"""
        return self.slice_data.get(slice_name)

    def get_mathematical_content(self) -> Dict[str, Any]:
        """Extract mathematical content for slice processing"""
        return {
            "raw_data": self.raw_data,
            "e8_coords": self.e8_coordinates,
            "lattice_pos": self.lattice_position,
            "id": self.id,
            "hash": self.sha256
        }

    def get_total_energy(self) -> float:
        """Compute total energy across all slices"""
        return sum(data.energy for data in self.slice_data.values())

    def get_active_slices(self) -> List[str]:
        """Get list of slices with validated data"""
        return [name for name, data in self.slice_data.items() if data.validated]

    def update_e8_coordinates(self, coordinates: List[float]):
        """Update E8 lattice coordinates"""
        if len(coordinates) != 8:
            raise ValueError("E8 coordinates must be 8-dimensional")
        self.e8_coordinates = coordinates

    def to_dict(self) -> Dict[str, Any]:
        """Convert atom to dictionary representation"""
        return {
            "id": self.id,
            "sha256": self.sha256,
            "raw_data": self.raw_data,
            "created_at": self.created_at,
            "e8_coordinates": self.e8_coordinates,
            "lattice_position": self.lattice_position,
            "slice_data": {k: asdict(v) for k, v in self.slice_data.items()},
            "governance_bands": self.governance_bands,
            "parity_lanes": self.parity_lanes.hex(),
            "merkle_path": self.merkle_path,
            "total_energy": self.get_total_energy(),
            "active_slices": self.get_active_slices()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UniversalAtom':
        """Create atom from dictionary representation"""
        atom = cls(
            id=data["id"],
            sha256=data["sha256"],
            raw_data=data["raw_data"],
            created_at=data["created_at"]
        )

        atom.e8_coordinates = data["e8_coordinates"]
        atom.lattice_position = data["lattice_position"]
        atom.governance_bands = data["governance_bands"]
        atom.parity_lanes = bytes.fromhex(data["parity_lanes"])
        atom.merkle_path = data["merkle_path"]

        # Reconstruct slice data
        for slice_name, slice_dict in data["slice_data"].items():
            slice_data = SliceData(**slice_dict)
            atom.set_slice_data(slice_name, slice_data)

        return atom

    def __repr__(self):
        return f"UniversalAtom(id={self.id[:8]}..., slices={len(self.slice_data)}, energy={self.get_total_energy():.3f})"
