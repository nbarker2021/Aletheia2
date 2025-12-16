"""
CQE Overlay data structure - core representation of content in E8 space
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import hashlib


@dataclass
class CQEOverlay:
    """
    Core CQE overlay representing content in E8 framework.

    Structure:
    - 248-slot activation mask (240 E8 roots + 8 Cartan lanes)
    - Weights and phases for active slots
    - Pose metadata (gauge, symmetry, domain info)
    - Content-addressed hash ID for deterministic retrieval

    Attributes:
        present: 248-bit activation mask (bool array)
        w: Weights for all slots (float array)
        phi: Phases/angles for all slots (float array)
        pose: Metadata dictionary with domain info
        hash_id: Content-addressed unique identifier
        provenance: List of transformation history
    """

    present: np.ndarray          # 248-bit activation mask
    w: np.ndarray                # Weights for active slots
    phi: np.ndarray              # Phases (Coxeter angles)
    pose: Dict[str, Any]         # Gauge and metadata
    hash_id: Optional[str] = None
    provenance: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate overlay structure"""
        assert len(self.present) == 248, f"Must have 248 slots, got {len(self.present)}"
        assert len(self.w) == 248, f"Weights must match slots, got {len(self.w)}"
        assert len(self.phi) == 248, f"Phases must match slots, got {len(self.phi)}"

    @property
    def active_slots(self) -> np.ndarray:
        """Return indices of active slots"""
        return np.where(self.present)[0]

    @property
    def cartan_active(self) -> int:
        """Return count of active Cartan lanes (slots 240-247)"""
        return int(np.sum(self.present[240:248]))

    @property
    def root_active(self) -> int:
        """Return count of active root slots (slots 0-239)"""
        return int(np.sum(self.present[:240]))

    @property
    def is_canonical(self) -> bool:
        """Check if overlay has been canonicalized"""
        return self.hash_id is not None

    @property
    def sparsity(self) -> float:
        """Compute sparsity ratio (active/total)"""
        return np.sum(self.present) / 248.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize overlay to dictionary"""
        return {
            'present': self.present.tolist(),
            'w': self.w.tolist(),
            'phi': self.phi.tolist(),
            'pose': self.pose,
            'hash_id': self.hash_id,
            'provenance': self.provenance
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CQEOverlay':
        """Deserialize overlay from dictionary"""
        return cls(
            present=np.array(data['present'], dtype=bool),
            w=np.array(data['w']),
            phi=np.array(data['phi']),
            pose=data['pose'],
            hash_id=data.get('hash_id'),
            provenance=data.get('provenance', [])
        )

    def copy(self) -> 'CQEOverlay':
        """Create deep copy of overlay"""
        return CQEOverlay(
            present=self.present.copy(),
            w=self.w.copy(),
            phi=self.phi.copy(),
            pose=self.pose.copy(),
            hash_id=self.hash_id,
            provenance=self.provenance.copy()
        )

    def compute_hash(self) -> str:
        """
        Compute content-addressed hash for overlay.
        Uses present mask, weights, and phases.
        """
        canonical_bytes = (
            self.present.tobytes() + 
            np.round(self.w, 8).tobytes() + 
            np.round(self.phi, 9).tobytes()
        )
        return hashlib.sha256(canonical_bytes).hexdigest()[:16]

    def __repr__(self) -> str:
        return (
            f"CQEOverlay(active={np.sum(self.present)}/248, "
            f"cartan={self.cartan_active}/8, "
            f"hash={self.hash_id[:8] if self.hash_id else 'None'})"
        )
