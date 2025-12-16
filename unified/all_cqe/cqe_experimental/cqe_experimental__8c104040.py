"""
Handshake record for provenance tracking
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime


@dataclass
class HandshakeRecord:
    """
    Single handshake record in MORSR provenance chain.

    Tracks each operator application with:
    - Operator identity
    - Φ before/after values
    - Acceptance decision
    - Overlay hash for reproducibility
    """

    operator_name: str
    phi_before: float
    phi_after: float
    delta_phi: float
    accepted: bool
    reason: str
    overlay_hash: Optional[str]
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        return {
            'operator_name': self.operator_name,
            'phi_before': self.phi_before,
            'phi_after': self.phi_after,
            'delta_phi': self.delta_phi,
            'accepted': self.accepted,
            'reason': self.reason,
            'overlay_hash': self.overlay_hash,
            'timestamp': self.timestamp
        }


class HandshakeLogger:
    """Logger for MORSR handshake records"""

    def __init__(self):
        self._log: List[HandshakeRecord] = []

    def log(self, record: HandshakeRecord):
        """Add handshake record to log"""
        self._log.append(record)

    def get_log(self) -> List[HandshakeRecord]:
        """Retrieve all handshake records"""
        return self._log.copy()

    def clear(self):
        """Clear the log"""
        self._log.clear()

    def get_accepted(self) -> List[HandshakeRecord]:
        """Get only accepted handshakes"""
        return [h for h in self._log if h.accepted]

    def get_rejected(self) -> List[HandshakeRecord]:
        """Get only rejected handshakes"""
        return [h for h in self._log if not h.accepted]

    def acceptance_rate(self) -> float:
        """Compute acceptance rate"""
        if not self._log:
            return 0.0
        return sum(1 for h in self._log if h.accepted) / len(self._log)
