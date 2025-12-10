"""
Overlay System - Axiom A Implementation
Cartan-Quadratic Equivalence (CQE) System

From Whitepaper:
"States are overlays on E₈ with binary activations, optional weights/phase φ,
and immutable pose. Domain adapters embed native objects to overlays."

This module implements the fundamental state representation in CQE.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import hashlib
import json


@dataclass(frozen=True)
class ImmutablePose:
    """
    Immutable pose in E₈ space.
    
    Once created, the pose cannot be modified. This ensures state integrity
    and enables deterministic replay.
    """
    position: Tuple[float, ...]  # 8D position in E₈
    orientation: Tuple[float, ...]  # 8D orientation vector
    timestamp: float  # Creation timestamp
    
    def __post_init__(self):
        """Validate dimensions."""
        if len(self.position) != 8:
            raise ValueError(f"Position must be 8D, got {len(self.position)}D")
        if len(self.orientation) != 8:
            raise ValueError(f"Orientation must be 8D, got {len(self.orientation)}D")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'position': list(self.position),
            'orientation': list(self.orientation),
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImmutablePose':
        """Create from dictionary."""
        return cls(
            position=tuple(data['position']),
            orientation=tuple(data['orientation']),
            timestamp=data['timestamp']
        )
    
    def hash(self) -> str:
        """Compute hash for identity."""
        data = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class Overlay:
    """
    Overlay state on E₈ lattice.
    
    From Axiom A:
    - Binary activations: Which E₈ roots are active
    - Optional weights: Strength of each activation
    - Optional phase φ: Geometric phase
    - Immutable pose: Fixed position/orientation
    
    This is the fundamental state representation in CQE.
    """
    
    # Core fields (from Axiom A)
    e8_base: np.ndarray  # 8D base point in E₈
    activations: np.ndarray  # Binary mask (240 roots)
    weights: Optional[np.ndarray] = None  # Optional weights (240 values)
    phase: Optional[float] = None  # Optional phase φ
    pose: Optional[ImmutablePose] = None  # Immutable pose
    
    # Metadata
    overlay_id: Optional[str] = None
    parent_id: Optional[str] = None
    creation_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate overlay structure."""
        # Validate e8_base
        if self.e8_base.shape != (8,):
            raise ValueError(f"e8_base must be 8D, got shape {self.e8_base.shape}")
        
        # Validate activations
        if self.activations.shape != (240,):
            raise ValueError(f"activations must be 240D, got shape {self.activations.shape}")
        if not np.all((self.activations == 0) | (self.activations == 1)):
            raise ValueError("activations must be binary (0 or 1)")
        
        # Validate weights if present
        if self.weights is not None:
            if self.weights.shape != (240,):
                raise ValueError(f"weights must be 240D, got shape {self.weights.shape}")
        
        # Generate overlay_id if not provided
        if self.overlay_id is None:
            self.overlay_id = self._generate_id()
        
        # Set creation time if not provided
        if self.creation_time is None:
            import time
            self.creation_time = time.time()
    
    def _generate_id(self) -> str:
        """Generate unique overlay ID."""
        data = {
            'e8_base': self.e8_base.tolist(),
            'activations': self.activations.tolist(),
            'weights': self.weights.tolist() if self.weights is not None else None,
            'phase': self.phase,
            'pose': self.pose.to_dict() if self.pose else None,
            'creation_time': self.creation_time
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
    
    def active_roots(self) -> np.ndarray:
        """Get indices of active roots."""
        return np.where(self.activations == 1)[0]
    
    def num_active(self) -> int:
        """Count active roots."""
        return int(np.sum(self.activations))
    
    def get_weighted_projection(self) -> np.ndarray:
        """
        Get weighted projection of active roots.
        
        If weights are provided, use them. Otherwise, treat all active
        roots equally.
        """
        if self.weights is not None:
            return self.activations * self.weights
        else:
            return self.activations.astype(float)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'overlay_id': self.overlay_id,
            'parent_id': self.parent_id,
            'e8_base': self.e8_base.tolist(),
            'activations': self.activations.tolist(),
            'weights': self.weights.tolist() if self.weights is not None else None,
            'phase': self.phase,
            'pose': self.pose.to_dict() if self.pose else None,
            'creation_time': self.creation_time,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Overlay':
        """Create overlay from dictionary."""
        pose = None
        if data.get('pose'):
            pose = ImmutablePose.from_dict(data['pose'])
        
        weights = None
        if data.get('weights'):
            weights = np.array(data['weights'])
        
        return cls(
            e8_base=np.array(data['e8_base']),
            activations=np.array(data['activations']),
            weights=weights,
            phase=data.get('phase'),
            pose=pose,
            overlay_id=data.get('overlay_id'),
            parent_id=data.get('parent_id'),
            creation_time=data.get('creation_time'),
            metadata=data.get('metadata', {})
        )
    
    def clone(self, **updates) -> 'Overlay':
        """
        Create a modified copy of this overlay.
        
        Args:
            **updates: Fields to update in the clone
        
        Returns:
            New Overlay with updated fields
        """
        data = self.to_dict()
        data.update(updates)
        # Clear overlay_id to generate new one
        data['overlay_id'] = None
        # Set parent to current overlay
        data['parent_id'] = self.overlay_id
        return Overlay.from_dict(data)


class DomainAdapter:
    """
    Base class for domain adapters.
    
    From Axiom A:
    "Domain adapters embed native objects to overlays."
    
    Each domain (text, images, audio, etc.) needs an adapter to convert
    native objects into overlay representations.
    """
    
    def __init__(self, domain_name: str):
        self.domain_name = domain_name
    
    def embed(self, native_object: Any) -> Overlay:
        """
        Embed native object into an overlay.
        
        Args:
            native_object: Domain-specific object
        
        Returns:
            Overlay representation
        """
        raise NotImplementedError("Subclasses must implement embed()")
    
    def extract(self, overlay: Overlay) -> Any:
        """
        Extract native object from overlay.
        
        Args:
            overlay: Overlay representation
        
        Returns:
            Domain-specific object
        """
        raise NotImplementedError("Subclasses must implement extract()")


class NumericAdapter(DomainAdapter):
    """Adapter for numeric data."""
    
    def __init__(self):
        super().__init__("numeric")
    
    def embed(self, value: float) -> Overlay:
        """Embed numeric value into overlay."""
        # Simple embedding: map value to E₈ coordinates
        # Use digital root for stratification
        dr = int(abs(value)) % 9
        
        # Create E₈ base from value
        e8_base = np.zeros(8)
        e8_base[0] = value
        e8_base[dr % 8] = value * 0.1  # Secondary component
        
        # Activate roots based on digital root
        activations = np.zeros(240, dtype=int)
        # Activate every 9th root starting from DR
        activations[dr::9] = 1
        
        # Create pose
        import time
        pose = ImmutablePose(
            position=tuple(e8_base),
            orientation=tuple(np.eye(8)[dr % 8]),
            timestamp=time.time()
        )
        
        return Overlay(
            e8_base=e8_base,
            activations=activations,
            pose=pose,
            metadata={'value': value, 'digital_root': dr}
        )
    
    def extract(self, overlay: Overlay) -> float:
        """Extract numeric value from overlay."""
        return overlay.metadata.get('value', overlay.e8_base[0])


class TextAdapter(DomainAdapter):
    """Adapter for text data."""
    
    def __init__(self):
        super().__init__("text")
    
    def embed(self, text: str) -> Overlay:
        """Embed text into overlay."""
        # Compute hash for text
        text_hash = hashlib.sha256(text.encode()).digest()
        
        # Map hash to E₈ coordinates
        e8_base = np.frombuffer(text_hash[:32], dtype=np.float32)[:8]
        e8_base = e8_base / np.linalg.norm(e8_base)  # Normalize
        e8_base = e8_base.astype(np.float64)  # Convert to float64 for JSON compatibility
        
        # Activate roots based on character frequencies
        activations = np.zeros(240, dtype=int)
        for i, char in enumerate(text[:240]):
            activations[ord(char) % 240] = 1
        
        # Create pose
        import time
        pose = ImmutablePose(
            position=tuple(e8_base),
            orientation=tuple(np.eye(8)[0]),
            timestamp=time.time()
        )
        
        return Overlay(
            e8_base=e8_base,
            activations=activations,
            pose=pose,
            metadata={'text': text, 'length': len(text)}
        )
    
    def extract(self, overlay: Overlay) -> str:
        """Extract text from overlay."""
        return overlay.metadata.get('text', '')


class OverlayStore:
    """
    Storage and retrieval system for overlays.
    
    Maintains overlay history and enables deterministic replay.
    """
    
    def __init__(self):
        self.overlays: Dict[str, Overlay] = {}
        self.history: List[str] = []
    
    def store(self, overlay: Overlay) -> str:
        """
        Store overlay and return its ID.
        
        Args:
            overlay: Overlay to store
        
        Returns:
            Overlay ID
        """
        overlay_id = overlay.overlay_id
        self.overlays[overlay_id] = overlay
        self.history.append(overlay_id)
        return overlay_id
    
    def retrieve(self, overlay_id: str) -> Optional[Overlay]:
        """
        Retrieve overlay by ID.
        
        Args:
            overlay_id: Overlay ID
        
        Returns:
            Overlay if found, None otherwise
        """
        return self.overlays.get(overlay_id)
    
    def get_history(self) -> List[str]:
        """Get overlay history (IDs in order)."""
        return self.history.copy()
    
    def get_lineage(self, overlay_id: str) -> List[str]:
        """
        Get lineage of an overlay (parent chain).
        
        Args:
            overlay_id: Overlay ID
        
        Returns:
            List of overlay IDs from root to current
        """
        lineage = []
        current_id = overlay_id
        
        while current_id:
            lineage.append(current_id)
            overlay = self.retrieve(current_id)
            if not overlay:
                break
            current_id = overlay.parent_id
        
        return list(reversed(lineage))
    
    def save(self, filepath: str):
        """Save overlay store to file."""
        data = {
            'overlays': {
                oid: overlay.to_dict()
                for oid, overlay in self.overlays.items()
            },
            'history': self.history
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'OverlayStore':
        """Load overlay store from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        store = cls()
        store.overlays = {
            oid: Overlay.from_dict(odata)
            for oid, odata in data['overlays'].items()
        }
        store.history = data['history']
        return store


# Example usage and tests
if __name__ == "__main__":
    print("=== Overlay System Test ===\n")
    
    # Test 1: Create basic overlay
    print("Test 1: Basic Overlay Creation")
    e8_base = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0])
    activations = np.zeros(240, dtype=int)
    activations[0:10] = 1  # Activate first 10 roots
    
    overlay1 = Overlay(e8_base=e8_base, activations=activations)
    print(f"Created overlay: {overlay1.overlay_id}")
    print(f"Active roots: {overlay1.num_active()}")
    print()
    
    # Test 2: Overlay with weights and phase
    print("Test 2: Overlay with Weights and Phase")
    weights = np.random.random(240)
    overlay2 = Overlay(
        e8_base=e8_base,
        activations=activations,
        weights=weights,
        phase=0.5
    )
    print(f"Created overlay: {overlay2.overlay_id}")
    print(f"Phase: {overlay2.phase}")
    print()
    
    # Test 3: Immutable pose
    print("Test 3: Immutable Pose")
    import time
    pose = ImmutablePose(
        position=tuple(e8_base),
        orientation=tuple(np.eye(8)[0]),
        timestamp=time.time()
    )
    overlay3 = Overlay(e8_base=e8_base, activations=activations, pose=pose)
    print(f"Created overlay with pose: {overlay3.overlay_id}")
    print(f"Pose hash: {pose.hash()[:16]}")
    print()
    
    # Test 4: Domain adapters
    print("Test 4: Domain Adapters")
    
    # Numeric adapter
    num_adapter = NumericAdapter()
    num_overlay = num_adapter.embed(42.0)
    print(f"Numeric overlay: {num_overlay.overlay_id}")
    print(f"Extracted value: {num_adapter.extract(num_overlay)}")
    
    # Text adapter
    text_adapter = TextAdapter()
    text_overlay = text_adapter.embed("Hello CQE!")
    print(f"Text overlay: {text_overlay.overlay_id}")
    print(f"Extracted text: {text_adapter.extract(text_overlay)}")
    print()
    
    # Test 5: Overlay store
    print("Test 5: Overlay Store")
    store = OverlayStore()
    
    id1 = store.store(overlay1)
    id2 = store.store(overlay2)
    id3 = store.store(overlay3)
    
    print(f"Stored {len(store.overlays)} overlays")
    print(f"History: {store.get_history()}")
    
    # Test lineage
    child = overlay1.clone(phase=0.8)
    store.store(child)
    lineage = store.get_lineage(child.overlay_id)
    print(f"Lineage for {child.overlay_id}: {lineage}")
    print()
    
    print("=== All Tests Passed ===")
