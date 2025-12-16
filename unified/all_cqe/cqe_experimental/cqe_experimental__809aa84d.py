"""
Serialization utilities for CQE objects

Handles numpy arrays, overlays, and other CQE data structures.
"""

import json
import numpy as np
from typing import Any, Dict
from cqe.core.overlay import CQEOverlay


class CQEJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for CQE objects.

    Handles:
    - numpy arrays and scalars
    - CQE overlays
    - Complex nested structures
    """

    def default(self, obj):
        """Encode CQE objects to JSON-serializable format"""

        # Handle numpy types
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # Scalar numpy types
            return obj.item()

        # Handle CQE overlay
        elif isinstance(obj, CQEOverlay):
            return overlay_to_dict(obj)

        # Handle complex numbers
        elif isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag}

        # Default behavior
        return super().default(obj)


def overlay_to_dict(overlay: CQEOverlay) -> Dict[str, Any]:
    """
    Convert overlay to JSON-serializable dictionary.

    Args:
        overlay: CQEOverlay instance

    Returns:
        Dictionary with all overlay data
    """
    return {
        'present': overlay.present.tolist(),
        'w': overlay.w.tolist(),
        'phi': overlay.phi.tolist(),
        'pose': overlay.pose,
        'hash_id': overlay.hash_id,
        'provenance': overlay.provenance,
        'metadata': {
            'active_slots': len(overlay.active_slots),
            'cartan_active': overlay.cartan_active,
            'root_active': overlay.root_active,
            'sparsity': overlay.sparsity
        }
    }


def overlay_from_dict(data: Dict[str, Any]) -> CQEOverlay:
    """
    Reconstruct overlay from dictionary.

    Args:
        data: Dictionary from overlay_to_dict()

    Returns:
        Reconstructed CQEOverlay
    """
    return CQEOverlay(
        present=np.array(data['present'], dtype=bool),
        w=np.array(data['w'], dtype=np.float64),
        phi=np.array(data['phi'], dtype=np.float64),
        pose=data['pose'],
        hash_id=data.get('hash_id'),
        provenance=data.get('provenance', [])
    )


def serialize_overlay(overlay: CQEOverlay) -> str:
    """
    Serialize overlay to JSON string.

    Args:
        overlay: CQEOverlay to serialize

    Returns:
        JSON string
    """
    return json.dumps(overlay_to_dict(overlay), cls=CQEJSONEncoder)


def deserialize_overlay(json_str: str) -> CQEOverlay:
    """
    Deserialize overlay from JSON string.

    Args:
        json_str: JSON string from serialize_overlay()

    Returns:
        Reconstructed CQEOverlay
    """
    data = json.loads(json_str)
    return overlay_from_dict(data)
