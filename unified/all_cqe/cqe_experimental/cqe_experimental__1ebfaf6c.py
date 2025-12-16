"""
High-level CQE client API
"""

from typing import List, Optional, Dict, Any
import numpy as np
from cqe.core.lattice import E8Lattice
from cqe.core.embedding import BabaiEmbedder
from cqe.core.phi import PhiComputer
from cqe.core.canonicalization import Canonicalizer
from cqe.core.overlay import CQEOverlay
from cqe.morsr.protocol import MORSRProtocol
from cqe.adapters.text import TextAdapter
from cqe.operators.base import CQEOperator


class CQEClient:
    """
    High-level client for CQE operations.

    Provides simple interface for:
    - Embedding content
    - Querying similar overlays
    - Applying transformations
    - Computing metrics
    """

    def __init__(self):
        """Initialize CQE client with default configuration"""
        # Core components
        self.lattice = E8Lattice()
        self.embedder = BabaiEmbedder(self.lattice)
        self.phi_computer = PhiComputer()
        self.canonicalizer = Canonicalizer(self.lattice)

        # MORSR protocol
        self.morsr = MORSRProtocol(self.phi_computer, self.canonicalizer)

        # Adapters
        self.text_adapter = TextAdapter()

        # Overlay cache
        self._overlay_cache: Dict[str, CQEOverlay] = {}

    def embed(
        self,
        content: str,
        domain: str = "text",
        optimize: bool = True
    ) -> CQEOverlay:
        """
        Embed content into E8 space.

        Args:
            content: Content to embed
            domain: Domain type (text, code, etc.)
            optimize: Apply MORSR optimization

        Returns:
            CQEOverlay representation
        """
        # Extract features based on domain
        if domain == "text":
            features = self.text_adapter.extract_features(content)
        else:
            raise ValueError(f"Unsupported domain: {domain}")

        # Embed into E8
        overlay = self.embedder.embed(features, domain)

        # Canonicalize
        overlay = self.canonicalizer.canonicalize(overlay)

        # Optimize with MORSR if requested
        if optimize:
            overlay = self.morsr.pulse_sweep(overlay, max_iterations=5)

        # Cache
        self._overlay_cache[overlay.hash_id] = overlay

        return overlay

    def get_phi_metrics(self, overlay: CQEOverlay) -> Dict[str, float]:
        """
        Compute all Φ metrics for overlay.

        Args:
            overlay: Overlay to analyze

        Returns:
            Dictionary of Φ components and total
        """
        components = self.phi_computer.compute_components(overlay)
        total = self.phi_computer.compute_total(components)

        return {
            'phi_geom': components['geom'],
            'phi_parity': components['parity'],
            'phi_sparsity': components['sparsity'],
            'phi_kissing': components['kissing'],
            'phi_total': total
        }

    def apply_operator(
        self,
        operator_name: str,
        overlay: CQEOverlay
    ) -> CQEOverlay:
        """
        Apply named operator to overlay.

        Args:
            operator_name: Name of operator (rotation, midpoint, etc.)
            overlay: Input overlay

        Returns:
            Transformed overlay
        """
        # Import operators dynamically
        from cqe.operators.rotation import RotationOperator
        from cqe.operators.midpoint import MidpointOperator
        from cqe.operators.parity import ParityMirrorOperator

        # Map names to operators
        operator_map = {
            'rotation': RotationOperator(),
            'midpoint': MidpointOperator(),
            'parity': ParityMirrorOperator(),
        }

        if operator_name not in operator_map:
            raise ValueError(f"Unknown operator: {operator_name}")

        operator = operator_map[operator_name]
        result = operator.apply(overlay)

        return self.canonicalizer.canonicalize(result)

    def find_similar(
        self,
        query_overlay: CQEOverlay,
        top_k: int = 10
    ) -> List[tuple]:
        """
        Find similar overlays in cache.

        Args:
            query_overlay: Query overlay
            top_k: Number of results

        Returns:
            List of (overlay, distance) tuples
        """
        results = []
        query_phi = self.phi_computer.compute_total(
            self.phi_computer.compute_components(query_overlay)
        )

        for cached_overlay in self._overlay_cache.values():
            if cached_overlay.hash_id == query_overlay.hash_id:
                continue

            cached_phi = self.phi_computer.compute_total(
                self.phi_computer.compute_components(cached_overlay)
            )

            # Simple Φ distance
            distance = abs(query_phi - cached_phi)
            results.append((cached_overlay, distance))

        # Sort by distance
        results.sort(key=lambda x: x[1])

        return results[:top_k]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self._overlay_cache),
            'overlays': list(self._overlay_cache.keys())
        }
