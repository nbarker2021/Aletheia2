"""
Proper Phi Metric Implementation with 4 Components
Based on CQE principles: Geometric, Parity, Sparsity, Kissing
"""

import numpy as np
from typing import Dict, Any

class ProperPhiMetric:
    """
    Complete phi metric with 4 weighted components:
    - Geometric (40%): Lattice alignment quality
    - Parity (30%): Even/odd structure preservation  
    - Sparsity (20%): Information density
    - Kissing (10%): Neighbor relationships
    """
    
    def __init__(self):
        self.weights = {
            'geometric': 0.40,
            'parity': 0.30,
            'sparsity': 0.20,
            'kissing': 0.10
        }
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
    def calculate(self, vector: np.ndarray, context: Dict[str, Any] = None) -> float:
        """
        Calculate complete phi metric for a vector.
        
        Args:
            vector: Input vector (any dimension)
            context: Optional context with lattice info, previous states, etc.
            
        Returns:
            Composite phi score [0, 1] where higher is better
        """
        if context is None:
            context = {}
            
        # Calculate each component
        geometric = self._geometric_component(vector, context)
        parity = self._parity_component(vector)
        sparsity = self._sparsity_component(vector)
        kissing = self._kissing_component(vector, context)
        
        # Weighted sum
        phi_score = (
            self.weights['geometric'] * geometric +
            self.weights['parity'] * parity +
            self.weights['sparsity'] * sparsity +
            self.weights['kissing'] * kissing
        )
        
        return float(phi_score)
    
    def _geometric_component(self, vector: np.ndarray, context: Dict) -> float:
        """
        Measure how well vector aligns with lattice structure.
        Uses norm stability and golden ratio relationships.
        """
        norm = np.linalg.norm(vector)
        
        if norm < 1e-10:
            return 0.0
            
        # Normalized vector
        unit = vector / norm
        
        # Check if norm is close to golden ratio powers
        phi_powers = [self.phi ** i for i in range(-3, 4)]
        min_phi_dist = min(abs(norm - p) for p in phi_powers)
        phi_alignment = np.exp(-min_phi_dist)
        
        # Check coordinate alignment (prefer aligned coordinates)
        coord_variance = np.var(np.abs(unit))
        coord_score = 1.0 / (1.0 + coord_variance)
        
        # Combine
        geometric_score = 0.6 * phi_alignment + 0.4 * coord_score
        
        return float(geometric_score)
    
    def _parity_component(self, vector: np.ndarray) -> float:
        """
        Measure parity structure preservation.
        Even/odd balance indicates geometric integrity.
        """
        # Round to integers for parity check
        int_coords = np.round(vector).astype(int)
        
        # Count even and odd coordinates
        even_count = np.sum(int_coords % 2 == 0)
        odd_count = len(int_coords) - even_count
        
        # Perfect balance = 0.5 each
        even_ratio = even_count / len(int_coords)
        
        # Score based on how close to balanced (0.5)
        parity_score = 1.0 - 2.0 * abs(even_ratio - 0.5)
        
        return float(parity_score)
    
    def _sparsity_component(self, vector: np.ndarray) -> float:
        """
        Measure information density via sparsity.
        Sparser = more structured = higher score.
        """
        # Threshold for considering a coordinate "zero"
        threshold = 0.1 * np.max(np.abs(vector)) if np.max(np.abs(vector)) > 0 else 0.1
        
        # Count near-zero coordinates
        zero_count = np.sum(np.abs(vector) < threshold)
        sparsity_ratio = zero_count / len(vector)
        
        # Moderate sparsity is best (not too sparse, not too dense)
        # Optimal around 0.3-0.5
        optimal_sparsity = 0.4
        sparsity_score = 1.0 - abs(sparsity_ratio - optimal_sparsity) / optimal_sparsity
        sparsity_score = max(0.0, min(1.0, sparsity_score))
        
        return float(sparsity_score)
    
    def _kissing_component(self, vector: np.ndarray, context: Dict) -> float:
        """
        Measure neighbor relationships.
        Uses previous vectors if available in context.
        """
        if 'previous_vectors' not in context or len(context['previous_vectors']) == 0:
            # No history, use self-similarity
            return 0.5
        
        prev_vectors = context['previous_vectors']
        
        # Calculate distances to recent neighbors
        distances = []
        for prev in prev_vectors[-5:]:  # Last 5 vectors
            dist = np.linalg.norm(vector - prev)
            distances.append(dist)
        
        if len(distances) == 0:
            return 0.5
        
        # Prefer moderate distances (not too close, not too far)
        avg_dist = np.mean(distances)
        std_dist = np.std(distances) if len(distances) > 1 else 0.0
        
        # Score based on consistency (low std) and moderate distance
        consistency_score = np.exp(-std_dist)
        
        # Optimal distance around 1.0 (normalized)
        optimal_dist = 1.0
        distance_score = np.exp(-abs(avg_dist - optimal_dist))
        
        kissing_score = 0.6 * consistency_score + 0.4 * distance_score
        
        return float(kissing_score)
    
    def calculate_temporal_coherence(self, vectors: list) -> float:
        """
        Calculate coherence across a sequence of vectors.
        Measures how smoothly the phi metric evolves.
        
        Args:
            vectors: List of vectors in temporal order
            
        Returns:
            Temporal coherence score [0, 1]
        """
        if len(vectors) < 2:
            return 1.0
        
        # Calculate phi scores for each vector
        phi_scores = []
        for i, vec in enumerate(vectors):
            context = {'previous_vectors': vectors[:i]}
            score = self.calculate(vec, context)
            phi_scores.append(score)
        
        # Measure smoothness (low variance = high coherence)
        phi_variance = np.var(phi_scores)
        temporal_coherence = np.exp(-phi_variance)
        
        return float(temporal_coherence)
    
    def detect_anomaly(self, vector: np.ndarray, context: Dict, 
                      threshold: float = 0.3) -> bool:
        """
        Detect if vector is anomalous based on phi metric.
        
        Args:
            vector: Vector to check
            context: Context with previous vectors
            threshold: Phi score below this is anomalous
            
        Returns:
            True if anomalous, False otherwise
        """
        phi_score = self.calculate(vector, context)
        
        # Also check temporal coherence if we have history
        if 'previous_vectors' in context and len(context['previous_vectors']) > 0:
            recent_vectors = context['previous_vectors'][-10:] + [vector]
            temporal_coh = self.calculate_temporal_coherence(recent_vectors)
            
            # Anomaly if either phi is low OR temporal coherence drops
            is_anomalous = (phi_score < threshold) or (temporal_coh < 0.5)
        else:
            is_anomalous = phi_score < threshold
        
        return is_anomalous
