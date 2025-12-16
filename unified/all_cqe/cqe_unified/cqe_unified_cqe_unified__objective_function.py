
import numpy as np
from typing import Dict, Optional, Tuple
from .e8_lattice import E8Lattice
from .parity_channels import ParityChannels

class CQEObjectiveFunction:
    def __init__(self, e8_lattice: E8Lattice, parity_channels: ParityChannels):
        self.e8_lattice = e8_lattice
        self.parity_channels = parity_channels
        self.weights = {"lattice_quality":0.3,"parity_consistency":0.25,"chamber_stability":0.2,"geometric_separation":0.15,"domain_coherence":0.1}

    def evaluate(self, vector: np.ndarray, reference_channels: Dict[str,float], domain_context: Optional[Dict]=None)->Dict[str,float]:
        if len(vector)!=8: raise ValueError("Vector must be 8-dimensional")
        lattice_score = self._evaluate_lattice_quality(vector)
        parity_score = self._evaluate_parity_consistency(vector, reference_channels)
        chamber_score = self._evaluate_chamber_stability(vector)
        separation_score = self._evaluate_geometric_separation(vector, domain_context)
        coherence_score = self._evaluate_domain_coherence(vector, domain_context)
        phi_total = (self.weights["lattice_quality"]*lattice_score +
                     self.weights["parity_consistency"]*parity_score +
                     self.weights["chamber_stability"]*chamber_score +
                     self.weights["geometric_separation"]*separation_score +
                     self.weights["domain_coherence"]*coherence_score)
        return {"phi_total":float(phi_total),"lattice_quality":float(lattice_score),"parity_consistency":float(parity_score),
                "chamber_stability":float(chamber_score),"geometric_separation":float(separation_score),"domain_coherence":float(coherence_score)}

    def _evaluate_lattice_quality(self, vector):
        q = self.e8_lattice.root_embedding_quality(vector)
        root_score = max(0.0, 1.0 - q["nearest_root_distance"]/2.0)
        depth_score = min(1.0, q["chamber_depth"]/0.5)
        symmetry_score = max(0.0, 1.0 - q["symmetry_score"])
        return 0.5*root_score + 0.3*depth_score + 0.2*symmetry_score

    def _evaluate_parity_consistency(self, vector, ref):
        penalty = self.parity_channels.calculate_parity_penalty(vector, ref)
        return max(0.0, 1.0 - penalty/2.0)

    def _evaluate_chamber_stability(self, vector):
        sig, inner = self.e8_lattice.determine_chamber(vector)
        min_boundary = float(np.min(np.abs(inner)))
        stability = min(1.0, min_boundary/0.3)
        bonus = 0.1 if sig=="11111111" else 0.0
        return stability + bonus

    def _evaluate_geometric_separation(self, vector, ctx):
        if not ctx or "complexity_class" not in ctx: return 0.5
        cc = ctx["complexity_class"]
        if cc=="P":
            target = np.array([0.3,0.1,0.8,0.4,0.5,0.3,0.4,0.2])
        elif cc=="NP":
            target = np.array([0.6,0.9,0.5,0.8,0.7,0.6,0.8,0.5])
        else:
            return 0.5
        dist = float(np.linalg.norm(vector - target))
        return max(0.0, 1.0 - dist/2.0)

    def _evaluate_domain_coherence(self, vector, ctx):
        if not ctx: return 0.5
        dt = ctx.get("domain_type","unknown")
        if dt=="optimization":
            return max(0.0, min(1.0, 1.0 - float(np.std(vector))))
        elif dt=="creative":
            return min(1.0, float(np.std(vector))*2.0)
        elif dt=="computational":
            balance = abs(float(np.mean(vector)) - 0.5); return max(0.0, 1.0 - balance*2.0)
        return 0.5
