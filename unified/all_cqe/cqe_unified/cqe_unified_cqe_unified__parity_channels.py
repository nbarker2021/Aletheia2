
import numpy as np
from typing import Dict, List, Tuple, Optional

class ParityChannels:
    def __init__(self):
        self.num_channels = 8
        self.golay_generator = self._generate_golay_matrix()
        self.hamming_generator = self._generate_hamming_matrix()

    def _generate_golay_matrix(self)->np.ndarray:
        G = np.zeros((12,24), dtype=int)
        G[:12,:12] = np.eye(12, dtype=int)
        for i in range(12):
            for j in range(12,24):
                G[i,j] = (i+j)%2
        return G

    def _generate_hamming_matrix(self)->np.ndarray:
        return np.array([
            [1,0,0,0,1,1,0],
            [0,1,0,0,1,0,1],
            [0,0,1,0,0,1,1],
            [0,0,0,1,1,1,1]
        ], dtype=int)

    def extract_channels(self, vector: np.ndarray)->Dict[str, float]:
        if len(vector)!=8: raise ValueError("Vector must be 8-dimensional")
        channels = {}
        binary_vec = (vector>0.5).astype(int)
        for i in range(self.num_channels):
            mask = np.array([(i>>j)&1 for j in range(8)], dtype=int)
            parity = int(np.sum(binary_vec*mask)%2)
            refinement = float(np.mean(vector*mask)) if np.sum(mask)>0 else 0.0
            channels[f"channel_{i+1}"] = 0.8*float(parity) + 0.2*refinement
        return channels

    def calculate_parity_penalty(self, vector: np.ndarray, reference_channels: Dict[str,float])->float:
        cur = self.extract_channels(vector)
        return float(sum(abs(cur.get(k,0)-v)**2 for k,v in reference_channels.items()))

    def enforce_parity(self, vector: np.ndarray, target_channels: Dict[str,float])->np.ndarray:
        corrected = vector.copy()
        for it in range(3):
            cur = self.extract_channels(corrected)
            total_error = sum(abs(cur.get(k,0)-v) for k,v in target_channels.items())
            if total_error < 0.1: break
            for i,(k,v) in enumerate(target_channels.items()):
                err = v - cur.get(k,0)
                corr = 0.1*err/(it+1)
                for j in range(8):
                    weight = ((i+j)%8)/8.0
                    corrected[j] += corr*weight
        return corrected
