
from typing import Dict, Any

class Governance:
    def __init__(self):
        self.gates = {
            "W4": 0.7,
            "W80": 0.7,
            "Wexp": 0.6,
            "LAWFUL": 0.8
        }

    def evaluate(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        results = {}
        for gate, thr in self.gates.items():
            score = float(metrics.get(gate, 0.0))
            results[gate] = {"score": score, "threshold": thr, "pass": score >= thr}
        return results
