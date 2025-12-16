
from typing import Dict, Any
import math

def extract_semantics(e8_distance: float, angle_hint: float) -> Dict[str, Any]:
    # Placeholder thresholds; real system should calibrate
    relation = "NEAR" if e8_distance < 0.25 else "FAR"
    structure = "ALIGNED" if abs(angle_hint) < 0.35 else "CROSSED"
    confidence = max(0.0, 1.0 - e8_distance) * (1.0 - min(1.0, abs(angle_hint)))
    return {"relation": relation, "structure": structure, "confidence": round(confidence, 3)}
