
from typing import List, Dict

def summarize_lane(meta: Dict) -> List[float]:
    ch = float(meta.get("channel", 3))
    dphi = float(meta.get("delta_phi", 0.0))
    scope = 1.0 if meta.get("scope") else 0.0
    return [ch/9.0, dphi, scope]
