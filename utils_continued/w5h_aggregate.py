def w5h_aggregate(beacon: dict) -> Dict[str, float]:
    """Return per-dimension and final aggregate score according to policy."""
    w5h = beacon["w5h"]
    policy = beacon.get("policy", {})
    method = policy.get("aggregation", "mean")
    weights = policy.get("weights", {})
    priority = policy.get("priority_contexts", [])

    def dim_score(dim: str) -> float:
        ctxs = w5h[dim]["contexts"]
        vals = [float(c["score"]) for c in ctxs]
        names = [c["name"] for c in ctxs]
        return _agg(vals, method, weights, names)

    dims = ["who","what","where","when","why","how"]
    per_dim = {d: dim_score(d) for d in dims}

    # Final score: aggregate chosen priority contexts when present, else aggregate per-dim
    if priority:
        # Map priority names to find them inside contexts across dims
        collected = []
        for d in dims:
            for c in w5h[d]["contexts"]:
                if c["name"] in priority:
                    collected.append((c["name"], float(c["score"])))
        if collected:
            names = [n for n,_ in collected]
            vals = [v for _,v in collected]
            final = _agg(vals, method, weights, names)
        else:
            final = _agg(list(per_dim.values()), method)
    else:
        final = _agg(list(per_dim.values()), method)

    return {"final": final, **per_dim}
from .unified_system import EnhancedCQESystem, create_enhanced_cqe_system
__all__ = ["EnhancedCQESystem", "create_enhanced_cqe_system"]
"""
Enhanced CQE System - Unified Integration of Legacy Variations

Integrates TQF governance, UVIBS extensions, multi-dimensional logic,
and scene-based debugging into a comprehensive CQE framework.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path

# Import base CQE components
from ..core import E8Lattice, MORSRExplorer, CQEObjectiveFunction
from ..core.parity_channels import ParityChannels
from ..domains import DomainAdapter
from ..validation import ValidationFramework
