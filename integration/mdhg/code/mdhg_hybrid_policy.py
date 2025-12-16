
# Decision helper for when to use MDHG vs native hashing
# Usage: from mdhg_hybrid_policy import choose_hash
from dataclasses import dataclass

@dataclass
class HashDecision:
    use_mdhg: bool
    reason: str

def choose_hash(persist: bool, needs_semantic_routing: bool, needs_cross_run_invariance: bool, payload_size: int) -> HashDecision:
    # Very simple heuristic; tune later.
    if needs_semantic_routing or needs_cross_run_invariance:
        return HashDecision(True, "Semantic identity or invariance required.")
    if persist and payload_size > 0:
        return HashDecision(True, "Persisted identity benefits from MDHG axes encoding.")
    return HashDecision(False, "Local, ephemeral hashing prefers native speed.")
