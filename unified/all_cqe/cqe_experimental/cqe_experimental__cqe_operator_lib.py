
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import hashlib, json

@dataclass
class Octet:
    views: Dict[str, str]  # V1..V8 -> token

@dataclass
class MirrorEvidence:
    pairs: List[Tuple[str,str]]  # [(V1,V8), ...]
    rationale: Dict[str, str]    # "V1-V8" -> text

@dataclass
class StrictResult:
    level: str                 # "LOOSE", "BALANCED", "HARD"
    reasons: List[str] = field(default_factory=list)

@dataclass
class DeltaResult:
    tags: List[str]
    statement: str
    admissible: bool
    reasons: List[str] = field(default_factory=list)

def four_bit_commit(seed_text: str) -> str:
    h = hashlib.sha256(seed_text.encode()).hexdigest()
    return format(int(h[0],16), "04b")

def mirror_pairs(octet: Octet):
    return [("V1","V8"),("V2","V7"),("V3","V6"),("V4","V5")]

def require_witness(mirr: MirrorEvidence) -> bool:
    # All canonical pairs must have rationale
    for a,b in mirr.pairs:
        key = f"{a}-{b}"
        if key not in mirr.rationale or not mirr.rationale[key].strip():
            return False
    return True

def strict_ratchet(tokens: List[str]) -> StrictResult:
    reasons = []
    lvl = 0
    tjoin = " | ".join(t.lower() for t in tokens)
    if "strict bounds" in tjoin:
        lvl += 1; reasons.append("Strict Bounds present")
    if "metric" in tjoin:
        lvl += 1; reasons.append("Metric present")
    if "4-bit" in tjoin or "commit" in tjoin:
        lvl += 1; reasons.append("Receipts present")
    if "no glue" in tjoin:
        lvl += 1; reasons.append("No-glue clause present")
    level = ["LOOSE","BALANCED","HARD"][min(lvl//2, 2)]
    return StrictResult(level=level, reasons=reasons)

def delta_lift(octet: Octet, strict: StrictResult, tokens: List[str], mirror_ok: bool) -> DeltaResult:
    tags = []
    tjoin = " ".join(tokens).lower()
    if "parity" in tjoin: tags.append("parity")
    if "mirror" in tjoin: tags.append("reflection")
    if "witness" in tjoin or mirror_ok: tags.append("witness")
    if "loom" in tjoin or "overlay" in tjoin: tags.append("synthesis")
    if "station" in tjoin: tags.append("routing")
    if "rate" in tjoin or "budget" in tjoin: tags.append("rate/budget")
    # Admissibility rules
    reasons = []
    admissible = True
    if strict.level == "HARD":
        # Require Mirror witness and Strict Bounds token
        if "strict bounds" not in tjoin: admissible=False; reasons.append("Missing Strict Bounds under HARD")
        if not mirror_ok: admissible=False; reasons.append("Mirror witness required under HARD")
    if strict.level == "BALANCED":
        if not mirror_ok: reasons.append("Mirror witness recommended")
    stmt = "Promote Parity Twin to bounded, witnessed twin; measurements recorded; Δ maintains invariants."
    return DeltaResult(tags=sorted(set(tags))[:3], statement=stmt, admissible=admissible, reasons=reasons)
