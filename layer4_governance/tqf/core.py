
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import hashlib, random

BASES = [2,4,8,16,32,64]

@dataclass
class Invariants:
    W4: bool = False
    Pal8: bool = False
    W80: bool = False

    def as_tuple(self) -> Tuple[int,int,int]:
        return (1 if self.W4 else 0, 1 if self.Pal8 else 0, 1 if self.W80 else 0)

@dataclass
class Receipts:
    QF: int = 0
    P8: int = 0
    dS: int = 0  # entropy valuation

    def copy(self) -> 'Receipts':
        return Receipts(self.QF, self.P8, self.dS)

    def as_tuple(self) -> Tuple[int,int,int]:
        return (self.QF, self.P8, self.dS)

@dataclass
class PartitionDesc:
    n: int
    B: int
    parts: List[int]

    def as_tuple(self) -> Tuple[int, int, Tuple[int, ...]]:
        return (self.n, self.B, tuple(self.parts))

@dataclass
class RAGEvent:
    op: str
    data: Dict[str, Any]

@dataclass
class HPObject:
    name: str
    context: str
    base: int
    moduli: List[int]
    partition: PartitionDesc
    invariants: Invariants
    receipts: Receipts
    regime: str = "+"  # '+' for Taxicab, '±' for Cabtaxi
    rag: List[RAGEvent] = field(default_factory=list)

    def clone(self) -> 'HPObject':
        return HPObject(
            name=self.name,
            context=self.context,
            base=self.base,
            moduli=list(self.moduli),
            partition=PartitionDesc(self.partition.n, self.partition.B, list(self.partition.parts)),
            invariants=Invariants(self.invariants.W4, self.invariants.Pal8, self.invariants.W80),
            receipts=self.receipts.copy(),
            regime=self.regime,
            rag=list(self.rag),
        )

def stable_hash(*xs: Any) -> int:
    m = hashlib.sha256()
    for x in xs:
        m.update(repr(x).encode("utf-8"))
    return int.from_bytes(m.digest()[:8], "big")

def partition_normalize(n: int, B: int) -> PartitionDesc:
    # Greedy normalization: split n into parts <= B/2
    assert B in BASES, "Unsupported base"
    maxp = B//2
    parts = []
    rem = n
    while rem > 0:
        p = min(maxp, rem)
        parts.append(p)
        rem -= p
    return PartitionDesc(n=n, B=B, parts=parts)

def modular_legalize(obj: HPObject) -> HPObject:
    # Heuristic invariants; minimal receipts to legalize
    new = obj.clone()
    B = new.base
    mods = set(new.moduli)
    inv = Invariants()
    inv.W4 = (B % 4 == 0) or (4 in mods)
    inv.Pal8 = (B >= 8) or (8 in mods)
    inv.W80 = (80 in mods) or (B >= 16 and 5 in mods)  # toy proxy
    if not inv.W4:
        new.receipts.QF += 1; inv.W4 = True
    if not inv.Pal8:
        new.receipts.P8 += 1; inv.Pal8 = True
    if not inv.W80:
        new.receipts.dS += 1; inv.W80 = True
    new.invariants = inv
    new.rag.append(RAGEvent("N_B,M", {"base": B, "moduli": list(mods), "receipts": new.receipts.as_tuple(), "invariants": inv.as_tuple()}))
    return new

def lift(obj: HPObject) -> HPObject:
    # Base lift B -> 2*B; update receipts deterministically
    assert obj.base in BASES and obj.base != 64, "Cannot lift beyond B64"
    new = obj.clone()
    oldB = obj.base
    new.base = oldB * 2
    new.partition = partition_normalize(obj.partition.n, new.base)
    new.receipts.QF += 1
    if new.base >= 8:
        new.receipts.P8 += 1
    new.rag.append(RAGEvent("LIFT", {"from": oldB, "to": new.base, "receipts": new.receipts.as_tuple()}))
    return new

def downlift(obj: HPObject) -> HPObject:
    # Inverse of lift: halve base and reverse deterministic receipt updates
    assert obj.base in BASES and obj.base != 2, "Cannot downlift below B2"
    new = obj.clone()
    oldB = obj.base
    new.base = oldB // 2
    new.partition = partition_normalize(obj.partition.n, new.base)
    if oldB >= 8:
        new.receipts.P8 -= 1
    new.receipts.QF -= 1
    new.rag.append(RAGEvent("DOWNLIFT", {"from": oldB, "to": new.base, "receipts": new.receipts.as_tuple()}))
    return new

def taxicab_witness(x:int,y:int,z:int,w:int, signed:bool=False) -> bool:
    lhs = x**3 + y**3
    rhs = z**3 + w**3
    if not signed and (x<0 or y<0 or z<0 or w<0):
        return False
    return lhs == rhs

def aperture(obj: HPObject, sigma: str, witness: Tuple[int,int,int,int]) -> HPObject:
    # Switch regime using a valid taxicab/cabtaxi witness; receipts tick for signed regime
    assert sigma in ["+","±"], "sigma must be '+' (Taxicab) or '±' (Cabtaxi)"
    x,y,z,w = witness
    ok = taxicab_witness(x,y,z,w, signed=(sigma=="±"))
    if not ok:
        raise ValueError("Invalid taxicab/cabtaxi witness for selected regime")
    new = obj.clone()
    prev = new.regime
    new.regime = sigma
    new.receipts.dS += 1 if sigma=="±" and prev=="+" else 0
    new.receipts.dS -= 1 if sigma=="+" and prev=="±" else 0
    new.rag.append(RAGEvent("APERTURE", {"from": prev, "to": sigma, "witness": witness, "receipts": new.receipts.as_tuple()}))
    return new

def embed(obj: HPObject, d: int=8) -> Tuple[float, ...]:
    # Deterministic placeholder embedding
    rnd = random.Random(stable_hash(obj.partition.as_tuple(), obj.base, obj.regime))
    vec = [rnd.uniform(-1.0, 1.0) for _ in range(d)]
    return tuple(vec)

def canonical_form(vec: Tuple[float, ...]) -> Tuple[int, ...]:
    # Discretize and sort as a placeholder CNF
    q = tuple(int(round(x*1000)) for x in vec)
    return tuple(sorted(q))

def qme_tuple(obj: HPObject, d: int=8) -> Tuple[Any, ...]:
    vec = embed(obj, d=d)
    cnf = canonical_form(vec)
    return (
        cnf,
        obj.partition.as_tuple(),
        obj.receipts.as_tuple(),
        obj.invariants.as_tuple(),
        obj.regime,
        obj.base,
        tuple(sorted(obj.moduli)),
    )

def make_hp(name: str, context: str, n: int, B: int, moduli: List[int], regime: str="+") -> HPObject:
    part = partition_normalize(n, B)
    inv = Invariants(False, False, False)
    rec = Receipts(0,0,0)
    hp = HPObject(name=name, context=context, base=B, moduli=list(moduli), partition=part, invariants=inv, receipts=rec, regime=regime)
    hp = modular_legalize(hp)
    return hp
