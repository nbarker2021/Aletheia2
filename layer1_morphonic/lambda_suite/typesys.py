
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass(frozen=True)
class TyVar:
    name: str

@dataclass(frozen=True)
class Arrow:
    src: 'Type'
    dst: 'Type'

@dataclass(frozen=True)
class Bool: pass

@dataclass(frozen=True)
class Nat: pass

@dataclass(frozen=True)
class Prod:
    fst: 'Type'
    snd: 'Type'

@dataclass(frozen=True)
class Grade:
    """A non-negative 'energy' grade for ΔΦ accounting."""
    value: float

Type = object

def pretty(t: Type) -> str:
    if isinstance(t, Arrow): return f"({pretty(t.src)} -> {pretty(t.dst)})"
    if isinstance(t, Prod): return f"({pretty(t.fst)} × {pretty(t.snd)})"
    if isinstance(t, Bool): return "Bool"
    if isinstance(t, Nat): return "Nat"
    if isinstance(t, TyVar): return t.name
    if isinstance(t, Grade): return f"⟦ΔΦ≤{t.value:.3g}⟧"
    return str(t)
