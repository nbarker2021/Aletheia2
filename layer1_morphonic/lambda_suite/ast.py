
from dataclasses import dataclass
from typing import Any, List, Optional

@dataclass(frozen=True)
class Var:
    name: str

@dataclass(frozen=True)
class Lam:
    var: str
    body: Any  # Term

@dataclass(frozen=True)
class App:
    fn: Any
    arg: Any

@dataclass(frozen=True)
class Let:
    var: str
    val: Any
    body: Any

@dataclass(frozen=True)
class Const:
    name: str
    value: Any

@dataclass(frozen=True)
class Pair:
    fst: Any
    snd: Any

@dataclass(frozen=True)
class Fst:
    pair: Any

@dataclass(frozen=True)
class Snd:
    pair: Any

@dataclass(frozen=True)
class If:
    cond: Any
    then: Any
    els: Any

@dataclass(frozen=True)
class Mu:
    var: str
    body: Any  # recursive binding μx.body
