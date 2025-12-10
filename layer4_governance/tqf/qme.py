
from __future__ import annotations
from typing import Any, Tuple
from .core import qme_tuple, HPObject

def equivalent_QME(a: HPObject, b: HPObject, d: int=8) -> bool:
    return qme_tuple(a, d=d) == qme_tuple(b, d=d)
