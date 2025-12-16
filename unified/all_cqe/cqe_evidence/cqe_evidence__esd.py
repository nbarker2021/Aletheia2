"""
Evidence Surface Decoder (ESD)

Interprets a user-observed remainder pattern of the form:
  decimal . zeros...  digits...  single 0   oddDigit repeated...
  e.g.  .000  3719  0  7 7 7 7 7...
The interpretation (per the user's hypothesis):
  - the decimal point: interaction/superposition boundary
  - zeros count: number of shells outward to the connection layer
  - digit run: operation class (kind/complexity of ops to equate)
  - single zero: number of additional tori layers to move
  - repeating odd digit: modulus to apply and rotation count

This module parses such strings into a structured dict. The semantics
are preserved in the returned fields for downstream use.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import Optional, Dict

# Simple pattern: integer part optional; decimal point; >=1 zeros; >=1 digits; a single 0; trailing odd digits
PATTERN = re.compile(r"""
    ^
    (?P<int>\d*)?
    \.                      # decimal point
    (?P<zeros>0+)
    (?P<ops>\d+)
    0
    (?P<odd>[13579]+)$
""", re.X)

@dataclass
class ESDResult:
    raw: str
    shells: int                 # count of leading zeros after decimal
    ops_digits: str             # the run right after zeros
    extra_layers: int           # from the single '0' marker (currently 1 if present)
    odd_digit: str              # last odd digit seen
    odd_run_len: int            # count of trailing odd digits
    note: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

def parse_decimal_remainder(s: str) -> Optional[ESDResult]:
    s = s.strip().replace(" ", "")
    if s.startswith("."):
        s = "0" + s
    m = PATTERN.match(s)
    if not m:
        return None
    zeros = m.group("zeros")
    ops = m.group("ops")
    odd = m.group("odd")
    odd_digit = odd[-1]
    return ESDResult(
        raw=s,
        shells=len(zeros),
        ops_digits=ops,
        extra_layers=1,
        odd_digit=odd_digit,
        odd_run_len=len(odd),
        note="Parsed per ESD hypothesis; semantics are user-defined",
    )
