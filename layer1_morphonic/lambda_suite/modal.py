
from dataclasses import dataclass
from typing import Any
from . import ast as A

@dataclass(frozen=True)
class Box:
    term: Any  # □t

@dataclass(frozen=True)
class Diamond:
    term: Any  # ◇t

# Semantics are left as rules to the evaluator harness; these act as tags for now.
