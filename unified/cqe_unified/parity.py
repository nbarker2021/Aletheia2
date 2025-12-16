
from typing import Dict, Any, List

class ParityState:
    """Tracks parity lanes (8-lane octet) and simple invariants."""
    def __init__(self):
        # lanes[i] in {"E","O"} for even/odd; start neutral (None)
        self.lanes: List[str] = [None]*8
        self.violations: List[str] = []

    def set_lane(self, idx: int, parity: str):
        assert parity in ("E","O")
        self.lanes[idx] = parity

    def check_octal_lock(self) -> bool:
        """Octet forcing: require (E,O) alternation pattern or all-defined consistency."""
        # Simple rule: if any two consecutive lanes both defined, they cannot be equal.
        ok = True
        for i in range(7):
            a, b = self.lanes[i], self.lanes[i+1]
            if a is not None and b is not None and a == b:
                self.violations.append(f"adjacent_equal:{i}-{i+1}")
                ok = False
        return ok

    def parity_of(self, x: int) -> str:
        return "E" if (x % 2 == 0) else "O"
