"""
Extended binary Golay code G_24 - stubs.
WARNING: This file contains interfaces only; the actual generator/
parity matrices and syndrome decoding are NOT implemented yet.
"""
from __future__ import annotations
from typing import Iterable, Tuple

Bit = int
Word = Tuple[Bit, ...]

class Golay24:
    def __init__(self):
        self.G = None  # 12x24
        self.H = None  # 12x24

    def encode(self, message_12b: Iterable[Bit]) -> Word:
        raise NotImplementedError("Golay24.encode is not implemented")

    def syndrome(self, word_24b: Iterable[Bit]) -> Tuple[Bit, ...]:
        raise NotImplementedError("Golay24.syndrome is not implemented")

    def decode(self, received_24b: Iterable[Bit]) -> Word:
        w = tuple(int(b) & 1 for b in received_24b)
        if len(w) != 24:
            raise ValueError("Golay24.decode expects 24 bits")
        return w
