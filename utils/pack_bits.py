def pack_bits(s:str):
    s = s.strip().lower()
    if not re.fullmatch(r"[ab]{4}", s):
        raise ValueError(f"invalid shape pack: {s}")
    return [1 if c=="a" else 0 for c in s]

# Map 4-bit shape pack to primitive op mnemonic
SHAPE_OP = {
    "bbbb": "NOP",
    "bbba": "DLIFT",
    "bbab": "MIRROR",
    "bbaa": "RATCHET",
    "babb": "SNAP",
    "baba": "ANNIHILATE",
    "baab": "POSE",
    "baaa": "TICKET",
    "abbb": "BIND",
    "abba": "ROLE",
    "abab": "EMIT",
    "abaa": "CALL",
    "aabb": "MAP",
    "aaba": "FORK",
    "aaab": "JOIN",
    "aaaa": "ASSERT"
}

# ================= Geometry helpers (E8 cap + pose) =================