import hashlib, struct
class MidstatePlanner:
    def __init__(self): self.rebuilds=0; self.attempts=0
    @property
    def reuse_R(self): return (self.attempts/self.rebuilds) if self.rebuilds else 1.0
    def build_header(self, tpl, rails)->bytes:
        version = rails.version & 0xFFFFFFFF; time_ = rails.timestamp & 0xFFFFFFFF
        nBits = tpl.get("bits"); nBits_int = int(nBits,16) if isinstance(nBits,str) else int(nBits or 0x1d00ffff)
        nonce = rails.nonce & 0xFFFFFFFF
        merkle_low = (rails.extranonce ^ (rails.merkle_class * 0x9E3779B1)) & 0xFFFFFFFF
        prevhash = int(tpl.get("previousblockhash","0"*64)[:8] or "0", 16)
        return struct.pack("<LLLLLL", version, prevhash, merkle_low, time_, nBits_int, nonce)
    def double_sha256(self, header:bytes)->bytes: 
        return hashlib.sha256(hashlib.sha256(header).digest()).digest()
    def hash_header(self, tpl, rails, event:str):
        if event in ("xtra_merkle","version_nonce","time"): self.rebuilds += 1
        self.attempts += 1; return self.double_sha256(self.build_header(tpl, rails))
    @staticmethod
    def meets_target(digest:bytes, target_hex:str)->bool: 
        return int.from_bytes(digest,"big") <= int(target_hex,16)
