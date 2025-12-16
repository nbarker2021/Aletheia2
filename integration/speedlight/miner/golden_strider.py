from dataclasses import dataclass
@dataclass
class StriderCfg: Kx:int; Km:int; Kv:int
def palindromic_window(n:int):
    arr=list(range(n))+list(range(n-1,-1,-1))
    for i in arr: yield i
def tri_rail_batches(cfg:StriderCfg):
    for _ in palindromic_window(4):
        for _ in range(cfg.Kx): yield ("nonce",)
        yield ("xtra_merkle",)
        for _ in range(cfg.Km-1): yield ("nonce",)
        for _ in range(cfg.Kv): yield ("version_nonce",)
        yield ("time",)
