import time, random
class SimNode:
    def __init__(self, seed: int=42):
        random.seed(seed); self.height=0; self.bits="1d00ffff"; self.prevhash="00"*32; self.longpollid="sim"
    def getblocktemplate(self):
        self.height += 1
        return {"version":0x20000000,"previousblockhash":self.prevhash,"curtime":int(time.time()),
                "mediantime":int(time.time())-60,"bits":self.bits,"height":self.height,"transactions":[],
                "longpollid":f"{self.longpollid}-{self.height}"}
    def submitblock(self, hexblock: str): self.prevhash = ("%064x" % random.getrandbits(256))
