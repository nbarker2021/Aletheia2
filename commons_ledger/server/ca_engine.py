import os, json, random, time
from .util import master_append, sha256_hex

class TickEngine:
    def __init__(self, root, rng_scope):
        self.root = root
        self.rng_scope = rng_scope
        self.idx = 0

    def tick(self, hmac_key):
        self.idx += 1
        seed_src = f"{self.rng_scope}|{self.idx}|{time.time()}"
        rng_proof = sha256_hex(seed_src)
        rec = {"type":"tick","idx":self.idx,"rng_seed":rng_proof}
        return master_append(self.root, rec, hmac_key)

class Council:
    def __init__(self, root):
        self.root = root
    def draw(self, pool, count, rng_proof, hmac_key):
        rnd = int(rng_proof[:8],16)
        random.seed(rnd)
        picked = random.sample(pool, min(count, len(pool)))
        rec = {"type":"council_draw","picked":picked,"rng_proof":rng_proof}
        return master_append(self.root, rec, hmac_key)
    def vote(self, proposal, picked, hmac_key):
        ayes = max(1, len(picked)//2+1); nays = len(picked)-ayes; abstain = 0
        rec = {"type":"council_vote","proposal":proposal,"ayes":ayes,"nays":nays,"abstain":abstain}
        return master_append(self.root, rec, hmac_key)

class HarmFloor:
    def __init__(self, root):
        self.root = root
    def evaluate(self, net, hmac_key):
        breach = net.get("demand_MW",0)>net.get("supply_MW",0) or net.get("delta_phi_breach",False)
        if breach:
            rec = {"type":"harm_gate","net":net,"decision":"blocked"}
            return master_append(self.root, rec, hmac_key)
        return None

class RoleScheduler:
    def decide(self, metrics):
        score = metrics.get("AEGIS.Impact",0)*metrics.get("BridgeΔ",0)*metrics.get("ResourceΔ",0)
        return score
