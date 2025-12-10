from .util import master_append

class OMPS:
    def __init__(self): pass
    def check(self, witness):
        ok = all(k in witness for k in ("candidate@pos","mirror@-pos","alias@pos","alias@-pos"))
        return {"ok":ok,"witness":witness}

class JokerBank:
    def __init__(self):
        self.used = set()
    def consume(self, token_id):
        if token_id in self.used: return False
        self.used.add(token_id); return True

class EnergyGate:
    def __init__(self):
        self.last_E = None
    def update(self, E):
        if self.last_E is None:
            self.last_E = E
            return True
        ok = (E <= self.last_E + 1e-9)
        self.last_E = E
        return ok

def ring_checkpoint(root, idx, hmac_key):
    rec = {"type":"ring_checkpoint","idx":idx}
    return master_append(root, rec, hmac_key)
