from .util import master_append

class Markets:
    def __init__(self, root):
        self.root = root
    def shock(self, soc, shock, hmac_key):
        rec = {"type":"market_shock","soc":soc,"shock":shock}
        return master_append(self.root, rec, hmac_key)
    def stabilize(self, soc, method, hmac_key):
        rec = {"type":"market_stabilize","soc":soc,"k":method}
        return master_append(self.root, rec, hmac_key)
    def understanding_spend(self, soc, agent, topic, chits, hmac_key):
        rec = {"type":"understanding_spend","soc":soc,"agent":agent,"topic":topic,"chits":chits}
        return master_append(self.root, rec, hmac_key)
