from .util import master_append

class ToolsRegistry:
    def __init__(self, root): self.root = root
    def register(self, tool, shares, hmac_key):
        rec = {"type":"tool_register","tool":tool,"shares":shares}
        return master_append(self.root, rec, hmac_key)
    def residual(self, tool, tx, amounts, split, hmac_key):
        rec = {"type":"tool_residual","tool":tool,"tx":tx,"amount":amounts,"split":split}
        return master_append(self.root, rec, hmac_key)

class BountyBoard:
    def __init__(self, root): self.root = root
    def open(self, bid, title, scorecard, hmac_key):
        rec = {"type":"bounty_open","id":bid,"title":title,"scorecard":scorecard}
        return master_append(self.root, rec, hmac_key)
    def claim(self, bid, user, evidence, hmac_key):
        rec = {"type":"bounty_claim","id":bid,"by":user,"evidence":evidence}
        return master_append(self.root, rec, hmac_key)
    def payout(self, bid, to_user, amounts, hmac_key):
        rec = {"type":"bounty_payout","id":bid,"to":to_user,"amount":amounts}
        return master_append(self.root, rec, hmac_key)
