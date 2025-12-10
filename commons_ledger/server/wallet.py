import os, json
from .util import master_append

class Wallet:
    def __init__(self, root, coins, payout_wallet):
        self.root = root
        self.path = os.path.join(root,"data","wallet.json")
        self.book = os.path.join(root,"data","ledgers","WalletBook.jsonl")
        self.payout_wallet = payout_wallet
        self.coins = coins
        if not os.path.exists(self.path):
            with open(self.path,"w") as f: json.dump({"balances":{c:0.0 for c in coins}}, f)

    def load(self):
        with open(self.path) as f: return json.load(f)

    def save(self, obj):
        with open(self.path,"w") as f: json.dump(obj, f, indent=2)

    def credit(self, amounts, hmac_key):
        obj = self.load()
        for k,v in amounts.items():
            obj["balances"][k] = obj["balances"].get(k,0.0)+float(v)
        self.save(obj)
        rec = {"type":"wallet_credit","amounts":amounts}
        return master_append(self.root, rec, hmac_key)

class Vesting:
    def __init__(self, root):
        self.root = root
        self.book = os.path.join(root,"data","ledgers","VestBook.jsonl")

    def create(self, vid, schedule, hmac_key):
        rec = {"type":"vesting_create","id":vid,"schedule":schedule}
        return master_append(self.root, rec, hmac_key)

    def release(self, vid, amounts, hmac_key):
        rec = {"type":"vesting_release","id":vid,"released":amounts}
        return master_append(self.root, rec, hmac_key)

class Treasury:
    def __init__(self, root):
        self.root = root
        self.book = os.path.join(root,"data","ledgers","TreasuryBook.jsonl")

    def payout_autodeposit(self, profit_amounts, hmac_key):
        rec = {"type":"payout_autodeposit","profit":profit_amounts,"to_payout":{k:v*0.5 for k,v in profit_amounts.items()}}
        return master_append(self.root, rec, hmac_key)
