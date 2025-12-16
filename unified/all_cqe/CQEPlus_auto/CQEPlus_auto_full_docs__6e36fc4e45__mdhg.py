from typing import Dict, Any, Optional
import hashlib
from .utils import now_receipt, sha256_json

def jump_hash(key_hash:int, buckets:int) -> int:
    b, j = -1, 0
    while j < buckets:
        b = j
        key_hash = (key_hash * 2862933555777941757 + 1) & ((1<<64)-1)
        j = int((b + 1) * (1<<31) / ((key_hash >> 33) + 1))
    return b

class TinyLFU:
    def __init__(self, size:int=8192): self.size=size; self.table=[0]*size
    def _idx(self, key:str)->int: return int(hashlib.blake2b(key.encode(), digest_size=8).hexdigest(),16)%self.size
    def admit(self, key:str): i=self._idx(key); self.table[i]=min(255, self.table[i]+1)
    def score(self, key:str)->int: return self.table[self._idx(key)]

class MDHG:
    def __init__(self, buckets:int=256, lfu_size:int=8192):
        self.buckets=buckets; self.lfu=TinyLFU(lfu_size); self.store={}; self.place={}
    def _place(self, key:str)->int:
        h=int(hashlib.blake2b(key.encode(), digest_size=8).hexdigest(),16); return jump_hash(h, self.buckets)
    def put(self, item: Dict[str, Any]):
        if "key" not in item:
            item=dict(item); item["key"]=sha256_json(item)
        key=item["key"]; idx=self._place(key); self.store[key]=item; self.place[key]=idx; self.lfu.admit(key)
        return now_receipt({"stage":"mdhg.put","key":key,"bucket":idx})
    def propose_topK(self, k:int=8):
        keys=list(self.store.keys()); keys.sort(key=lambda x:self.lfu.score(x), reverse=True); return keys[:k]
    def manifest(self): return now_receipt({"stage":"mdhg.manifest","count":len(self.store)})
