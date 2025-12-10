import json, hashlib, time, os
class Receipts:
    def __init__(self, path: str, anchor_period: float = 3600.0):
        self.path=path; self.anchor_period=anchor_period; os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f=open(self.path,"a",encoding="utf-8"); self._roll=[]; self._t0=time.time()
    def write(self, kind:str, what, gov):
        rec={"ts":time.time(),"kind":kind,"WHAT":what,"GOV":gov}
        line=json.dumps(rec,sort_keys=True); self._f.write(line+"\n"); self._f.flush(); self._roll.append(line.encode())
        if time.time()-self._t0>=self.anchor_period: self.anchor()
    def anchor(self):
        if not self._roll: return
        h=hashlib.sha256()
        for ln in self._roll: h.update(ln)
        self._f.write(json.dumps({"ts":time.time(),"kind":"anchor","root":h.hexdigest()})+"\n"); self._f.flush()
        self._roll.clear(); self._t0=time.time()
    def close(self): self.anchor(); self._f.close()
