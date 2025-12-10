class MDHGCache:
    def __init__(self): self.stats={}
    def note(self, key:int, rebuild_cost:int):
        s=self.stats.setdefault(key,{"hits":0,"cost":0}); s["hits"]+=1; s["cost"]+=rebuild_cost
    def score(self, key:int)->float:
        s=self.stats.get(key); 
        if not s: return 0.0
        return s["hits"]/max(1,s["cost"])
    def hot(self, threshold:float=2.0): return [k for k in self.stats if self.score(k)>=threshold]
