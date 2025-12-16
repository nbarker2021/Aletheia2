from typing import Dict, Any, List
import random, time, math
from .utils import now_receipt

def tsp_cost(route: List[int], dist: List[List[float]]) -> float:
    c=0.0
    for i in range(len(route)):
        c+=dist[route[i]][route[(i+1)%len(route)]]
    return c

def move_2opt(route: List[int], i: int, k: int) -> List[int]:
    return route[:i] + list(reversed(route[i:k+1])) + route[k+1:]

def ucb1(counts: List[int], rewards: List[float], t: int, c: float=1.414) -> int:
    vals=[]
    for n,r in zip(counts, rewards):
        if n==0: vals.append(float("inf"))
        else: vals.append((r/n) + c*math.sqrt(math.log(t+1)/(n)))
    return int(max(range(len(vals)), key=lambda i: vals[i]))

def run(problem: Dict[str, Any], budget:int=200, seed:int=42) -> Dict[str, Any]:
    rnd=random.Random(seed); dist=problem["dist"]; n=len(dist)
    route=list(range(n)); rnd.shuffle(route)
    best=route[:]; best_cost=tsp_cost(best, dist)
    windows=[2,3,4,5]; counts=[0]*len(windows); rewards=[0.0]*len(windows)
    receipts=[]
    for t in range(budget):
        arm=ucb1(counts,rewards,t); win=windows[arm]
        i=rnd.randrange(0, n-win); k=i+win
        cand=move_2opt(route,i,k); c_cost=tsp_cost(cand,dist); r_cost=tsp_cost(route,dist); delta=r_cost-c_cost
        if delta>0:
            route=cand
            if c_cost<best_cost: best, best_cost = cand[:], c_cost
        counts[arm]+=1; rewards[arm]+=max(0.0, delta)
        receipts.append({"t":t,"arm":int(arm),"win":int(win),"delta":float(delta),"best_cost":float(best_cost)})
    rec=now_receipt({"stage":"agrm.run","budget":int(budget),"best_cost":float(best_cost),"steps":len(receipts)})
    return {"best_route":best,"best_cost":best_cost,"receipts":receipts,"summary":rec}
