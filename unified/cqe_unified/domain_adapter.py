
import numpy as np, hashlib
from typing import Dict, Any

class DomainAdapter:
    def __init__(self):
        self.feature_dim = 8
    def embed_p_problem(self, size:int, complexity_hint:int=1)->np.ndarray:
        f=np.zeros(8)
        f[0]=np.log10(max(1,size))/10.0; f[1]=0.1*complexity_hint; f[2]=0.8+0.1*np.sin(size*0.1)
        f[3]=min(0.9, (size**0.3)/100.0); f[4]=0.5+0.2*np.cos(size*0.05); f[5]=0.3+0.1*np.sin(size*0.03)
        f[6]=0.4+0.15*np.cos(size*0.07); f[7]=0.2+0.1*np.sin(size*0.02); return f
    def embed_np_problem(self, size:int, nondet:float=0.8)->np.ndarray:
        f=np.zeros(8); f[0]=np.log10(max(1,size))/10.0; f[1]=0.9+0.1*nondet; f[2]=nondet
        f[3]=min(1.0, (size**0.5)/50.0); f[4]=0.7+0.3*np.sin(size*0.1*nondet); f[5]=0.6+0.2*np.cos(size*0.08*nondet)
        f[6]=0.8+0.2*np.sin(size*0.12*nondet); f[7]=0.5+0.3*np.cos(size*0.15*nondet); return f
    def embed_optimization_problem(self, variables:int, constraints:int, objective_type:str="linear")->np.ndarray:
        f=np.zeros(8); f[0]=np.log10(max(1,variables))/10.0; f[1]=np.log10(max(1,constraints))/10.0
        enc={"linear":0.2,"quadratic":0.5,"nonlinear":0.8}; f[2]=enc.get(objective_type,0.5)
        density=constraints/max(1,variables); f[3]=min(1.0, density/10.0)
        f[4]=0.5+0.2*np.sin(variables*0.1); f[5]=0.4+0.3*np.cos(constraints*0.05)
        f[6]=0.6+0.1*np.sin((variables+constraints)*0.03); f[7]=0.3+0.2*np.cos(density); return f
    def embed_scene_problem(self, scene_complexity:int, narrative_depth:int, character_count:int)->np.ndarray:
        f=np.zeros(8); f[0]=min(1.0, scene_complexity/100.0); f[1]=min(1.0, narrative_depth/50.0); f[2]=min(1.0, character_count/20.0)
        tension=(scene_complexity*narrative_depth)/(character_count+1); f[3]=min(1.0, tension/1000.0)
        f[4]=0.4+0.3*np.sin(scene_complexity*0.1); f[5]=0.5+0.2*np.cos(narrative_depth*0.2)
        f[6]=0.3+0.4*np.sin(character_count*0.3); f[7]=0.6+0.1*np.cos(tension*0.01); return f
    def hash_to_features(self, data:str)->np.ndarray:
        hb=hashlib.sha256(data.encode()).digest(); return np.array([b/255.0 for b in hb[:8]])
    def validate_features(self, f:np.ndarray)->bool:
        if len(f)!=8: return False
        return not (np.any(f<-2.0) or np.any(f>2.0))
