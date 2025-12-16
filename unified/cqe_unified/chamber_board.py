
import numpy as np, itertools
from enum import Enum
from typing import Dict, List, Tuple, Optional, Set

class ConstructionType(Enum):
    A="A"; B="B"; C="C"; D="D"
class PolicyChannel(Enum):
    TYPE_1=1; TYPE_2=2; TYPE_3=3; TYPE_4=4; TYPE_5=5; TYPE_6=6; TYPE_7=7; TYPE_8=8

class ChamberBoard:
    def __init__(self):
        self.conway_frame = np.array([[1,2,2,1],[3,4,4,3],[3,4,4,3],[1,2,2,1]])
        self.constructions = {
            ConstructionType.A: [(0,0),(0,3),(3,0),(3,3)],
            ConstructionType.B: [(0,1),(0,2),(1,0),(1,3),(2,0),(2,3),(3,1),(3,2)],
            ConstructionType.C: [(1,1),(1,2),(2,1),(2,2)],
            ConstructionType.D: [(0,1),(1,0),(2,3),(3,2)]
        }
        self.policy_params = {
            PolicyChannel.TYPE_1: {"base":0.1,"step":0.1,"pattern":"linear"},
            PolicyChannel.TYPE_2: {"base":0.05,"ratio":1.5,"pattern":"exponential"},
            PolicyChannel.TYPE_3: {"scale":0.3,"offset":0.1,"pattern":"logarithmic"},
            PolicyChannel.TYPE_4: {"amplitude":0.4,"frequency":1.0,"pattern":"harmonic"},
            PolicyChannel.TYPE_5: {"seed1":0.1,"seed2":0.2,"pattern":"fibonacci"},
            PolicyChannel.TYPE_6: {"primes":[2,3,5,7,11,13,17,19],"scale":0.05,"pattern":"prime"},
            PolicyChannel.TYPE_7: {"chaos_param":3.7,"initial":0.3,"pattern":"chaotic"},
            PolicyChannel.TYPE_8: {"weights":[0.2,0.15,0.25,0.1,0.1,0.05,0.1,0.05],"pattern":"balanced"}
        }
        self.enumeration_count = 0
        self.explored_gates = set()

    def _apply_phase_shift(self, params: Dict)->Dict:
        shifted = dict(params); pattern = params.get("pattern","linear")
        if pattern=="linear": shifted["step"] = params.get("step",0.1)*1.5
        elif pattern=="exponential": shifted["ratio"]= params.get("ratio",1.5)*0.8
        elif pattern=="logarithmic": shifted["scale"]= params.get("scale",0.3)*1.2
        elif pattern=="harmonic": shifted["frequency"]= params.get("frequency",1.0)*2.0
        elif pattern=="chaotic": shifted["chaos_param"]= params.get("chaos_param",3.7)*1.1
        return shifted

    def enumerate_gates(self, max_count: Optional[int]=None)->List[Dict]:
        gates=[]; 
        for construction in ConstructionType:
            for policy in PolicyChannel:
                for phase in [1,2]:
                    g = {"construction": construction, "policy_channel":policy, "phase":phase,
                         "gate_id": f"{construction.value}{policy.value}{phase}",
                         "cells": self.constructions[construction],
                         "parameters": dict(self.policy_params[policy])}
                    if phase==2:
                        g["parameters"] = self._apply_phase_shift(g["parameters"])
                    gates.append(g); self.enumeration_count += 1
                    if max_count and self.enumeration_count>=max_count: return gates
        return gates

    def generate_gate_vector(self, gate_config: Dict, index:int=0)->np.ndarray:
        construction = gate_config["construction"]; policy = gate_config["policy_channel"]
        phase = gate_config["phase"]; params = gate_config["parameters"]; pattern = params.get("pattern","linear")
        vector = np.zeros(8); cells = gate_config["cells"]
        for i,(r,c) in enumerate(cells):
            if i>=8: break
            base = self.conway_frame[r,c]/4.0
            if pattern=="linear": val = base + params.get("step",0.1)*index
            elif pattern=="exponential": val = base * (params.get("ratio",1.5)**(index%4))
            elif pattern=="logarithmic": val = base + params.get("scale",0.3)*np.log(index+1)
            elif pattern=="harmonic":
                freq=params.get("frequency",1.0); amp=params.get("amplitude",0.4); val= base + amp*np.sin(freq*index*np.pi/4)
            elif pattern=="fibonacci":
                val = base * min(2.0, (1+5**0.5)/2)  # simple golden cap
            elif pattern=="prime":
                primes=params.get("primes",[2,3,5,7]); val= base + params.get("scale",0.05)*primes[index%len(primes)]
            elif pattern=="chaotic":
                r=params.get("chaos_param",3.7); x=base
                for _ in range(index%10): x = r*x*(1-x); x = x%1.0
                val = x
            elif pattern=="balanced":
                w=params.get("weights",[0.125]*8); val = base * w[i%len(w)]
            else: val = base
            if phase==2: val = val*0.8 + 0.1
            vector[i] = val if i<4 else (val*0.7 + vector[i-4]*0.3)
        for i in range(len(cells),8):
            vector[i] = np.mean(vector[:len(cells)]) * (0.5 + 0.1*i)
        return np.clip(vector,0,1)
