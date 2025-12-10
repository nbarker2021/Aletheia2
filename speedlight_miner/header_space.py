from dataclasses import dataclass
import time
@dataclass
class RailsState: nonce:int; extranonce:int; merkle_class:int; version:int; timestamp:int
@dataclass
class RailConfig: Kx:int=4096; Km:int=256; Kv:int=64; time_nudge:int=1
class HeaderSpace:
    def __init__(self, tpl, seed:int, rail_cfg:RailConfig):
        t = int(time.time())
        self.tpl=tpl; self.cfg=rail_cfg
        self.state=RailsState(0,0,0,tpl.get("version",0x20000000), max(t,tpl.get("curtime",t)))
    def legality_guard(self, new_time:int, mtp_floor:int)->int:
        up=max(new_time, mtp_floor); cap=self.tpl.get("curtime", up)+7200; return min(up, cap)
    def step_nonce(self): self.state.nonce = (self.state.nonce+1)&0xFFFFFFFF
    def step_extranonce(self): self.state.extranonce = (self.state.extranonce+1)&0xFFFFFFFF
    def step_merkle_class(self): self.state.merkle_class += 1
    def roll_version(self): self.state.version = (self.state.version ^ 0x1) & 0xFFFFFFFF
    def nudge_time(self, mtp_floor:int): self.state.timestamp = self.legality_guard(self.state.timestamp+self.cfg.time_nudge, mtp_floor)
