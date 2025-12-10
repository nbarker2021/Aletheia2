from typing import Dict, Any
def P5_startcap(ctx: Dict[str, Any]) -> Dict[str, Any]:
    ctx=dict(ctx); ctx.setdefault("ops",[]).append("P5_startcap"); return ctx
def residue_endcap(ctx: Dict[str, Any], harmonics=(24,)) -> Dict[str, Any]:
    ctx=dict(ctx); ctx.setdefault("ops",[]).append("Rρ_endcap"); ctx["harmonics"]=list(harmonics); return ctx
def B_obs(ctx: Dict[str, Any], obs_bits:int=3) -> Dict[str, Any]:
    ctx=dict(ctx); ctx.setdefault("ops",[]).append("B_obs"); ctx.setdefault("ledger",{}).setdefault("info_bits",0); ctx["ledger"]["info_bits"]+=int(obs_bits); return ctx
def B_soft(ctx: Dict[str, Any]) -> Dict[str, Any]:
    ctx=dict(ctx); ctx.setdefault("ops",[]).append("B_soft"); return ctx
def B_higgs(ctx: Dict[str, Any], parity_order:str='odd-first') -> Dict[str, Any]:
    ctx=dict(ctx); ctx.setdefault("ops",[]).append("B_higgs"); ctx["parity_order"]=parity_order; return ctx
def B_ward(ctx: Dict[str, Any]) -> Dict[str, Any]:
    ctx=dict(ctx); ctx.setdefault("ops",[]).append("B_ward"); return ctx
def Bridge(ctx: Dict[str, Any], src='E8', dst='Niemeier') -> Dict[str, Any]:
    ctx=dict(ctx); ctx.setdefault("ops",[]).append("Bridge"); ctx["bridge"]={"src":src,"dst":dst}; return ctx
