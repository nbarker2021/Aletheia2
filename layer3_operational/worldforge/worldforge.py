from typing import Dict, Any, List
from .operators import P5_startcap, residue_endcap, B_obs, B_soft, B_higgs, B_ward, Bridge
from .receipts import stamp
from .renderers import plan_image, plan_video, plan_text, plan_audio

def compose(seed:int, prompt:str, modes:List[str], lenses:List[str]) -> Dict[str, Any]:
    ctx = {"seed":int(seed), "prompt":prompt, "lenses":lenses, "ops":[]}
    for op in (P5_startcap, B_obs, residue_endcap, B_soft, B_higgs, B_ward, Bridge):
        if op is B_obs: ctx = op(ctx, obs_bits=3)
        elif op is residue_endcap: ctx = op(ctx, harmonics=(24,))
        else: ctx = op(ctx)
    plans=[]
    for m in modes:
        if m=='image': plans.append(plan_image(ctx))
        elif m=='video': plans.append(plan_video(ctx))
        elif m=='text': plans.append(plan_text(ctx))
        elif m=='audio': plans.append(plan_audio(ctx))
    return stamp(ctx, {"render_plan":{"plans":plans}})
