from typing import Dict, Any

def plan_image(ctx: Dict[str, Any]) -> Dict[str, Any]:
    return {"mode":"image","params":{"res":"1024x1024","steps":30,"seed":ctx.get("seed",0)},"ops":ctx.get("ops",[])}

def plan_video(ctx: Dict[str, Any]) -> Dict[str, Any]:
    return {"mode":"video","params":{"res":"768p","frames":48,"fps":12,"seed":ctx.get("seed",0)},"ops":ctx.get("ops",[])}

def plan_text(ctx: Dict[str, Any]) -> Dict[str, Any]:
    return {"mode":"text","params":{"max_tokens":512,"seed":ctx.get("seed",0)},"ops":ctx.get("ops",[])}

def plan_audio(ctx: Dict[str, Any]) -> Dict[str, Any]:
    return {"mode":"audio","params":{"sr":48000,"seconds":8,"seed":ctx.get("seed",0)},"ops":ctx.get("ops",[])}
