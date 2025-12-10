from .e8_tiler import E8Tiler
def chamber_key_from_vec(vec8):
    try:
        import importlib
        mod=importlib.import_module("geometric_transformer_standalone")
        if hasattr(mod,"chamber_key"): return mod.chamber_key(vec8)
    except Exception: pass
    h=0
    for x in vec8: h=(h*1315423911+int(x)) & 0xffffffffffffffff
    return h
def owns(agent_id:int, agent_count:int, key:int)->bool: return (key % max(1,agent_count)) == agent_id
