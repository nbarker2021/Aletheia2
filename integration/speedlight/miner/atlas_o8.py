def extract_O8(meta: dict):
    version=float(meta.get("version",0)); chain_tight=float(meta.get("height",0)%2016)
    merkle_entropy=float(meta.get("txcount",1))**0.5; diff_ratio=1.0
    weight=float(meta.get("weight",1_000_000))/4_000_000.0; script_mix=float(meta.get("sigops",0))/20000.0
    flags=float(len(meta.get("coinbaseaux",{}).get("flags",""))%64)/64.0; cadence=float(meta.get("template_age",0.0))
    return [version,chain_tight,merkle_entropy,diff_ratio,weight,script_mix,flags,cadence]
def embed_E8(vec8): return [float(x) for x in vec8]
