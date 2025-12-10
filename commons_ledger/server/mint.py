from .features_schema import validate_features

COIN_LIST = ["MERIT","THETA","DAVINCI","VANGOGH","MYTHOS","EULER","HYPATIA","CURIE","TESLA","TURING","SENECA"]

def novelty_boost(novelty):
    return 1.0 + 0.5*novelty

def saturation_penalty(sat):
    return 1.0/(1.0+sat)

def compute_mint_splits(features, cfg):
    ok, errs = validate_features(features)
    if not ok:
        return {"ok":False,"errors":errs}
    A = features["AEGIS"]
    Gg = features["GEO_GLOBAL"]
    novelty = float(Gg.get("Novelty",0.0))
    field_mass = features["field_mass"]
    base = (A.get("Impact",0.0)*0.6 + A.get("Evidence",0.0)*0.4) * novelty_boost(novelty)
    base = max(0.0, min(base, 1.0))
    MERIT = 100.0 * base
    sat = float(cfg.get("domain_saturation",0.15))
    bridge = Gg.get("Bridges",[])
    bridge_strengths = {}
    for b in bridge:
        to_field = b.get("to_field")
        strength = float(b.get("strength",0.0))
        bridge_strengths[to_field] = max(bridge_strengths.get(to_field,0.0), strength)
    splits = {"MERIT": MERIT}
    total_mass = sum(field_mass.values())
    if total_mass <= 0:
        total_mass = 1.0
    for coin, w in field_mass.items():
        if coin not in COIN_LIST: 
            continue
        bf = bridge_strengths.get(coin, 0.0)
        amt = MERIT * (w/total_mass) * (1.0 + 0.5*bf) * saturation_penalty(sat)
        splits[coin] = amt
    return {"ok":True,"splits":splits}
