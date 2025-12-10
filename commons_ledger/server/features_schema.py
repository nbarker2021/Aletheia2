from typing import Dict, Any, List, Tuple

FEATURES_KEYS = {
    "AEGIS": dict,
    "FIVEWH1": dict,
    "GEO_GLOBAL": dict,
    "GEO_LOCAL": dict,
    "field_mass": dict,
    "_meta": dict,
}

def validate_features(feat: Dict[str,Any]) -> Tuple[bool, List[str]]:
    errs = []
    for k, typ in FEATURES_KEYS.items():
        if k not in feat:
            errs.append(f"missing key: {k}")
        elif not isinstance(feat[k], typ):
            errs.append(f"bad type for {k}: expected {typ.__name__}")
    # Basic checks
    aeg = feat.get("AEGIS",{})
    for key in ("Impact","Risk","Reversibility","Evidence"):
        v = aeg.get(key)
        if v is None or not isinstance(v,(int,float)) or not (0.0 <= v <= 1.0):
            errs.append(f"AEGIS.{key} must be in [0,1]")
    five = feat.get("FIVEWH1",{})
    for key in ("Who","What","Where","When","Why","How"):
        if key not in five:
            errs.append(f"FIVEWH1.{key} missing")
    gloc = feat.get("GEO_LOCAL",{})
    vec = gloc.get("vector")
    if not (isinstance(vec, list) and len(vec)==24):
        errs.append("GEO_LOCAL.vector must be length-24 list")
    field_mass = feat.get("field_mass",{})
    if not field_mass:
        errs.append("field_mass must have entries")
    else:
        s = sum(field_mass.values()) if all(isinstance(x,(int,float)) for x in field_mass.values()) else -1
        if s <= 0:
            errs.append("field_mass values must be numeric positive")
    return (len(errs)==0, errs)
