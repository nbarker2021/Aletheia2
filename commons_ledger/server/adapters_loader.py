import os, importlib.util

def load_adapters(root):
    adir = os.path.join(root, "adapters")
    loaded = []
    for name in os.listdir(adir):
        if not name.endswith(".py"): continue
        path = os.path.join(adir, name)
        spec = importlib.util.spec_from_file_location(name[:-3], path)
        try:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            loaded.append(mod)
        except Exception:
            continue
    return loaded

def extract_auto(root, text_or_obj):
    adapters = load_adapters(root)
    merged = {
        "AEGIS": {"Impact":0.5,"Risk":0.2,"Reversibility":0.8,"Evidence":0.5},
        "FIVEWH1": {k:{"clarity":0.5,"sources":[],"scope":"field"} for k in ["Who","What","Where","When","Why","How"]},
        "GEO_GLOBAL": {"Novelty":0.3,"Bridges":[],"Lattices":[],"Spectra":[]},
        "GEO_LOCAL": {"vector":[0]*24,"endorsements":[]},
        "field_mass": {"MERIT":1.0,"THETA":0.5,"DAVINCI":0.5,"VANGOGH":0.2,"MYTHOS":0.2,"EULER":0.5,"HYPATIA":0.3,"CURIE":0.6,"TESLA":0.4,"TURING":0.6,"SENECA":0.3},
        "_meta": {"adapters":[], "notes":""}
    }
    for mod in adapters:
        fn = getattr(mod, "extract_features", None)
        if fn:
            try:
                out = fn(text_or_obj)
                if isinstance(out, dict):
                    for k,v in out.items():
                        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                            merged[k].update(v)
                        else:
                            merged[k]=v
                    merged["_meta"]["adapters"].append(mod.__name__)
            except Exception:
                continue
    return merged
