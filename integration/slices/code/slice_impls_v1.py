
from typing import Dict, Any
from .slices import slice_op
from .embedding_e8 import embed_string
from .fractal_mandelbrot import analyze_string
from .toroidal import generate_toroidal_shell
from .objective_phi import feature_pack, compute_phi
from .semantics import extract_semantics
from .governance import Governance

# lane indices are 0..7; we alternate parity E/O as a simple rule of thumb
@slice_op("ingest_text", category="IO", lane=0, parity="E", provides=["text_digest"])
def ingest_text(system, payload: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    text = payload.get("text","")
    meta = system.io.ingest_text(text)
    return {"text_digest": meta["digest"]}

@slice_op("e8_embed", category="E8", lane=1, parity="O", requires=["text_digest"], provides=["e8"])
def e8_embed(system, payload: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    text = payload.get("text","")
    e8 = embed_string(text, seed=system.cfg.seed)
    return {"e8": e8}

@slice_op("fractal", category="FRACTAL", lane=2, parity="E", requires=["text_digest"], provides=["fractal"])
def fractal(system, payload: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    text = payload.get("text","")
    fr = analyze_string(text) if system.cfg.enable_mandelbrot else {"behavior": "SKIP"}
    return {"fractal": fr}

@slice_op("toroidal", category="TORUS", lane=3, parity="O", provides=["toroidal_patterns"])
def toroidal(system, payload: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    tor = generate_toroidal_shell(n_points=64, seed=system.cfg.seed) if system.cfg.enable_toroidal else []
    patterns = {}
    for p in tor:
        patterns[p["pattern"]] = patterns.get(p["pattern"], 0) + 1
    return {"toroidal_patterns": patterns}

@slice_op("phi", category="OBJECTIVE", lane=4, parity="E", requires=["e8"], provides=["phi","phi_features"])
def phi(system, payload: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    e8 = state["e8"]
    features = feature_pack(
        geom = max(0.0, 1.0 - e8["root_distance"]),
        parity = 0.5,
        sparsity = 0.4,
        kissing = 0.3
    )
    val = compute_phi(features, system.cfg.phi_weights)
    return {"phi": val, "phi_features": features}

@slice_op("semantics", category="SEMANTICS", lane=5, parity="O", requires=["e8"], provides=["semantics"])
def semantics(system, payload: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    e8 = state["e8"]
    sem = extract_semantics(e8_distance=e8["root_distance"], angle_hint=0.2)
    return {"semantics": sem}

@slice_op("governance", category="GOV", lane=6, parity="E", requires=["phi_features"], provides=["gov_results","v_total","band"])
def governance(system, payload: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    gates_in = {
        "W4": state["phi_features"]["geom"],
        "W80": state["phi_features"]["parity"],
        "Wexp": state["phi_features"]["sparsity"],
        "LAWFUL": state["phi_features"]["kissing"]
    }
    gov = Governance().evaluate(gates_in)
    # Simple V_total as in validation module
    from .validation import compute_v_total, band_for
    v = compute_v_total(scores={k:v["score"] for k,v in gov.items()}, weights={"W4":0.3,"W80":0.3,"Wexp":0.2,"LAWFUL":0.2})
    band = band_for(v)
    return {"gov_results": gov, "v_total": v, "band": band}


# === Advanced slices ===

@slice_op("mandelbrot_null_model", category="FRACTAL", lane=2, parity="E", requires=["fractal"], provides=["mndl_p_value","mndl_effect_size","mndl_stat","null_mean","null_std"])
def mandelbrot_null_model(system, payload: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute significance of a simple alignment between behavior and a 3-6-9 style mapping.
    Observed statistic: S=1 if behavior in {"ESCAPING"→"OUTWARD","PERIODIC"→"TRANSFORM"} else 0.
    Null: permute labels 1,000x and compute rank-based p plus Cohen's d.
    """
    behavior = state["fractal"]["behavior"]
    # Map behavior → sacred class (demo)
    sacred_map = {"ESCAPING":"OUTWARD","BOUNDED":"INWARD","PERIODIC":"TRANSFORM"}
    obs = 1.0 if sacred_map.get(behavior) in ("OUTWARD","TRANSFORM") else 0.0

    # Build a tiny cohort by perturbing the text signature (if available)
    # Here we just reuse the same behavior 10× to keep it headless-speedy; real code would sample nearby c's.
    cohort = [obs for _ in range(10)]
    mu_obs = sum(cohort)/len(cohort)

    # Null via permutations: Bernoulli(0.5) draws as a stand-in
    N = 1000
    null = []
    for _ in range(N):
        # shuffled labels → draw same-size cohort with p=0.5
        k = sum(1 if random.random()<0.5 else 0 for _ in range(len(cohort)))
        null.append(k/len(cohort))

    mu = sum(null)/N
    sd = (sum((x-mu)**2 for x in null)/(N-1))**0.5 or 1.0

    # Rank-based p-value
    rank = sum(1 for x in null if x >= mu_obs)
    p = (rank + 1) / (N + 1)

    # Effect size (Cohen's d) of observed against null mean
    d = (mu_obs - mu) / sd
    return {"mndl_p_value": p, "mndl_effect_size": d, "mndl_stat": mu_obs, "null_mean": mu, "null_std": sd}
@slice_op("toroidal_e8_nearest_root", category="TORUS", lane=3, parity="O", provides=["torus_flip_rate","torus_root_distance_mean"])
def toroidal_e8_nearest_root(system, payload: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Project the centroid of toroidal shell points into E8 and estimate robustness:
    small perturbations should rarely flip the nearest-root identity.
    """
    from .embedding_e8 import _normalize, nearest_root
    # if toroidal patterns were not computed, generate a tiny shell
    pts = []
    if "toroidal_patterns" not in state:
        pts = generate_toroidal_shell(n_points=64, seed=system.cfg.seed)
    else:
        # regenerate to have coordinates (patterns-only state doesn't retain coords); small demo shell
        pts = generate_toroidal_shell(n_points=64, seed=system.cfg.seed)

    # build a naive 8D feature by aggregating simple moments (demo)
    # (mean x,y,z, var x,y,z, and a bias) → normalized
    xs = [p["x"] for p in pts]; ys = [p["y"] for p in pts]; zs = [p["z"] for p in pts]
    def var(a):
        m = sum(a)/len(a)
        return sum((x-m)**2 for x in a)/len(a)
    vec = [sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs),
           var(xs), var(ys), var(zs), 0.0, 1.0]
    v = _normalize(vec)
    base_root, base_dist = nearest_root(v)

    # perturbations
    flips = 0; D = []
    for _ in range(50):
        vp = [x + random.uniform(-1e-3, 1e-3) for x in v]
        rp, dp = nearest_root(vp)
        D.append(dp)
        if rp != base_root:
            flips += 1
    flip_rate = flips/50.0
    mean_dist = sum(D)/len(D)
    return {"torus_flip_rate": flip_rate, "torus_root_distance_mean": mean_dist}

@slice_op("uvibs_windows", category="GOV", lane=6, parity="E", requires=["phi_features"], provides=["uvibs_windows"])
def uvibs_windows(system, payload: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate multiple windows (W4/W80/Wexp/Monster) as a structured dict.
    Placeholder scoring: tie each to a phi_feature for now; real versions replace these.
    """
    pf = state["phi_features"]
    windows = {
        "W4": {"score": pf["geom"], "threshold": 0.7, "msg": "Coxeter-plane / local geometry"},
        "W80": {"score": pf["parity"], "threshold": 0.7, "msg": "E8x10 / lane parity window"},
        "Wexp": {"score": pf["sparsity"], "threshold": 0.6, "msg": "Exponential window / sparsity"},
        "Monster": {"score": pf["kissing"], "threshold": 0.8, "msg": "24D projection sanity"}
    }
    for k,v in windows.items():
        v["pass"] = v["score"] >= v["threshold"]
    return {"uvibs_windows": windows}


@slice_op("pose_gauge_invariance", category="TORUS", lane=3, parity="O", provides=["pose_invariance_stat"])
def pose_gauge_invariance(system, payload: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate two toroidal shells with theta/phi offsets and compare pattern distributions.
    Report L1 difference; smaller is more invariant.
    """
    base = generate_toroidal_shell(n_points=256, seed=system.cfg.seed, theta_offset=0.0, phi_offset=0.0)
    off  = generate_toroidal_shell(n_points=256, seed=system.cfg.seed, theta_offset=0.37, phi_offset=0.23)
    def dist(pts):
        d = {}
        for p in pts:
            d[p["pattern"]] = d.get(p["pattern"],0)+1
        total = sum(d.values())
        for k in d: d[k] /= total
        return d
    db, do = dist(base), dist(off)
    keys = set(db)|set(do)
    l1 = sum(abs(db.get(k,0)-do.get(k,0)) for k in keys)
    return {"pose_invariance_stat": 1.0 - 0.5*l1}  # 1==identical, 0==disjoint

@slice_op("phi_regimen_bench", category="OBJECTIVE", lane=4, parity="E", requires=["phi_features"], provides=["phi_bench"])
def phi_regimen_bench(system, payload: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare three simplistic regimens on the same features snapshot:
    A: static weights; B: annealed (slightly different weights); C: adaptive (rebalance by feature magnitude).
    Report final phi and deltas.
    """
    from .objective_phi import compute_phi
    feats = state["phi_features"]
    W_A = system.cfg.phi_weights
    W_B = {k:(v*0.95 if k!="geom" else v*1.05) for k,v in W_A.items()}
    # Adaptive: prioritize larger features
    s = sum(feats.values()) or 1.0
    W_C = {k:(0.25 + 0.75*(feats[k]/s))/4 for k in feats}  # renormalize roughly

    phi_A = compute_phi(feats, W_A)
    phi_B = compute_phi(feats, W_B)
    phi_C = compute_phi(feats, W_C)
    return {"phi_bench": {"A":phi_A, "B":phi_B, "C":phi_C, "C_minus_A":phi_C - phi_A, "B_minus_A":phi_B - phi_A}}

@slice_op("semantics_calibration", category="SEMANTICS", lane=5, parity="O", requires=["e8"], provides=["sem_calibration"])
def semantics_calibration(system, payload: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a tiny synthetic cohort around the current e8 root distance, produce confidences via extract_semantics,
    and compute Brier score + a simple ECE over 5 bins (toy). Labels are derived from a threshold on distance.
    """
    import math
    e8d = state["e8"]["root_distance"]
    xs = [max(0.0, min(1.0, e8d + (i-5)*0.02)) for i in range(11)]
    preds = []
    labels = []
    for x in xs:
        sem = extract_semantics(e8_distance=x, angle_hint=0.2)
        p = max(0.0, min(1.0, sem["confidence"]))  # already 0..1-ish
        y = 1 if x < 0.25 else 0  # toy ground truth: NEAR→1
        preds.append(p); labels.append(y)
    # Brier
    brier = sum((p - y)**2 for p,y in zip(preds, labels)) / len(labels)
    # ECE with 5 bins
    bins = [[] for _ in range(5)]
    for p,y in zip(preds, labels):
        idx = min(4, int(p*5))
        bins[idx].append((p,y))
    ece = 0.0
    for b in bins:
        if not b: continue
        conf = sum(p for p,_ in b)/len(b)
        acc = sum(y for _,y in b)/len(b)
        ece += (len(b)/len(labels)) * abs(conf-acc)
    return {"sem_calibration": {"brier": brier, "ece5": ece, "n": len(labels)}}


@slice_op("morsr_complete_traversal", category="GOV", lane=6, parity="E", requires=["e8"], provides=["morsr_best","morsr_overlay_path"])
def morsr_complete_traversal(system, payload: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    from .e8_lattice import normalized_e8_roots, chamber_signature
    v = state["e8"]["vector"]
    roots = normalized_e8_roots()
    best = {"score": -1e9, "index": None, "root": None, "chamber": None}
    scores = []
    chambers = {}
    from .embedding_e8 import cosine
    for i, r in enumerate(roots):
        s = cosine(v, r)
        ch = chamber_signature(r)
        scores.append(s)
        chambers[ch] = chambers.get(ch, 0) + 1
        system.emit("morsr_node", {"i": i, "score": s, "chamber": ch})
        if s > best["score"]:
            best.update({"score": s, "index": i, "root": r, "chamber": ch})
    overlay = {
        "best": best,
        "score_min": min(scores), "score_max": max(scores), "score_mean": sum(scores)/len(scores),
        "chambers": chambers, "n": len(scores)
    }
    import json, os
    path = os.path.join(system.cfg.artifacts_dir, f"morsr_overlay_{system.run_id}.json")
    with open(path, "w") as fh:
        json.dump(overlay, fh, indent=2)
    system.emit("morsr_overlay", {"artifact": path})
    return {"morsr_best": best, "morsr_overlay_path": path}

def _digital_root(n: int) -> int:
    n = abs(n)
    while n >= 10:
        s = 0
        while n > 0:
            s += n % 10
            n //= 10
        n = s
    return n

@slice_op("carlson_correspondence_check", category="EVIDENCE", lane=7, parity="O", provides=["carlson_exhibit"])
def carlson_correspondence_check(system, payload: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    invariants = {"dim_E8": 8, "roots": 240, "weyl_order": 696729600, "coxeter_number": 30}
    drs = {k: _digital_root(v) for k,v in invariants.items()}
    sacred_set = {3,6,9}
    observed = sum(1 for v in drs.values() if v in sacred_set)
    import random, json, os
    s = "".join(str(v) for v in invariants.values())
    N = 200
    null = []
    for _ in range(N):
        t = "".join(random.sample(s, len(s)))
        chunks = [t[:1], t[1:4], t[4:13], t[13:]]
        vals = [int(c) if c else 0 for c in chunks]
        d = [_digital_root(x) for x in vals]
        null.append(sum(1 for x in d if x in sacred_set))
    mu = sum(null)/N
    sd = (sum((x-mu)**2 for x in null)/(N-1))**0.5 or 1.0
    z = (observed - mu)/sd
    exhibit = {"invariants": invariants, "digital_roots": drs, "observed_sacred_hits": observed, "null_mean": mu, "null_sd": sd, "z_score": z}
    path = os.path.join(system.cfg.artifacts_dir, f"carlson_exhibit_{system.run_id}.json")
    with open(path, "w") as fh:
        json.dump(exhibit, fh, indent=2)
    system.emit("carlson_exhibit", {"artifact": path})
    return {"carlson_exhibit": path}

@slice_op("governance_recommendations", category="GOV", lane=6, parity="E", requires=["phi_features"], provides=["recommendations"])
def governance_recommendations(system, payload: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    feats = state["phi_features"]
    recs = []
    if feats["geom"] < 0.7:
        recs.append("Improve geometric alignment: run morsr_complete_traversal to search better root alignment.")
    if feats["parity"] < 0.7:
        recs.append("Strengthen lane parity features; consider W80 tuning.")
    if feats["sparsity"] < 0.6:
        recs.append("Promote sparsity; prune low-signal atoms.")
    if feats["kissing"] < 0.8:
        recs.append("Check local packing; adjust toroidal shell parameters.")
    return {"recommendations": recs}


@slice_op("pose_optimizer", category="GEOM", lane=2, parity="E", requires=["e8"], provides=["pose_optimized","pose_gain"])
def pose_optimizer(system, payload, state):
    """
    Search simple pose transforms (cyclic shifts and sign flips) to improve nearest-root cosine.
    """
    from .embedding_e8 import nearest_root
    v = state["e8"]["vector"]
    best = (v, nearest_root(v)[1])  # distance = 1-cos; lower better
    # generate candidates
    def shifts(vec):
        for s in range(len(vec)):
            yield vec[s:]+vec[:s]
    def sign_flips(vec):
        for mask in range(1<<3):  # flip up to 3 dims for speed
            vv = vec[:]
            for i in range(3):
                if (mask>>i)&1: vv[i] = -vv[i]
            yield vv
    for cand in shifts(v):
        for c2 in sign_flips(cand):
            d = nearest_root(c2)[1]
            if d < best[1]:
                best = (c2, d)
    gain = state["e8"]["root_distance"] - best[1]
    system.emit("pose_search", {"gain": gain, "improved": gain>0})
    return {"pose_optimized": best[0], "pose_gain": gain}

@slice_op("uvibs_monster_validate", category="GOV", lane=6, parity="E", requires=["e8"], provides=["uvibs_monster"])
def uvibs_monster_validate(system, payload, state):
    """
    Compute UVIBS (W4/W80/Wexp) and a Monster-style 24D heuristic, emit scores and pass flags.
    """
    from .uvibs_monster import w4_geometry, w80_parity_stability, wexp_sparsity, monster_24D_projection, monster_pass
    v = state["e8"]["vector"]
    w4 = w4_geometry(v)
    w80 = w80_parity_stability(v)
    wexp = wexp_sparsity(v)
    inv = monster_24D_projection(v)
    mon = monster_pass(inv)
    windows = {
        "W4": {"score": w4, "threshold": 0.7, "pass": w4>=0.7},
        "W80": {"score": w80, "threshold": 0.7, "pass": w80>=0.7},
        "Wexp": {"score": wexp, "threshold": 0.6, "pass": wexp>=0.6},
        "Monster": {"score": mon, "threshold": 0.6, "pass": mon>=0.6, "invariants": inv}
    }
    system.emit("uvibs_monster", {"windows": windows})
    return {"uvibs_monster": windows}

@slice_op("glyphs_apply_overlay", category="GLYPH", lane=0, parity="E", provides=["glyph_receipt"])
def glyphs_apply_overlay(system, payload, state):
    """
    Apply an overlay (program) for a glyph on a simple pose state using one rung with parallel rails.
    payload: {"glyph":"A","rails":[["×","÷","#2"],["÷","~"]]}
    """
    from .glyphs_lambda import init_state, run_rung, compute_phi, OverlayRegistry
    glyph = payload.get("glyph","A")
    rails = payload.get("rails",[["×","÷","#2"],["÷","~"]])
    S0 = init_state(n=8, seed=system.cfg.seed)
    S1, rec = run_rung(S0, rails)
    system.emit("glyph_rung", {"glyph": glyph, **rec})
    return {"glyph_receipt": rec}

@slice_op("hyperperm_update", category="GLYPH", lane=1, parity="O", provides=["hyperperm_status"])
def hyperperm_update(system, payload, state):
    """
    Update hyperpermutation oracle with an observed ordering from a given channel.
    payload: {"atoms":["×","÷","%","~"], "sequence":["%","÷","~"], "channel":"runner"}
    """
    from .glyphs_lambda import HyperpermOracle
    if not hasattr(system, "_hyperperm"):
        system._hyperperm = HyperpermOracle()
    atoms = payload.get("atoms",["×","÷","%","~"])
    seq = payload.get("sequence",["%","÷","~"])
    ch = payload.get("channel","runner")
    status = system._hyperperm.add_order(atoms, seq, ch)
    system.emit("hyperperm", {"key":"|".join(sorted(atoms)), "locked": status["locked"], "count": len(status["orders"])})
    return {"hyperperm_status": {"locked": status["locked"], "n": len(status["orders"]) }}
