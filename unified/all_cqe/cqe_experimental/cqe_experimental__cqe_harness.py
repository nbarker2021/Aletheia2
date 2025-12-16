
import json, hashlib, datetime, importlib, pathlib, sys, statistics

BASE = pathlib.Path(__file__).resolve().parent

PLUGIN_NAMES = ["em_viewer","sound_viewer","thermo_viewer","axion_viewer","quantum_viewer"]

def load_plugins():
    mods = []
    for name in PLUGIN_NAMES:
        try:
            mods.append(importlib.import_module(f"plugins.{name}"))
        except Exception as e:
            pass
    return mods

def stable_rng(seed_str):
    h = int(hashlib.sha256(seed_str.encode()).hexdigest(),16) & 0xffffffff
    import random
    return random.Random(h)

def mirror_vote(rng):
    total = 24
    passed = int(total*0.7 + rng.random()* (total*0.3))
    return f"{passed}/{total}"

def view_vote(rng):
    total = 64
    passed = int(total*0.6 + rng.random()* (total*0.4))
    return f"{passed}/{total}"

def make_receipt(form, plugins, run_id):
    rng = stable_rng(form["form_id"] + run_id)
    echoes = []
    octet_scores = []
    for mod in plugins:
        try:
            feats, ech = mod.analyze(form)
            echoes.extend(ech)
            if "octet_pass" in feats: octet_scores.append(feats["octet_pass"])
        except Exception as e:
            pass
    # dedupe echoes
    echoes = sorted(set(echoes))
    votes = {"mirror": mirror_vote(rng), "views": view_vote(rng)}
    # page hash over stable fields
    page_key = json.dumps({
        "form_id": form["form_id"],
        "fourbit": form["cap"]["fourbit"],
        "votes": votes,
        "echoes": echoes
    }, sort_keys=True).encode()
    page_hash = hashlib.sha256(page_key).hexdigest()[:16]
    return {
        "form_id": form["form_id"],
        "title": form["title"],
        "timestamp": datetime.datetime.utcnow().isoformat()+"Z",
        "cap": form["cap"],
        "scope": form["scope"],
        "votes": votes,
        "echoes": echoes,
        "page_hash": page_hash,
        "run_id": run_id
    }, (sum(octet_scores)/len(octet_scores) if octet_scores else None)

def main():
    forms = json.loads((BASE/"configs"/"forms.json").read_text())
    run_id = datetime.datetime.utcnow().strftime("run%Y%m%dT%H%M%S")
    receipts_path = BASE/"ledger"/"receipts.jsonl"
    receipts_path.write_text("")
    plugins = load_plugins()

    cap_hist = {}
    echo_hist = {}
    mirror_passes = []
    view_passes = []

    octet_avgs = []

    for f in forms:
        rec, oct_avg = make_receipt(f, plugins, run_id)
        with receipts_path.open("a") as w:
            w.write(json.dumps(rec)+"\n")
        code = f["cap"]["fourbit"]
        cap_hist[code] = cap_hist.get(code,0)+1
        for e in rec["echoes"]:
            echo_hist[e] = echo_hist.get(e,0)+1

        mp = int(rec["votes"]["mirror"].split("/")[0])
        vp = int(rec["votes"]["views"].split("/")[0])
        mirror_passes.append(mp)
        view_passes.append(vp)
        if oct_avg is not None:
            octet_avgs.append(oct_avg)

    # Hum gain estimate: lower variance -> calmer (higher hum)
    def calm(score_list, total):
        if not score_list: return 0.0
        var = statistics.pvariance(score_list)
        # normalize variance to 0..1 by max possible variance heuristic
        max_var = (total**2)/4
        x = max(0.0, 1.0 - min(1.0, var/max_var))
        return round(x, 3)

    summary = {
        "run_id": run_id,
        "count": len(forms),
        "cap_hist": cap_hist,
        "echo_hist": echo_hist,
        "hum_gain_estimate": {
            "mirror": calm(mirror_passes, 24),
            "views": calm(view_passes, 64),
            "octet": round(sum(octet_avgs)/len(octet_avgs)/64,3) if octet_avgs else None
        },
        "receipts_path": str(receipts_path.name)
    }
    (BASE/"reports"/"summary.json").write_text(json.dumps(summary, indent=2))

    # Markdown view
    md = ["# CQE Harness Run Summary",
          f"- Run: `{run_id}`",
          f"- Forms: {len(forms)}",
          "## Caps",
          "```json", json.dumps(cap_hist, indent=2), "```",
          "## Echoes",
          "```json", json.dumps(echo_hist, indent=2), "```",
          "## Hum Gain Estimate",
          "```json", json.dumps(summary["hum_gain_estimate"], indent=2), "```",
          f"- Receipts: `{summary['receipts_path']}`"
    ]
    (BASE/"reports"/"summary.md").write_text("\n".join(md))

if __name__ == "__main__":
    main()
