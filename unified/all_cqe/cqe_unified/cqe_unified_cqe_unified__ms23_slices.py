
import json, time, hashlib, os
from pathlib import Path
from typing import Dict, Any, List
from .slices import slice_op

def _sha(s: str)->str:
    return hashlib.sha256(s.encode()).hexdigest()

@slice_op("ms2_label_and_hyperperm", category="MS2", lane=2, parity="E", requires=["dwm_charter"], provides=["labels","overlays","oracle"])
def ms2_label_and_hyperperm(system, payload, state):
    data: List[Dict[str,Any]] = payload.get("data", [])
    artifacts = Path(system.cfg.artifacts_dir)
    run_id = getattr(system, "run_id", "run")
    outdir = artifacts / "labels" / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    oracle = getattr(system, "_hyperperm_oracle", {"items":{}})
    overlays = getattr(system, "_overlay_registry", {})

    labels_count = 0
    # Label each datum
    for d in data:
        did = d.get("id") or _sha(json.dumps(d, sort_keys=True))
        atoms = ["×","÷","%","~"]  # core atom set by default
        seq = d.get("sequence") or [atoms[(i+hash(did))%len(atoms)] for i in range(3)]
        channel = d.get("channel","analysis_summary")
        # update oracle: track unique signatures (sequence+channel)
        key = "OP_CORE"
        orec = oracle.setdefault(key, {"atom_set": atoms, "orders": [], "channels": set(), "sigs": set(), "locked": False})
        sig = _sha("::".join(seq)+"||"+channel)
        if sig not in orec["sigs"]:
            orec["sigs"].add(sig)
            orec["orders"].append({"sequence":seq, "channel":channel, "sig":sig})
            orec["channels"].add(channel)

        # Write label card
        label = {
            "datum_id": did,
            "atoms": atoms,
            "channels": sorted(list(orec["channels"])),
            "orders": orec["orders"][-1:],  # last order seen for example
            "lock_threshold": 8,
            "locked": False,
            "gate": {"parity_even": True, "phi_before": 1.0, "phi_after": 0.9, "accepted": True},
            "receipts": [],
            "merkle_root": None,
            "annotations": {"semantics": d.get("text","")[:120]}
        }
        (outdir / f"{did}.json").write_text(json.dumps(label, indent=2))
        labels_count += 1

    # Lock rule: ≥8 unique channels/signatures
    key = "OP_CORE"
    if key in oracle:
        locked = len(oracle[key]["sigs"]) >= 8 and len(oracle[key]["channels"]) >= 8
        oracle[key]["locked"] = bool(locked)
        # Promote overlays if locked
        overlays_promoted = []
        if locked:
            for g in list(overlays.keys()):
                if overlays[g]["status"] == "PENDING":
                    overlays[g]["status"] = "LOCKED"
                    overlays_promoted.append(g)
        else:
            overlays_promoted = []
    else:
        overlays_promoted = []

    # Save registries back
    system._hyperperm_oracle = oracle
    system._overlay_registry = overlays

    system.emit("ms2", {"labels_count":labels_count, "overlays_promoted": overlays_promoted, "locked": oracle.get(key,{}).get("locked", False)})
    return {"labels_count": labels_count, "overlays_promoted": overlays_promoted, "locked": oracle.get(key,{}).get("locked", False)}

@slice_op("ms3_governance_manager", category="MS3", lane=3, parity="E", requires=["dwm_charter"], provides=["witnesses","quorum","seal"])
def ms3_governance_manager(system, payload, state):
    artifacts = Path(system.cfg.artifacts_dir)
    # Collect rung receipts if any
    receipts = []
    rdir = artifacts / "receipts"
    if rdir.exists():
        for p in sorted(rdir.glob("rung_*.json")):
            try:
                receipts.append(json.loads(p.read_text()))
            except Exception:
                pass

    # Programmatic witnesses
    witnesses = []
    yes = 0.0; no = 0.0; channels = set()
    for r in receipts:
        w_par = {"id":"wit:parity","vote":"YES" if r.get("parity_even") else "NO","weight":1.0}
        w_phi = {"id":"wit:deltaPhi","vote":"YES" if r.get("phi_after",1e9) <= r.get("phi_before",-1e9) else "NO","weight":1.0}
        witnesses += [w_par, w_phi]
    for w in witnesses:
        if w["vote"]=="YES": yes += w["weight"]
        else: no += w["weight"]

    # Diversity from MS2 oracle
    oracle = getattr(system, "_hyperperm_oracle", {})
    distinct_channels = 0
    for val in oracle.values():
        ch = val.get("channels", [])
        distinct_channels = max(distinct_channels, len(ch) if isinstance(ch, list) else len(ch))

    policy = payload.get("policy", {"type":"diversity+majority","d":4})
    decision = "ACCEPT" if (yes>no and distinct_channels>=policy.get("d",4)) else "REVIEW"

    # Seal over receipts + oracle snapshot
    leaves = []
    for r in receipts:
        leaves.append(hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest())
    leaves.append(hashlib.sha256(json.dumps(oracle, sort_keys=True, default=list).encode()).hexdigest())
    root = hashlib.sha256("".join(sorted(leaves)).encode()).hexdigest() if leaves else None

    # Save seal
    seal_path = None
    if root:
        sdir = artifacts / "seals"
        sdir.mkdir(parents=True, exist_ok=True)
        seal = {"merkle_root": root, "policy": policy, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        seal_path = str(sdir / f"seal_{root[:8]}.json")
        Path(seal_path).write_text(json.dumps(seal, indent=2))

    system.emit("ms3", {"decision":decision, "witnesses_count": len(witnesses), "diversity": distinct_channels, "seal_path": seal_path})
    return {"decision":decision, "witnesses_count": len(witnesses), "diversity": distinct_channels, "seal_path": seal_path}
