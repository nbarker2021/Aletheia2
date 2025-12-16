
from typing import Dict, Any, List
import json, os, time, hashlib, uuid

def write_json(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def merkle_root(items: List[str]) -> str:
    if not items:
        return hashlib.sha256(b"").hexdigest()
    nodes = [hashlib.sha256(x.encode("utf-8")).hexdigest() for x in items]
    while len(nodes) > 1:
        nxt = []
        for i in range(0, len(nodes), 2):
            a = nodes[i]
            b = nodes[i+1] if i+1 < len(nodes) else a
            nxt.append(hashlib.sha256((a+b).encode("utf-8")).hexdigest())
        nodes = nxt
    return nodes[0]

def make_startup_packet(run_id: str, demand: Dict[str,Any], receipts: List[Dict[str,Any]], extras: Dict[str,Any], outdir: str) -> str:
    pkt = {
        "run_id": run_id,
        "ts": time.time(),
        "demand": demand,
        "receipts": receipts,
        "extras": extras
    }
    path = os.path.join(outdir, f"startup_{run_id}.json")
    write_json(path, pkt)
    return path

def make_pause_token(run_id: str, pointer: str, outdir: str) -> str:
    tok = {"run_id": run_id, "pointer": pointer, "ts": time.time()}
    path = os.path.join(outdir, f"pause_{run_id}.json")
    write_json(path, tok)
    return path

def make_resume_token(run_id: str, pointer: str, outdir: str) -> str:
    tok = {"run_id": run_id, "pointer": pointer, "ts": time.time()}
    path = os.path.join(outdir, f"resume_{run_id}.json")
    write_json(path, tok)
    return path

def seal_envelope(run_id: str, receipts: List[Dict[str,Any]], config: Dict[str,Any], witnesses: List[Dict[str,Any]], outdir: str) -> str:
    leaves = []
    for r in receipts:
        leaves.append(json.dumps(r, sort_keys=True))
    leaves.append(json.dumps(config, sort_keys=True))
    for w in witnesses:
        leaves.append(json.dumps(w, sort_keys=True))
    root = merkle_root(leaves)
    env = {"run_id": run_id, "merkle_root": root, "count": len(leaves), "ts": time.time()}
    path = os.path.join(outdir, f"seal_{run_id}.json")
    write_json(path, env)
    return path, root
