
import json, pathlib, sys, datetime

def migrate_v1_to_v2(v1: dict) -> dict:
    # Heuristic mapping — adapt field names as needed.
    now = datetime.datetime.utcnow().isoformat() + "Z"
    e8 = v1.get("e8", {})
    axes = e8.get("axes", v1.get("axes", {}))
    return {
        "schema_version": "2.0",
        "snap_id": v1.get("id") or v1.get("snap_id","unknown"),
        "created_at": v1.get("created_at") or now,
        "e8": {
            "version": "0.1",
            "coords": e8.get("coords",[1,0,0,0,0,0,0,0]),
            "root_loc": e8.get("root_loc", {"nearest_roots":[{"index":0,"inner_product":1.0}],"reflections":[],"adjacency_rule":"inner_product_eq_1"}),
            "axes": axes,
            "bridge_node": e8.get("bridge_node", []),
            "notes": e8.get("notes","")
        },
        "axes": axes,
        "kind": v1.get("kind","Run"),
        "parent_id": v1.get("parent_id"),
        "children": v1.get("children", []),
        "hashes": v1.get("hashes", {}),
        "payload": v1.get("payload", {"format":"json","location":"unknown","size_bytes":0,"secure":True}),
        "provenance": v1.get("provenance", {"code_version":"unknown","modules":[],"env":{}}),
        "security": v1.get("security", {"signed": False, "allow_pickle": False}),
        "metrics": v1.get("metrics", {}),
        "notes": v1.get("notes","")
    }

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python migrate_snap_v1_to_v2.py <in.json> <out.json>")
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    v1 = json.loads(pathlib.Path(src).read_text())
    v2 = migrate_v1_to_v2(v1)
    pathlib.Path(dst).write_text(json.dumps(v2, indent=2))
    print("Wrote", dst)
