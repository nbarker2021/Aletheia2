import json, hashlib, hmac, base64, time, os, threading

_lock = threading.Lock()

def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",",":"))

def sha256_hex(b):
    if isinstance(b, str):
        b = b.encode("utf-8")
    return hashlib.sha256(b).hexdigest()

def hmac_sig(key_b64, msg):
    key = base64.b64decode(key_b64.encode("ascii"))
    mac = hmac.new(key, msg.encode("utf-8"), digestmod=hashlib.sha256).digest()
    return "H:" + base64.b64encode(mac).decode("ascii")

def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def append_jsonl(path, obj):
    with _lock:
        with open(path,"a") as f:
            f.write(json.dumps(obj, separators=(",",":"))+"\n")

def last_hash(path):
    try:
        with open(path,"rb") as f:
            lines = f.read().splitlines()
        if not lines: return None
        last = json.loads(lines[-1].decode("utf-8"))
        return last.get("hash")
    except FileNotFoundError:
        return None

def master_append(root, record, hmac_key):
    mpath = os.path.join(root,"data","ledgers","master.jsonl")
    prev = last_hash(mpath)
    record["ts"] = now_iso()
    record["prev"] = prev
    payload = canon(record)
    digest = sha256_hex(payload)
    record["hash"] = "h:"+digest
    record["sig"] = hmac_sig(hmac_key, payload)
    append_jsonl(mpath, record)
    return record

def verify_master(root, hmac_key):
    mpath = os.path.join(root,"data","ledgers","master.jsonl")
    try:
        with open(mpath,"rb") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return {"ok":True,"count":0}
    prev = None
    count = 0
    for raw in lines:
        rec = json.loads(raw.decode("utf-8"))
        payload = canon({k:v for k,v in rec.items() if k not in ("hash","sig")})
        if rec.get("prev") != prev:
            return {"ok":False,"at":count,"error":"prev_mismatch"}
        if rec.get("hash") != "h:"+sha256_hex(payload):
            return {"ok":False,"at":count,"error":"hash_mismatch"}
        if rec.get("sig") != hmac_sig(hmac_key, payload):
            return {"ok":False,"at":count,"error":"sig_mismatch"}
        prev = rec.get("hash")
        count += 1
    return {"ok":True,"count":count}
