import os, json, time, threading, importlib.util, hashlib

class SimpleSidecar:
    def __init__(self, mem_bytes=10_000_000):
        self.cache = {}
        self.mem_bytes = mem_bytes
        self.lock = threading.Lock()

    def compute(self, payload_key, scope="default", channel=3, compute_fn=None):
        key = hashlib.sha256((scope+"|"+str(channel)+"|"+json.dumps(payload_key,sort_keys=True)).encode()).hexdigest()
        with self.lock:
            if key in self.cache:
                res = self.cache[key]
                return res["result"], {"hits":1,"misses":0,"cost":0.0}, res["rid"]
        start = time.time()
        result = compute_fn() if compute_fn else None
        dur = time.time()-start
        rid = "R:"+key[:16]
        with self.lock:
            self.cache[key] = {"result":result,"rid":rid,"ts":time.time(),"E":dur}
        return result, {"hits":0,"misses":1,"cost":dur}, rid

def load_speedlight_plus(path):
    try:
        spec = importlib.util.spec_from_file_location("speedlight_sidecar_plus", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = getattr(mod, "SpeedLightPlus", None)
        if cls is None:
            return None
        return cls(mem_bytes=256_000_000, disk_dir=None, ledger_path=None)
    except Exception:
        return None

def attach_sidecar(root):
    plus_path = os.path.join(root, "speedlight", "speedlight_sidecar_plus.py")
    sl = load_speedlight_plus(plus_path)
    if sl is None:
        sl = SimpleSidecar()
    return sl
