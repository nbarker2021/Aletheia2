
# cqe_sidecar_mini.sidecar — stdlib-only "mini-CQE" sidecar with adapters
import os, json, time, hashlib, collections, ast, types

def _sha(s: bytes) -> str:
    return hashlib.sha256(s).hexdigest()

class SafeExecError(Exception):
    pass

class SafeSandbox:
    """Strict safe exec (no imports, no I/O, no dunders)."""
    import ast as _ast
    ALLOWED_BUILTINS = {
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len, "range": range,
        "enumerate": enumerate, "sorted": sorted, "map": map, "filter": filter,
        "all": all, "any": any, "zip": zip, "reversed": reversed, "round": round
    }
    ALLOWED_NODES = (
        _ast.Module, _ast.FunctionDef, _ast.arguments, _ast.arg, _ast.Return, _ast.Assign, _ast.AnnAssign,
        _ast.Expr, _ast.Call, _ast.Name, _ast.Load, _ast.Store, _ast.Constant, _ast.List, _ast.Tuple, _ast.Dict,
        _ast.BinOp, _ast.UnaryOp, _ast.operator, _ast.unaryop, _ast.BoolOp, _ast.Compare, _ast.comprehension,
        _ast.ListComp, _ast.DictComp, _ast.GeneratorExp, _ast.If, _ast.For, _ast.While, _ast.Break, _ast.Continue,
        _ast.IfExp, _ast.Subscript, _ast.Slice, _ast.Attribute
    )
    FORBIDDEN_NAMES = {"__import__", "__loader__", "__spec__", "__builtins__", "__file__", "__name__", "__package__"}

    def _check_ast(self, node):
        import ast as _ast
        for child in _ast.walk(node):
            if not isinstance(child, self.ALLOWED_NODES):
                raise SafeExecError(f"Disallowed AST node: {type(child).__name__}")
            if isinstance(child, (_ast.Import, _ast.ImportFrom)):
                raise SafeExecError("Imports not allowed")
            if isinstance(child, _ast.Attribute):
                if str(child.attr).startswith("_"):
                    raise SafeExecError("Private/dunder attribute access forbidden")
            if isinstance(child, _ast.Name):
                if child.id in self.FORBIDDEN_NAMES:
                    raise SafeExecError(f"Forbidden name: {child.id}")

    def compile(self, source: str):
        node = compile(source, "<ast>", "exec", dont_inherit=True, optimize=1)
        # reparse to validate (safer across py versions)
        import ast as _ast
        tree = _ast.parse(source, mode="exec")
        self._check_ast(tree)
        env = {"__builtins__": self.ALLOWED_BUILTINS}
        return node, env

    def run(self, source: str, func_name: str, args: dict, cqe_ns):
        code, env = self.compile(source)
        env["cqe"] = cqe_ns
        exec(code, env, env)
        if func_name not in env or not callable(env[func_name]):
            raise SafeExecError(f"Function {func_name} not found")
        return env[func_name](**(args or {}))

class CAS:
    def __init__(self, mem_limit=128*1024*1024, disk_dir=None):
        self.mem = {}
        self.order = collections.deque()
        self.size = 0
        self.mem_limit = mem_limit
        self.disk_dir = disk_dir
        if disk_dir: os.makedirs(disk_dir, exist_ok=True)

    def put(self, key: str, obj: dict, persist=True):
        payload = json.dumps(obj, sort_keys=True).encode("utf-8")
        sz = len(payload)
        while self.size + sz > self.mem_limit and self.order:
            oldest = self.order.popleft()
            blob = self.mem.pop(oldest, None)
            if blob: self.size -= len(blob)
        self.mem[key] = payload
        self.order.append(key)
        self.size += sz
        if persist and self.disk_dir:
            with open(os.path.join(self.disk_dir, key + ".json"), "wb") as f:
                f.write(payload)

    def get(self, key: str):
        if key in self.mem:
            return json.loads(self.mem[key].decode("utf-8"))
        if self.disk_dir:
            p = os.path.join(self.disk_dir, key + ".json")
            if os.path.exists(p):
                with open(p, "rb") as f: payload = f.read()
                self.mem[key] = payload; self.order.append(key); self.size += len(payload)
                return json.loads(payload.decode("utf-8"))
        return None

# Adapters namespace (kept pure-python, stdlib-only)
from . import adapters as _adapters

class CQESidecarMini:
    def __init__(self, disk_dir=".sidecar/cas", ledger_path=".sidecar/ledger.jsonl", policy=None):
        self.cas = CAS(disk_dir=disk_dir)
        self.ledger_path = ledger_path
        self.policy = policy or {"policy_version":"mint-v1.0","delta_phi_threshold":0.0}
        os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
        self.hits = 0; self.misses = 0; self.start = time.time()

    def _append(self, entry: dict):
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True)+"\n")

    def report(self):
        return {"uptime_s": round(time.time()-self.start,2), "cache_hits": self.hits, "cache_misses": self.misses}

    @property
    def cqe_ns(self):
        # small helper namespace + adapters, exposed to sandboxed tools
        def tokenize(text):
            return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in (text or "")).split() if t]
        def vectorize(tokens):
            d = {}; 
            for t in tokens: d[t] = d.get(t,0)+1
            return d
        def dot(a,b):
            return sum(a.get(k,0)*b.get(k,0) for k in set(a)|set(b))
        return types.SimpleNamespace(tokenize=tokenize, vectorize=vectorize, dot=dot, adapters=_adapters)

    def compute(self, payload: dict, scope="default", channel=0, tags=None, compute_fn=None):
        k = _sha(json.dumps({"payload":payload, "scope":scope, "channel":channel, "tags":sorted(tags or [])}, sort_keys=True).encode("utf-8"))
        cached = self.cas.get(k)
        t0 = time.perf_counter()
        if cached is not None:
            self.hits += 1
            cost = 0.0
            res = cached["result"]; rid = cached["receipt_id"]
            self._append({"type":"compute_hit","key":k,"scope":scope,"ch":channel,"tags":tags,"ts":time.time()})
            return res, cost, rid
        self.misses += 1
        if not callable(compute_fn):
            raise ValueError("compute_fn required on miss")
        res = compute_fn()
        t1 = time.perf_counter(); cost = t1 - t0
        base = res.get("_base_cost", cost) if isinstance(res, dict) else cost
        delta_phi = float(base) - float(cost)
        accepted = (delta_phi >= float(self.policy.get("delta_phi_threshold", 0.0)))
        rid = _sha(f"{k}:{time.time()}".encode("utf-8"))[:16]
        obj = {"result": res, "cost": cost, "delta_phi": delta_phi, "accepted": accepted, "receipt_id": rid, "policy_version": self.policy.get("policy_version","mint-v1.0")}
        self.cas.put(k, obj, persist=True)
        self._append({"type":"compute_store","key":k,"rid":rid,"scope":scope,"ch":channel,"tags":tags,"cost":cost,"delta_phi":delta_phi,"accepted":accepted,"ts":time.time()})
        return res, cost, rid

    def run_tool(self, source_code: str, func_name: str, args: dict):
        sand = SafeSandbox()
        t0 = time.perf_counter()
        out = sand.run(source_code, func_name, args or {}, self.cqe_ns)
        t1 = time.perf_counter(); cost = t1 - t0
        rid = _sha(f"{source_code}:{func_name}:{json.dumps(args, sort_keys=True)}:{t1}".encode("utf-8"))[:16]
        self._append({"type":"tool_run","func":func_name,"rid":rid,"cost":cost,"ts":time.time()})
        return {"output": out, "_base_cost": cost, "rid": rid}

    def run_pipeline(self, stages):
        acc = None; receipts = []
        for i, st in enumerate(stages or []):
            code = st.get("code",""); func = st.get("func","main"); args = st.get("args",{})
            if acc is not None and isinstance(args, dict):
                args = dict(args); args.setdefault("input", acc)
            r = self.run_tool(code, func, args)
            acc = r.get("output"); receipts.append(r.get("rid"))
        rid = _sha(("|".join(receipts)).encode("utf-8"))[:16] if receipts else None
        self._append({"type":"pipeline","rid":rid,"stages":len(stages or []),"ts":time.time()})
        return {"output": acc, "stages": len(stages or []), "rid": rid}
