
import os, json, time, hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from cqe_sidecar_mini import CQESidecarMini

BASE = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(BASE, "data"); WEB = os.path.join(BASE, "web"); CONFIG = os.path.join(BASE, "config")
os.makedirs(DATA, exist_ok=True)

LEDGER_PATH = os.path.join(DATA, "ledger.jsonl")
INVENTORY_PATH = os.path.join(DATA, "inventory.json")
ANCHOR_PATH = os.path.join(DATA, "ledger_anchor.json")
TOOLS_PATH = os.path.join(DATA, "tools.json")
TOOLCHAINS_PATH = os.path.join(DATA, "toolchains.json")
BALANCES_PATH = os.path.join(DATA, "balances.json")
TOOLTOKENS_PATH = os.path.join(DATA, "tooltokens.json")

def load_json(p, d): 
    try: return json.load(open(p,"r",encoding="utf-8"))
    except: return d
def save_json(p, o): json.dump(o, open(p,"w",encoding="utf-8"), indent=2)

POLICY = {"policy_version":"mint-v1.0","pricing":{"lab_run_base_merit":0.05,"suite_run_multiplier":1.2,"tool_register_merit":0.1},"conversion":{"MERIT_to_domain":{"THETA":0.9,"DAVINCI":0.9,"GAIA":0.9,"INFRA":0.9,"CARETAKER":0.9,"VANGOGH":0.9,"MYTHOS":0.9,"ORBIT":0.9,"AEGIS":0.9},"domain_to_MERIT_allowed":False}}
sid = CQESidecarMini(disk_dir=os.path.join(DATA,".cas"), ledger_path=os.path.join(DATA,".sidecar_ledger.jsonl"), policy=POLICY)

def _load_bal(): return load_json(BALANCES_PATH, {})
def _save_bal(b): save_json(BALANCES_PATH, b)
def credit(actor, coin, amt): b=_load_bal(); a=b.setdefault(actor,{}); a[coin]=float(a.get(coin,0))+float(amt); _save_bal(b); return a
def debit(actor, coin, amt): b=_load_bal(); a=b.setdefault(actor,{}); 
if float(a.get(coin,0))<float(amt): raise ValueError("insufficient"); a[coin]=float(a.get(coin,0))-float(amt); _save_bal(b); return a

def mrkl(lines):
    import hashlib
    if not lines: return None
    level = [hashlib.sha256((l if isinstance(l,str) else json.dumps(l)).encode()).digest() for l in lines]
    while len(level)>1:
        nxt=[]
        for i in range(0,len(level),2):
            L=level[i]; R=level[i+1] if i+1<len(level) else L
            nxt.append(hashlib.sha256(L+R).digest())
        level=nxt
    return level[0].hex()


# --- Universal Kernel: always-on 'Speedlight split' for every action ---
KERNEL_LOG = []  # small in-mem tail for UI

def kernel_split(action: str, meta: dict):
    # Split into channels 3/6/9 using the sidecar compute with a zero-cost baseline
    res = []
    for ch in (3,6,9):
        def compute_fn():
            return {"action": action, "meta": meta, "channel": ch, "_base_cost": 0.0}
        r, c, rid = sid.compute({"action": action, "meta": meta}, scope="kernel", channel=ch, tags=["speedlight","beam", action], compute_fn=compute_fn)
        res.append({"ch": ch, "rid": rid})
    KERNEL_LOG.append({"ts": time.time(), "action": action, "meta": meta, "rids": res[-3:]})
    if len(KERNEL_LOG) > 200:  # keep a short tail
        del KERNEL_LOG[:len(KERNEL_LOG)-200]
    return res


def adapters_status():
    # reflect what's available in the sidecar adapters
    try:
        from cqe_sidecar_mini import adapters
        return {"geotokenizer": callable(getattr(adapters,"geotokenize",None)),
                "mdhg": callable(getattr(adapters,"mdhg_signal",None)),
                "moonshine": callable(getattr(adapters,"moonshine_crosshit",None))}
    except Exception:
        return {"geotokenizer": False, "mdhg": False, "moonshine": False}

class H(BaseHTTPRequestHandler):
    def read_json(self):
        ln=int(self.headers.get("Content-Length","0") or "0")
        if ln<=0: return {}
        raw=self.rfile.read(ln).decode("utf-8")
        try: return json.loads(raw)
        except: return {}
    def write(self, obj, code=200):
        body=json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type","application/json")
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Access-Control-Allow-Headers","*")
        self.end_headers(); self.wfile.write(body)
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Access-Control-Allow-Headers","*")
        self.send_header("Access-Control-Allow-Methods","POST, OPTIONS")
        self.end_headers()

    def do_POST(self):
        b=self.read_json(); p=self.path
        if p=="/health": return self.write({"ok":True,"ts":time.time()})
        if p=="/adapters/status": return self.write({"ok":True,"adapters": adapters_status()})
        if p=="/adapters/run":
            name=b.get("name"); inp=b.get("input")
            from cqe_sidecar_mini import adapters
            fn=getattr(adapters, name, None)
            if not callable(fn): return self.write({"ok":False,"error":"adapter not found"})
            out=fn(inp) if name=="geotokenize" else fn(inp if isinstance(inp, dict) else {})
            return self.write({"ok":True,"output": out})
        if p=="/kernel/report": return self.write({"ok":True,"tail": KERNEL_LOG[-30:]})

if p=="/sim/hashworld/run":
    seed = str(b.get("seed","world"))
    ticks = int(b.get("ticks",16))
    cities = int(b.get("cities",4))
    from cqe_sidecar_mini import adapters as _ad
    out = _ad.agrm_hashcity(seed=seed, ticks=ticks, cities=cities)
    open(LEDGER_PATH,"a",encoding="utf-8").write(json.dumps({"type":"hashworld","seed":seed,"ticks":ticks,"cities":cities,"ts":time.time()})+"\n")
    return self.write({"ok":True,"result": out})

        if p=="/upload":
            kernel_split("upload", {"actor": b.get("actor_id","anon"), "name": b.get("name")})
            name=b.get("name","note.txt"); text=b.get("text","")
            up_dir=os.path.join(DATA,"uploads"); os.makedirs(up_dir,exist_ok=True)
            fp=os.path.join(up_dir,name)
            open(fp,"w",encoding="utf-8").write(text)
            inv=load_json(INVENTORY_PATH, {"papers":[],"drafts":[]})
            inv["papers"].append({"name":name,"path":fp}); save_json(INVENTORY_PATH,inv)
            open(LEDGER_PATH,"a",encoding="utf-8").write(json.dumps({"type":"upload","name":name,"ts":time.time()})+"\\n")
            return self.write({"ok":True,"name":name})

        if p=="/tiles/init":
            return self.write({"ok":True,"inventory": load_json(INVENTORY_PATH, {"papers":[],"drafts":[]})})

        if p=="/combine":
            kernel_split("combine", {"src":[b.get("p1"), b.get("p2")]})
            p1,p2=b.get("p1"),b.get("p2")
            inv=load_json(INVENTORY_PATH, {"papers":[],"drafts":[]})
            did=f"DF_{int(time.time()*1000)}"
            inv["drafts"].append({"draft_id":did,"sources":[p1,p2],
                "lab":{"passes":0,"fails":0},
                "evidence":{"endorsements":0,"lineage":0,"deployments":0,"quality":0,"delta_phi":0,"novelty":0,"safety":{"non_coercive":True,"non_weaponizable":True,"harm_reduction_pass":True}}})
            save_json(INVENTORY_PATH, inv)
            open(LEDGER_PATH,"a",encoding="utf-8").write(json.dumps({"type":"combine","draft_id":did,"sources":[p1,p2],"ts":time.time()})+"\\n")
            return self.write({"ok":True,"draft_id":did})

        if p=="/lab/record":
            kernel_split("lab_record", {"draft_id": b.get("draft_id"), "passed": bool(b.get("passed",False))})
            did=b.get("draft_id"); ok=bool(b.get("passed",False))
            inv=load_json(INVENTORY_PATH, {"papers":[],"drafts":[]})
            for d in inv["drafts"]:
                if d["draft_id"]==did:
                    d["lab"]["passes"]+=1 if ok else 0
                    d["lab"]["fails"]+=0 if ok else 1
                    save_json(INVENTORY_PATH,inv)
                    open(LEDGER_PATH,"a",encoding="utf-8").write(json.dumps({"type":"lab_mark","draft_id":did,"passed":ok,"ts":time.time()})+"\\n")
                    return self.write({"ok":True,"lab":d["lab"]})
            return self.write({"ok":False,"error":"draft not found"})

        if p=="/metrics/mint_preview":
            did=b.get("draft_id")
            inv=load_json(INVENTORY_PATH, {"papers":[],"drafts":[]})
            d=next((x for x in inv.get("drafts",[]) if x["draft_id"]==did), None)
            if not d: return self.write({"ok":False,"error":"draft not found"})
            sft=d["evidence"]["safety"]; fails=[]
            if not sft.get("non_coercive",False): fails.append("fails: non_coercive")
            if not sft.get("non_weaponizable",False): fails.append("fails: non_weaponizable")
            if not sft.get("harm_reduction_pass",False): fails.append("fails: harm_reduction_pass")
            return self.write({"ok":True,"policy_version":POLICY["policy_version"],
                "aegis_explanation":fails,"factors":d["evidence"],"lab":d["lab"],
                "domains":{"MERIT":1.0,"THETA":0.3,"DAVINCI":0.2}})

        if p=="/sidecar/run_tool":
            kernel_split("run_tool", {"func": b.get("func","main")})
            code=b.get("code",""); func=b.get("func","main"); args=b.get("args",{})
            try: out=sid.run_tool(code, func, args); return self.write({"ok":True, **out})
            except Exception as e: return self.write({"ok":False,"error":str(e)})

        if p=="/sidecar/run_pipeline":
            kernel_split("run_pipeline", {"stages": len(b.get("stages",[]))})
            stages=b.get("stages",[])
            try: out=sid.run_pipeline(stages); return self.write({"ok":True, **out})
            except Exception as e: return self.write({"ok":False,"error":str(e)})

        if p=="/sidecar/report":
            return self.write({"ok":True,"report": sid.report()})

        if p=="/pricing/quote":
            kind=b.get("kind","lab_run"); composed=bool(b.get("composed",False))
            price=POLICY["pricing"]["lab_run_base_merit"]
            if composed and kind=="lab_run": price*=POLICY["pricing"]["suite_run_multiplier"]
            return self.write({"ok":True,"MERIT":round(price,4),"fiat":2.5})

        if p=="/credits/balances":
            actor=b.get("actor_id"); bal=_load_bal(); 
            if actor: bal={actor:bal.get(actor,{})}
            return self.write({"ok":True,"balances":bal})

        if p=="/credits/convert":
            kernel_split("convert", {"actor": b.get("actor_id","anon"), "src": b.get("src"), "dst": b.get("dst")})
            actor=b.get("actor_id","anon"); src=b.get("src"); dst=b.get("dst"); amt=float(b.get("amount",0))
            conv=POLICY["conversion"]
            if src=="MERIT" and dst in conv["MERIT_to_domain"]:
                rate=float(conv["MERIT_to_domain"][dst])
                try: debit(actor,"MERIT",amt)
                except Exception: return self.write({"ok":False,"error":"insufficient MERIT"})
                credit(actor,dst,amt*rate)
                open(LEDGER_PATH,"a",encoding="utf-8").write(json.dumps({"type":"convert","actor":actor,"src":src,"dst":dst,"amount":amt,"rate":rate,"ts":time.time()})+"\\n")
                return self.write({"ok":True,"rate":rate})
            if src!="MERIT" and dst=="MERIT" and not conv.get("domain_to_MERIT_allowed",False):
                return self.write({"ok":False,"error":"Upward conversion disabled by policy"})
            return self.write({"ok":False,"error":"Unsupported conversion"})

        if p=="/tools/register":
            actor=b.get("actor_id","anon"); name=b.get("name"); code=b.get("code","")
            price=POLICY["pricing"]["tool_register_merit"]
            try: debit(actor,"MERIT",price)
            except Exception: return self.write({"ok":False,"error":"Insufficient MERIT","price":price})
            tools=load_json(TOOLS_PATH, []); tid=f"TL_{int(time.time()*1000)}"
            tools.append({"tool_id":tid,"name":name,"owner":actor,"code":code,"created_at":time.time()})
            save_json(TOOLS_PATH, tools)
            open(LEDGER_PATH,"a",encoding="utf-8").write(json.dumps({"type":"tool_register","actor":actor,"tool_id":tid,"ts":time.time()})+"\\n")
            return self.write({"ok":True,"tool_id":tid,"debited_merit":price})

        if p=="/tools/list":
            return self.write({"ok":True,"tools": load_json(TOOLS_PATH, [])})

        if p=="/tools/toolchains/create":
            actor=b.get("actor_id","anon"); name=b.get("name","suite"); steps=b.get("steps",[])
            tcs=load_json(TOOLCHAINS_PATH, []); tcid=f"TC_{int(time.time()*1000)}"
            tcs.append({"toolchain_id":tcid,"name":name,"owner":actor,"steps":steps,"created_at":time.time()})
            save_json(TOOLCHAINS_PATH, tcs)
            open(LEDGER_PATH,"a",encoding="utf-8").write(json.dumps({"type":"toolchain_create","actor":actor,"toolchain_id":tcid,"ts":time.time()})+"\\n")
            return self.write({"ok":True,"toolchain_id":tcid})

        if p=="/tools/toolchains/run":
            kernel_split("toolchain_run", {"tcid": b.get("toolchain_id")})
            actor=b.get("actor_id","anon"); tcid=b.get("toolchain_id"); data=b.get("input","")
            tcs=load_json(TOOLCHAINS_PATH, []); tools=load_json(TOOLS_PATH, [])
            tc=next((t for t in tcs if t["toolchain_id"]==tcid), None)
            if not tc: return self.write({"ok":False,"error":"toolchain not found"})
            price=POLICY["pricing"]["lab_run_base_merit"]*POLICY["pricing"]["suite_run_multiplier"]
            try: debit(actor,"MERIT",price)
            except Exception: return self.write({"ok":False,"error":"Insufficient MERIT for suite run","price":price})
            stages=[]; out=data
            for st in tc.get("steps",[]):
                tool = next((t for t in tools if t["tool_id"]==st.get("tool_id")), None)
                if not tool: continue
                stages.append({"code": tool.get("code",""), "func": st.get("func","main"), "args": st.get("args",{})})
            pipe = sid.run_pipeline(stages)
            rid = pipe.get("rid") or f"LABSUITE_{int(time.time()*1000)}"
            open(LEDGER_PATH,"a",encoding="utf-8").write(json.dumps({"type":"lab_suite","actor":actor,"toolchain_id":tcid,"rid":rid,"ts":time.time()})+"\\n")
            return self.write({"ok":True,"rid":rid,"result":pipe.get("output"),"debited_merit":price})

        if p=="/ledger/anchor":
            kernel_split("anchor", {})
            lines=open(LEDGER_PATH,"r",encoding="utf-8").read().splitlines() if os.path.exists(LEDGER_PATH) else []
            root=mrkl(lines) if lines else None; save_json(ANCHOR_PATH, {"ts":time.time(),"root":root,"count":len(lines)})
            return self.write({"ok":True,"root":root,"count":len(lines)})
        if p=="/ledger/get_root":
            a=load_json(ANCHOR_PATH,{}); return self.write({"ok":bool(a), **a})
        if p=="/ledger/verify_root":
            a=load_json(ANCHOR_PATH,{}); 
            if not a: return self.write({"ok":False,"error":"no anchor"})
            lines=open(LEDGER_PATH,"r",encoding="utf-8").read().splitlines() if os.path.exists(LEDGER_PATH) else []
            comp=mrkl(lines) if lines else None
            return self.write({"ok": comp==a.get("root"), "stored":a.get("root"), "computed":comp, "count":len(lines)})

        return self.write({"ok":False,"error":"unknown path"}, 404)

def run(host="0.0.0.0", port=8787):
    httpd = HTTPServer((host, port), H); print(f"Server http://{host}:{port}"); httpd.serve_forever()

if __name__=="__main__":
    run()
