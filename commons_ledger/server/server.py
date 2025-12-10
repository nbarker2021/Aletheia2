from http.server import HTTPServer, BaseHTTPRequestHandler
import os, json, urllib.parse, traceback

from .util import master_append, verify_master
from .sidecar_kernel import attach_sidecar
from .features_schema import validate_features
from .mint import compute_mint_splits
from .wallet import Wallet, Vesting, Treasury
from .adapters_loader import extract_auto
from .ca_engine import TickEngine, Council, HarmFloor, RoleScheduler
from .governance import OMPS, JokerBank, EnergyGate, ring_checkpoint
from .markets import Markets
from .tools_registry import ToolsRegistry, BountyBoard

ROOT = os.path.dirname(os.path.dirname(__file__))
with open(os.path.join(ROOT,"data","config.json")) as f:
    CFG = json.load(f)
HMAC_KEY = CFG["hmac_key"]

SIDE = attach_sidecar(ROOT)
WALLET = Wallet(ROOT, CFG["coins"], CFG["payout_wallet"])
VEST = Vesting(ROOT)
TREA = Treasury(ROOT)
TICK = TickEngine(ROOT, CFG["rng_scope"])
COUNCIL = Council(ROOT)
HARM = HarmFloor(ROOT)
ROLE = RoleScheduler()
OMPSCHK = OMPS()
JOKERS = JokerBank()
EGATE = EnergyGate()
MKT = Markets(ROOT)
TOOLS = ToolsRegistry(ROOT)
BOUNTY = BountyBoard(ROOT)

def json_body(handler):
    l = int(handler.headers.get('Content-Length','0'))
    if l==0: return {}
    raw = handler.rfile.read(l)
    try: return json.loads(raw.decode("utf-8"))
    except: return {}

class App(BaseHTTPRequestHandler):
    def _send(self, code, obj, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if isinstance(obj, (dict,list)):
            self.wfile.write(json.dumps(obj).encode("utf-8"))
        elif isinstance(obj, (bytes,bytearray)):
            self.wfile.write(obj)
        else:
            self.wfile.write(str(obj).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        try:
            if self.path.startswith("/") and (self.path == "/" or self.path.startswith("/web/")):
                path = self.path[1:] if self.path != "/" else "web/index.html"
                full = os.path.join(ROOT, path)
                if os.path.isdir(full):
                    full = os.path.join(full, "index.html")
                if not os.path.exists(full):
                    self._send(404, {"error":"not found"}); return
                with open(full,"rb") as f: data = f.read()
                ctype = "text/html"
                if full.endswith(".js"): ctype = "application/javascript"
                if full.endswith(".css"): ctype = "text/css"
                self._send(200, data, ctype); return

            if self.path == "/master/verify":
                self._send(200, verify_master(ROOT, HMAC_KEY)); return

            if self.path == "/wallet/balance":
                with open(os.path.join(ROOT,"data","wallet.json")) as f:
                    bal = json.load(f)
                self._send(200, bal); return

            self._send(404, {"error":"not found"})
        except Exception as e:
            self._send(500, {"error":str(e),"trace":traceback.format_exc()})

    def do_POST(self):
        try:
            if self.path == "/assets/upload":
                body = json_body(self)
                name = body.get("name","unnamed.txt")
                content = body.get("content","")
                apath = os.path.join(ROOT,"data","assets",name)
                with open(apath,"w",encoding="utf-8") as f: f.write(content)
                rec = {"type":"upload","name":name}
                master_append(ROOT, rec, HMAC_KEY)
                self._send(200, {"ok":True,"name":name}); return

            if self.path == "/features/extract_auto":
                body = json_body(self)
                source = body.get("source","")
                merged = extract_auto(ROOT, source)
                self._send(200, merged); return

            if self.path == "/features/validate":
                body = json_body(self)
                ok, errs = validate_features(body)
                self._send(200, {"ok":ok,"errors":errs}); return

            if self.path == "/mint/score":
                body = json_body(self)
                self._send(200, compute_mint_splits(body, CFG)); return

            if self.path == "/mint/mint_now":
                body = json_body(self)
                res = compute_mint_splits(body, CFG)
                if not res.get("ok"):
                    self._send(400, res); return
                splits = res["splits"]
                WALLET.credit(splits, HMAC_KEY)
                rec = {"type":"mint_now","splits":splits}
                master_append(ROOT, rec, HMAC_KEY)
                self._send(200, {"ok":True,"credited":splits}); return

            if self.path == "/tick":
                rec = TICK.tick(HMAC_KEY)
                self._send(200, rec); return

            if self.path == "/council/draw":
                body = json_body(self)
                pool = body.get("pool",["socA.agent1","socA.agent2","socB.agent1","socC.agent1","socU.agent1","socU.agent2","socA.agent3","socB.agent4"])
                rng_proof = str(TICK.idx or 1)
                rec = COUNCIL.draw(pool, 8, rng_proof, HMAC_KEY)
                self._send(200, rec); return

            if self.path == "/council/vote":
                body = json_body(self)
                proposal = body.get("proposal","n/a")
                picked = body.get("picked",[])
                rec = COUNCIL.vote(proposal, picked, HMAC_KEY)
                self._send(200, rec); return

            if self.path == "/harm/eval":
                body = json_body(self)
                rec = HARM.evaluate(body.get("net",{}), HMAC_KEY)
                self._send(200, {"ok": rec is None, "gate": rec}); return

            if self.path == "/markets/shock":
                body = json_body(self)
                rec = MKT.shock(body.get("soc","U"), body.get("shock",{}), HMAC_KEY)
                self._send(200, rec); return

            if self.path == "/markets/stabilize":
                body = json_body(self)
                rec = MKT.stabilize(body.get("soc","U"), body.get("k","EMA_4"), HMAC_KEY)
                self._send(200, rec); return

            if self.path == "/markets/understanding_spend":
                body = json_body(self)
                rec = MKT.understanding_spend(body.get("soc","A"), body.get("agent","room.A.3"), body.get("topic","n/a"), body.get("chits",1), HMAC_KEY)
                self._send(200, rec); return

            if self.path == "/tools/register":
                body = json_body(self)
                rec = TOOLS.register(body.get("tool","n/a"), body.get("shares",{}), HMAC_KEY)
                self._send(200, rec); return

            if self.path == "/bounty/open":
                body = json_body(self)
                rec = BOUNTY.open(body.get("id","B:x"), body.get("title","n/a"), body.get("scorecard",{}), HMAC_KEY)
                self._send(200, rec); return

            if self.path == "/bounty/claim":
                body = json_body(self)
                rec = BOUNTY.claim(body.get("id","B:x"), body.get("by","u:x"), body.get("evidence","n/a"), HMAC_KEY)
                self._send(200, rec); return

            if self.path == "/bounty/payout":
                body = json_body(self)
                rec = BOUNTY.payout(body.get("id","B:x"), body.get("to","u:x"), body.get("amount",{}), HMAC_KEY)
                self._send(200, rec); return

            if self.path == "/master/verify":
                self._send(200, verify_master(ROOT, HMAC_KEY)); return

            self._send(404, {"error":"not found"})
        except Exception as e:
            self._send(500, {"error":str(e), "trace":traceback.format_exc()})

def run(addr="127.0.0.1", port=8766):
    httpd = HTTPServer((addr, port), App)
    print(f"Serving on http://{addr}:{port}/web/")
    httpd.serve_forever()

if __name__ == "__main__":
    run()
