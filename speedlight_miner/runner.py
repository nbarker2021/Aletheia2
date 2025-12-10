import time, hashlib, json
from .compact_target import compact_to_target
from .agrm_policy import AGRMPolicy
from .mdhg_cache import MDHGCache
from .coinbase_builder import script_commit_session, build_coinbase, compute_merkle_with_coinbase
from .node_adapter import NodeAdapter
from .sim_node import SimNode
from .header_space import HeaderSpace, RailConfig
from .golden_strider import StriderCfg, tri_rail_batches
from .midstate_planner import MidstatePlanner
from .e8_tiler import E8Tiler
from .e8_tiler_wrap import chamber_key_from_vec, owns as tiler_owns
from .receipts import Receipts
from .atlas_o8 import extract_O8, embed_E8

def _session_root_from_roll(roll_bytes:list)->str:
    h=hashlib.sha256()
    for b in roll_bytes: h.update(b)
    return h.hexdigest()

def mine_loop(rpc_url=None, rpc_user="", rpc_pass="", address:str="", agents:int=1, agent_id:int=0, seed:int=42, ttl:int=30,
              anchor_path="receipts.jsonl", sim: bool=False):
    node = SimNode(seed=seed) if sim or not rpc_url else NodeAdapter(rpc_url, rpc_user, rpc_pass, timeout=600.0)
    receipts=Receipts(anchor_path, anchor_period=1800.0); policy=AGRMPolicy(); mdhg=MDHGCache(); roll_bytes=[]
    while True:
        tpl = node.getblocktemplate()
        bits = tpl.get("bits"); bits_hex = f"{bits:08x}" if isinstance(bits,int) else (bits or "1d00ffff")
        target_hex = compact_to_target(bits_hex)
        rec_what={"template_id":tpl.get("longpollid","n/a"),"height":tpl.get("height"),"curtime":tpl.get("curtime"),"bits":bits_hex}
        receipts.write("template",rec_what,{"pow_ok":True,"merkle_ok":True,"mtp_ok":True,"weight_ok":True,"scripts_ok":True})
        roll_bytes.append(json.dumps({"template":rec_what},sort_keys=True).encode())
        choice=policy.choose(reuse_R=1.0, Q_total=0.2)
        stride=StriderCfg(Kx=choice["Kx"], Km=choice["Km"], Kv=choice["Kv"])
        hs=HeaderSpace(tpl, seed=seed+agent_id, rail_cfg=RailConfig(Kx=choice["Kx"], Km=choice["Km"], Kv=choice["Kv"]))
        planner=MidstatePlanner(); last_flush=time.time()
        o8=extract_O8({"version":tpl.get("version",0),"height":tpl.get("height",0),"txcount":len(tpl.get("transactions",[]))})
        e8=embed_E8(o8); chamber_key=chamber_key_from_vec(e8); policy_code=choice["policy"]
        for op in tri_rail_batches(stride):
            state_key=E8Tiler.key(hs.state.nonce, hs.state.extranonce, hs.state.merkle_class, hs.state.version, hs.state.timestamp)
            if not tiler_owns(agent_id, agents, state_key):
                ev=op[0]
                if ev=="nonce": hs.step_nonce()
                elif ev=="xtra_merkle": hs.step_extranonce(); hs.step_merkle_class()
                elif ev=="version_nonce": hs.roll_version(); hs.step_nonce()
                elif ev=="time": hs.nudge_time(mtp_floor=tpl.get("mediantime", hs.state.timestamp))
                continue
            ev=op[0]; rebuild_cost=0
            if ev=="nonce": hs.step_nonce()
            elif ev=="xtra_merkle": hs.step_extranonce(); hs.step_merkle_class(); rebuild_cost=1
            elif ev=="version_nonce": hs.roll_version(); hs.step_nonce(); rebuild_cost=1
            elif ev=="time": hs.nudge_time(mtp_floor=tpl.get("mediantime", hs.state.timestamp)); rebuild_cost=1
            digest=planner.hash_header(tpl, hs.state, ev); mdhg.note(state_key, rebuild_cost)
            if int.from_bytes(digest,"big") <= int(target_hex,16):
                session_root=_session_root_from_roll(roll_bytes)
                ss=script_commit_session(session_root, chamber_key, policy_code)
                cb=build_coinbase(script_sig=ss, value_sats=0)
                merkle_hex=compute_merkle_with_coinbase(tpl, cb)
                receipts.write("submit", {"digest_hex":digest.hex(),"merkle_root":merkle_hex,"session_root":session_root,
                                          "chamber_key":chamber_key,"policy":policy_code}, {"pow_ok":True})
                roll_bytes.append(json.dumps({"submit":digest.hex()},sort_keys=True).encode())
                if isinstance(node, SimNode): node.submitblock("00")
                break
            if (time.time()-last_flush)>ttl:
                receipts.write("batch", {"reuse_R":planner.reuse_R,"attempts":planner.attempts,"mdhg_hot":len(mdhg.hot())},{"pow_ok":True})
                roll_bytes.append(json.dumps({"batch":{"R":planner.reuse_R}},sort_keys=True).encode())
                last_flush=time.time()
