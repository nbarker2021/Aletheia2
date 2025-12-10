import argparse, os
from .runner import mine_loop
def main():
    p=argparse.ArgumentParser(description="CQE–Speedlight Solo Miner (CQE-only MVP)")
    sub=p.add_subparsers(dest="cmd")
    m=sub.add_parser("mine", help="Run miner loop")
    m.add_argument("--rpc-url", default=os.getenv("BITCOIN_RPC_URL"))
    m.add_argument("--rpc-user", default=os.getenv("BITCOIN_RPC_USER",""))
    m.add_argument("--rpc-pass", default=os.getenv("BITCOIN_RPC_PASS",""))
    m.add_argument("--address", required=False)
    m.add_argument("--agents", type=int, default=1)
    m.add_argument("--agent-id", type=int, default=0)
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--ttl", type=int, default=30)
    m.add_argument("--receipts", default="receipts.jsonl")
    m.add_argument("--sim", action="store_true")
    args=p.parse_args()
    if args.cmd=="mine":
        mine_loop(args.rpc_url, args.rpc_user, args.rpc_pass, args.address or "", agents=args.agents,
                  agent_id=args.agent_id, seed=args.seed, ttl=args.ttl, anchor_path=args.receipts, sim=args.sim)
    else: p.print_help()
if __name__=="__main__": main()
