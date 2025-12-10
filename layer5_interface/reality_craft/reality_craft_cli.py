# reality_craft_cli.py
import argparse
from reality_craft_server import run_server
from ca_tile_generator import setup_ca_system
from lattice_viewer import run as run_viewer

def main():
    ap = argparse.ArgumentParser(description="RealityCraft CLI")
    ap.add_argument('cmd', choices=['serve','tiles','viewer'])
    ap.add_argument('--port', type=int, default=None)
    args = ap.parse_args()
    if args.cmd == 'serve':
        run_server(port=args.port or 8765)
    elif args.cmd == 'tiles':
        setup_ca_system()
    elif args.cmd == 'viewer':
        run_viewer(port=args.port or 8989)

if __name__ == '__main__':
    main()
