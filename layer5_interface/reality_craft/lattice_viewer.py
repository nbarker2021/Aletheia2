# lattice_viewer.py
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import json

class Viewer(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            html = Path(__file__).parent / 'lattice_viewer.html'
            if html.exists():
                self.send_response(200); self.send_header('Content-Type','text/html'); self.end_headers()
                self.wfile.write(html.read_bytes()); return
        if self.path == '/api/tiles':
            tiles_dir = Path('.reality_craft/ca_tiles')
            payload = []
            if tiles_dir.exists():
                for fp in tiles_dir.glob('*.json'):
                    try: payload.append(json.loads(fp.read_text(encoding='utf-8')))
                    except Exception: pass
            self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers()
            self.wfile.write(json.dumps(payload).encode()); return
        self.send_error(404)

def run(port=8989):
    server = HTTPServer(('localhost', port), Viewer)
    print(f"✓ Lattice viewer http://localhost:{port}")
    try: server.serve_forever()
    except KeyboardInterrupt: print("\n✓ Viewer stopped")

if __name__ == '__main__':
    run()
