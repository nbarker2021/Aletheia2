# backup_pi_server.py
from http.server import HTTPServer, BaseHTTPRequestHandler
import json, hashlib
from pathlib import Path
from datetime import datetime

class BackupPiServer(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/api/backup':
            length = int(self.headers.get('Content-Length','0')); data = json.loads(self.rfile.read(length) or b'{}')
            res = self.store_backup(data); self._json(res); return
        if self.path == '/api/verify':
            res = self.verify_backups(); self._json(res); return
        self.send_error(404)

    def do_GET(self):
        if self.path == '/api/list-backups':
            self._json({'backups': self.list_backups()}); return
        if self.path.startswith('/api/restore/'):
            bid = self.path.split('/')[-1]; self._json(self.restore_backup(bid)); return
        self.send_error(404)

    def store_backup(self, data):
        root = Path('./reality_craft_backups'); root.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().isoformat(); bid = hashlib.sha256(ts.encode()).hexdigest()[:16]
        fp = root / f"backup_{bid}.json"; fp.write_text(json.dumps({'id':bid,'timestamp':ts,'data':data}, indent=2), encoding='utf-8')
        chk = hashlib.sha256(fp.read_bytes()).hexdigest(); (root/f"backup_{bid}.sha256").write_text(chk, encoding='utf-8')
        self._cleanup(root, keep=10)
        return {'success': True, 'backup_id': bid, 'timestamp': ts, 'checksum': chk}

    def verify_backups(self):
        root = Path('./reality_craft_backups'); res = []
        for fp in sorted(root.glob('backup_*.json')):
            bid = fp.stem.replace('backup_',''); chkf = root/f"backup_{bid}.sha256"
            if not chkf.exists(): res.append({'id': bid, 'status':'error','message':'Checksum file missing'}); continue
            actual = hashlib.sha256(fp.read_bytes()).hexdigest(); expect = chkf.read_text().strip()
            res.append({'id': bid, 'status':'ok' if actual==expect else 'corrupted', 'message': 'Checksum verified' if actual==expect else 'Checksum mismatch'})
        return {'results': res}

    def list_backups(self):
        root = Path('./reality_craft_backups'); out = []
        for fp in sorted(root.glob('backup_*.json'), reverse=True):
            try:
                data = json.loads(fp.read_text(encoding='utf-8'))
                out.append({'id': data.get('id'), 'timestamp': data.get('timestamp'), 'size': fp.stat().st_size})
            except Exception:
                pass
        return out

    def restore_backup(self, bid):
        root = Path('./reality_craft_backups'); fp = root/f"backup_{bid}.json"
        if not fp.exists(): return {'error':'Backup not found'}
        return json.loads(fp.read_text(encoding='utf-8'))

    def _cleanup(self, root: Path, keep=10):
        items = sorted(root.glob('backup_*.json'), reverse=True)
        for old in items[keep:]:
            bid = old.stem.replace('backup_',''); old.unlink(missing_ok=True); (root/f"backup_{bid}.sha256").unlink(missing_ok=True)

    def _json(self, data):
        self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(json.dumps(data).encode())

def run_backup_server(port=8766):
    server = HTTPServer(('0.0.0.0', port), BackupPiServer)
    print(f"✓ Backup Pi server running on port {port}")
    try: server.serve_forever()
    except KeyboardInterrupt: print("\n✓ Backup server stopped")

if __name__ == '__main__':
    run_backup_server()
