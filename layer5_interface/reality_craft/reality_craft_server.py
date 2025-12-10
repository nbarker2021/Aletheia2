# reality_craft_server.py
import os, json, hashlib, mimetypes, math, random
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from speedlight_sidecar_plus import SpeedLightV2

def _shannon_entropy(data: bytes) -> float:
    if not data: return 0.0
    from collections import Counter
    n = len(data); c = Counter(data)
    return -sum((cnt/n)*math.log2(cnt/n) for cnt in c.values())

def _detect_type(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    types = {'.pdf':'Paper','.tex':'LaTeX','.md':'Markdown','.py':'Python Code','.js':'JavaScript','.csv':'Dataset','.json':'Data'}
    return types.get(ext, 'Document')

class RealityCraftServer(BaseHTTPRequestHandler):
    speedlight = None
    file_index = {}
    equivalence_db = {}

    @classmethod
    def initialize(cls):
        cls.speedlight = SpeedLightV2(mem_bytes=512*1024*1024, disk_dir='./.reality_craft/cache', ledger_path='./.reality_craft/ledger.jsonl')
        Path('./.reality_craft').mkdir(parents=True, exist_ok=True)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self): self.send_response(204); self.end_headers()

    def do_GET(self):
        p = urlparse(self.path).path
        if p == '/' or p == '/portal':
            self._serve_file('reality_craft_portal.html', 'text/html'); return
        if p == '/api/metrics':
            stats = self.speedlight.stats()
            metrics = {'hit_rate': self._hit_rate(), 'avg_query_time': round(stats['elapsed_s']*1000,3), 'storage_mb': self._storage_mb(), 'merit_balance': 0.0, 'active_simulations': 0}
            self._json(metrics); return
        if p == '/api/export':
            self._export_db(); return
        self.send_error(404)

    def do_POST(self):
        p = urlparse(self.path).path
        length = int(self.headers.get('Content-Length','0'))
        body = self.rfile.read(length) if length else b'{}'
        if p == '/api/scan':
            files = self._scan(); self._json({'files': files}); return
        if p == '/api/process':
            data = json.loads(body or b'{}'); path = data.get('path')
            res = self._process(path); self._json(res); return
        if p == '/api/combine':
            data = json.loads(body or b'{}')
            res = self._combine(data.get('class1'), data.get('class2')); self._json(res); return
        if p == '/api/sync-backup':
            ok = self._sync_backup(); self._json(ok); return
        self.send_error(404)

    # --- helpers ---
    def _scan(self):
        home = Path.home()
        target_dirs = [home/'Documents', home/'Desktop', home/'Downloads', home/'Papers']
        exts = {'.pdf','.tex','.md','.txt','.py','.js','.html','.css','.csv','.json','.xml','.doc','.docx'}
        files = []
        for d in target_dirs:
            if not d.exists(): continue
            for f in d.rglob('*'):
                if f.is_file() and f.suffix.lower() in exts:
                    files.append({'name': f.name, 'path': str(f), 'size': f.stat().st_size, 'modified': f.stat().st_mtime, 'scanned': False})
        self.file_index = {x['path']: x for x in files}
        return files

    def _process(self, filepath: str):
        if not filepath or not Path(filepath).exists():
            return {'error':'file not found','type':'Unknown','equivalence_class':'0'*64,'geometric_signature':{}}
        self.file_index.get(filepath, {'scanned': True})['scanned'] = True
        with open(filepath, 'rb') as f: content = f.read()
        result = self.speedlight.compute(payload={'path': filepath, 'size': len(content)}, scope='local', channel=3,
                                         compute_fn=lambda: {'hash': hashlib.sha256(content).hexdigest(),'size': len(content),'entropy': _shannon_entropy(content)})
        eq = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        self.equivalence_db[eq] = {'canonical_form': result, 'sources':[filepath], 'created': datetime.now().isoformat()}
        return {'type': _detect_type(filepath), 'equivalence_class': eq, 'geometric_signature': result}

    def _combine(self, c1, c2):
        c1d = self.equivalence_db.get(c1,{}).get('canonical_form'); c2d = self.equivalence_db.get(c2,{}).get('canonical_form')
        if not c1d or not c2d: return {'discovery': None}
        combo = {'combined': True, 'hash1': c1d.get('hash'), 'hash2': c2d.get('hash'), 'operation': 'monster_conjugation'}
        ch = hashlib.sha256(json.dumps(combo, sort_keys=True).encode()).hexdigest()
        if ch in self.equivalence_db: return {'discovery': None}
        import random
        merit = round(random.uniform(1,100),2)
        self.equivalence_db[ch] = {'canonical_form': combo, 'sources':[c1,c2], 'created': datetime.now().isoformat(), 'merit': merit}
        return {'discovery': True, 'title': f"Synthesis of {c1[:8]} and {c2[:8]}", 'equivalence_class': ch, 'merit': merit, 'proof_chain':[c1,c2,ch]}

    def _sync_backup(self):
        cfg = Path('.reality_craft/config.json')
        backup_ip = None
        if cfg.exists():
            try: backup_ip = json.loads(cfg.read_text()).get('backup_pi_ip')
            except Exception: backup_ip = None
        if not backup_ip: return {'success': False, 'error': 'Backup Pi not configured'}
        payload = {'equivalence_classes': self.equivalence_db, 'file_index': self.file_index, 'timestamp': datetime.now().isoformat()}
        try:
            import requests
            r = requests.post(f'http://{backup_ip}:8766/api/backup', json=payload, timeout=10)
            if r.status_code == 200: return {'success': True, 'timestamp': datetime.now().isoformat()}
            return {'success': False, 'error': str(r.text)}
        except Exception as e:
            try:
                from urllib.request import Request, urlopen
                req = Request(f'http://{backup_ip}:8766/api/backup', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
                with urlopen(req, timeout=10) as _:
                    return {'success': True, 'timestamp': datetime.now().isoformat()}
            except Exception as ee:
                return {'success': False, 'error': str(ee)}

    def _export_db(self):
        export = {'version':'1.0','timestamp': datetime.now().isoformat(), 'equivalence_classes': self.equivalence_db, 'file_index': self.file_index}
        payload = json.dumps(export, indent=2).encode()
        self.send_response(200); self.send_header('Content-Type','application/json')
        self.send_header('Content-Disposition', f'attachment; filename="reality_craft_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json"')
        self.end_headers(); self.wfile.write(payload)

    def _serve_file(self, filename, content_type):
        try:
            with open(filename, 'rb') as f:
                self.send_response(200); self.send_header('Content-Type', content_type); self.end_headers(); self.wfile.write(f.read())
        except FileNotFoundError:
            self.send_error(404)

    def _hit_rate(self):
        st = self.speedlight.stats(); tot = st['hits'] + st['misses']; return 0 if tot==0 else round((st['hits']/tot)*100, 1)

    def _storage_mb(self):
        d = Path('.reality_craft/cache'); 
        if not d.exists(): return 0.0
        total = 0
        for p in d.rglob('*'):
            if p.is_file(): total += p.stat().st_size
        return round(total/(1024*1024),3)

def run_server(port=8765):
    RealityCraftServer.initialize()
    if not Path('reality_craft_portal.html').exists():
        src_portal = Path(__file__).parent / 'reality_craft_portal.html'
        if src_portal.exists(): Path('reality_craft_portal.html').write_text(src_portal.read_text(encoding='utf-8'), encoding='utf-8')
    server = HTTPServer(('localhost', port), RealityCraftServer)
    print(f"✓ Reality Craft Portal running on http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n✓ Server stopped")

if __name__ == '__main__':
    run_server()
