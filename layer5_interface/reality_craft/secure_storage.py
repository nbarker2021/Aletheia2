# secure_storage.py
import os, json, hashlib, base64
from pathlib import Path
from datetime import datetime

class _XORCipher:
    def __init__(self, key: bytes): import hashlib as _h; self.key = _h.sha256(key).digest()
    def encrypt(self, data: bytes) -> bytes:
        out = bytes(b ^ self.key[i % len(self.key)] for i,b in enumerate(data))
        import base64 as _b; return _b.urlsafe_b64encode(out)
    def decrypt(self, tok: bytes) -> bytes:
        import base64 as _b; raw = _b.urlsafe_b64decode(tok)
        return bytes(b ^ self.key[i % len(self.key)] for i,b in enumerate(raw))

try:
    from cryptography.fernet import Fernet as _Fernet
    _HAVE_CRYPTO = True
except Exception:
    _Fernet = None; _HAVE_CRYPTO = False

class SecureStorage:
    def __init__(self, local_dir='.reality_craft/secure', backup_pi_ip=None):
        self.local_dir = Path(local_dir); self.local_dir.mkdir(parents=True, exist_ok=True)
        self.backup_pi_ip = backup_pi_ip
        self.key = self._get_or_create_key()
        self.cipher = (_Fernet(self.key) if _HAVE_CRYPTO else _XORCipher(self.key))

    def _get_or_create_key(self):
        key_file = self.local_dir / 'encryption.key'
        if key_file.exists(): return key_file.read_bytes()
        key = os.urandom(32) if not _HAVE_CRYPTO else _Fernet.generate_key()
        key_file.write_bytes(key); os.chmod(key_file, 0o600); return key

    def store(self, data_id, data, encrypt=True):
        blob = json.dumps(data, sort_keys=True).encode()
        stored = self.cipher.encrypt(blob) if encrypt else blob
        fp = self.local_dir / f"{data_id}.enc"; fp.write_bytes(stored)
        h = hashlib.sha256(stored).hexdigest()
        meta = {'id': data_id, 'hash': h, 'encrypted': encrypt, 'timestamp': datetime.now().isoformat(), 'size': len(stored), 'engine': 'fernet' if _HAVE_CRYPTO else 'xor-demo'}
        (self.local_dir / f"{data_id}.meta").write_text(json.dumps(meta, indent=2), encoding='utf-8')
        return {'success': True, 'hash': h, 'engine': meta['engine']}

    def retrieve(self, data_id, decrypt=True):
        fp = self.local_dir / f"{data_id}.enc"
        if not fp.exists(): return None
        data = fp.read_bytes()
        if decrypt: 
            try: plain = self.cipher.decrypt(data)
            except Exception: return None
        else: plain = data
        try: return json.loads(plain.decode())
        except Exception: return None

    def list_stored(self):
        out = []
        for m in self.local_dir.glob('*.meta'):
            try: out.append(json.loads(m.read_text(encoding='utf-8')))
            except Exception: pass
        return out

    def verify_integrity(self):
        res = []
        for m in self.local_dir.glob('*.meta'):
            try:
                meta = json.loads(m.read_text(encoding='utf-8'))
                fp = self.local_dir / f"{meta['id']}.enc"
                if not fp.exists(): res.append({'id': meta['id'], 'status': 'missing'}); continue
                ok = hashlib.sha256(fp.read_bytes()).hexdigest() == meta['hash']
                res.append({'id': meta['id'], 'status': 'ok' if ok else 'corrupted'})
            except Exception as e:
                res.append({'id': m.stem, 'status': 'error', 'message': str(e)})
        return res
