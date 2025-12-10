# ca_tile_generator.py
import json, random, hashlib
from pathlib import Path

NIEMEIER_LATTICES = [
    "A1^24","A2^12","A3^8","A4^6","A5^4+A4","A6^4","A7^2+A4^2","A8^3","A9^2+A6","A12^2","A15+A9",
    "A17+A7","A24","D4^6","D5^4+A4","D6^4","D7^2+A5^2","D8^3","D9+A15","D10^2+A4","D12^2","D16+A8","D24","E6^4"
] + ["E7^2+A5^2+A7","E8^3","Leech"][:1]  # keep to 24 baseline + optional

class CATileGenerator:
    def __init__(self, output_dir='.reality_craft/ca_tiles'):
        self.output_dir = Path(output_dir); self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_baseline_tiles(self):
        tiles = {}
        for i, name in enumerate(NIEMEIER_LATTICES[:24]):
            tiles[name] = self._create_tile(i, name, 64, 64)
            self._save_tile(tiles[name], name)
        return tiles

    def generate_custom_tile(self, lattice_name, paper_data):
        try: i = NIEMEIER_LATTICES.index(lattice_name)
        except ValueError: i = 0
        req = self._extract_requirements(paper_data)
        tile = self._create_tile(i, lattice_name, req.get('width',64), req.get('height',64), custom_rules=req.get('rules'))
        self._save_tile(tile, f"{lattice_name}_custom_{paper_data.get('hash','')[:8]}"); return tile

    def _create_tile(self, lattice_id, lattice_name, w, h, custom_rules=None):
        return {
            'id': lattice_id, 'name': lattice_name, 'dimensions': (w,h),
            'initial_state': self._init_state(w,h,lattice_name),
            'rules': custom_rules or self._default_rules(lattice_name),
            'julia_param': self._derive_julia_param(lattice_name),
            'boundary':'toroidal', 'neighbors': self._neighbors(lattice_id)
        }

    def _init_state(self, w, h, name):
        state = [[0]*w for _ in range(h)]
        seeds = 10 if 'Leech' in name else 30
        rnd = random.Random(int(hashlib.sha256(name.encode()).hexdigest(),16)%2**32)
        for _ in range(seeds):
            x = rnd.randrange(0,w); y = rnd.randrange(0,h); state[y][x]=1
        return state

    def _default_rules(self, lattice_name):
        return {'type':'morphonic','survive':[2,3],'birth':[3],'conservation': True,'lattice_coupling': True}

    def _derive_julia_param(self, name):
        h = int(hashlib.sha256(name.encode()).hexdigest(),16)
        real = ((h % 2001)/1000.0) - 1.0
        imag = (((h//2001) % 2001)/1000.0) - 1.0
        return {'real': round(real,3), 'imag': round(imag,3)}

    def _neighbors(self, i):
        row, col = divmod(i, 6)
        top = ((row-1)%4)*6 + col; bottom = ((row+1)%4)*6 + col
        left = row*6 + ((col-1)%6); right = row*6 + ((col+1)%6)
        return {'top': top, 'bottom': bottom, 'left': left, 'right': right}

    def _extract_requirements(self, paper):
        return {'width': 64, 'height': 64, 'rules': None}

    def _save_tile(self, tile, name):
        p = self.output_dir / f"{name}.json"; p.write_text(json.dumps(tile, indent=2), encoding='utf-8')

def setup_ca_system():
    gen = CATileGenerator(); base = gen.generate_baseline_tiles()
    print(f"✓ CA system ready with {len(base)} tiles"); return gen

if __name__ == '__main__':
    setup_ca_system()
