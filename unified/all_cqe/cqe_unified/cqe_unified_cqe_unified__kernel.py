
from typing import Dict, Any
from .utils import make_run_id, set_seed
from .receipts import Receipts
from .storage import Storage
from .io_manager import IOManager
from .embedding_e8 import embed_string
from .fractal_mandelbrot import analyze_string
from .toroidal import generate_toroidal_shell
from .objective_phi import feature_pack, compute_phi
from .semantics import extract_semantics
from .governance import Governance
from .validation import compute_v_total, band_for

class CQEUnifiedSystem:
    def __init__(self, cfg: "Config"):
        self.cfg = cfg
        self.run_id = make_run_id("cqe")
        self.receipts = Receipts(cfg.receipts_dir)
        self.storage = Storage("data")
        self.io = IOManager(cfg.artifacts_dir)
        self.gov = Governance()

    def emit(self, kind: str, payload: Dict[str, Any]):
        rec = {"run_id": self.run_id, "event": kind, **payload}
        self.receipts.emit(self.run_id, rec)

    def process_text(self, text: str) -> Dict[str, Any]:
        # Ingest
        ing = self.io.ingest_text(text)
        self.emit("ingest", {"ingest": ing})

        # E8 embed (stub)
        e8 = embed_string(text, seed=self.cfg.seed)
        self.emit("e8_embed", {"e8": e8})

        # Fractal (optional)
        fr = analyze_string(text) if self.cfg.enable_mandelbrot else {"behavior": "SKIP"}
        self.emit("fractal", {"fractal": fr})

        # Toroidal (optional) — collect a small shell, just to emit stats
        tor = generate_toroidal_shell(n_points=64, seed=self.cfg.seed) if self.cfg.enable_toroidal else []
        patterns = {}
        for p in tor:
            patterns[p["pattern"]] = patterns.get(p["pattern"], 0) + 1
        self.emit("toroidal", {"patterns": patterns, "n": len(tor)})

        # Φ (demo features)
        features = feature_pack(
            geom = max(0.0, 1.0 - e8["root_distance"]),
            parity = 0.5,  # placeholder
            sparsity = 0.4,  # placeholder
            kissing = 0.3   # placeholder
        )
        phi = compute_phi(features, self.cfg.phi_weights)
        self.emit("phi", {"features": features, "phi": phi})

        # Semantics (placeholder distance/angle)
        sem = extract_semantics(e8_distance=e8["root_distance"], angle_hint=0.2)
        self.emit("semantics", {"semantics": sem})

        # Governance gates (use demo metrics)
        gates_in = {"W4": features["geom"], "W80": features["parity"], "Wexp": features["sparsity"], "LAWFUL": features["kissing"]}
        gates_out = self.gov.evaluate(gates_in)
        self.emit("governance", {"input": gates_in, "results": gates_out})

        # Validation roll-up
        v = compute_v_total(scores={k: v["score"] for k,v in gates_out.items()}, weights={"W4":0.3,"W80":0.3,"Wexp":0.2,"LAWFUL":0.2})
        band = band_for(v)
        self.emit("validation", {"v_total": v, "band": band})

        return {"e8": e8, "fractal": fr, "patterns": patterns, "phi": phi, "semantics": sem, "v_total": v, "band": band}
