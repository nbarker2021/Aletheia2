
from typing import Dict, Any, List
from .config import Config
from .kernel import CQEUnifiedSystem
from .utils import set_seed

class MasterHarness:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.system = CQEUnifiedSystem(cfg)

    def run_on_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        results = []
        for t in texts:
            results.append(self.system.process_text(t))
        return results
