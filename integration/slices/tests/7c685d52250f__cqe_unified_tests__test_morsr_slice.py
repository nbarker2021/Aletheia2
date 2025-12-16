
from cqe_unified.config import Config
from cqe_unified.harness import MasterHarness

def test_morsr_runs():
    cfg = Config.from_file("configs/default.json")
    mh = MasterHarness(cfg)
    out = mh.system.process_text_sliced("morsr demo", plan=["ingest_text","e8_embed","morsr_complete_traversal"])
    assert out.get("octet_ok") in (True, False)
