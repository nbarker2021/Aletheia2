
from cqe_unified.config import Config
from cqe_unified.harness import MasterHarness

def test_carlson_exhibit():
    cfg = Config.from_file("configs/default.json")
    mh = MasterHarness(cfg)
    out = mh.system.process_text_sliced("carlson demo", plan=["carlson_correspondence_check"])
    assert out.get("octet_ok") in (True, False)
