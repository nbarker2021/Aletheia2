
from cqe_unified.config import Config
from cqe_unified.harness import MasterHarness

def test_pose_and_uvibs():
    cfg = Config.from_file("configs/default.json")
    mh = MasterHarness(cfg)
    # optimize pose then validate
    out = mh.system.process_text_sliced("pose uvibs", plan=["ingest_text","e8_embed","pose_optimizer","uvibs_monster_validate"])
    assert out.get("octet_ok") in (True, False)
