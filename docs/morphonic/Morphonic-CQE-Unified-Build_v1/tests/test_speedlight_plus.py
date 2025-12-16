from morphonic_cqe_unified.sidecar.speedlight_sidecar_plus import SpeedLightPlus
def test_basic_cache():
    sl = SpeedLightPlus(mem_bytes=5_000_000)
    payload = {"op":"square_sum","n":10000}
    def compute():
        return {"sum": sum(i*i for i in range(10000))}
    r1, c1, id1 = sl.compute(payload, scope="test", channel=3, compute_fn=compute)
    r2, c2, id2 = sl.compute(payload, scope="test", channel=3, compute_fn=compute)
    assert r1 == r2 and id1 == id2 and c2 == 0.0
