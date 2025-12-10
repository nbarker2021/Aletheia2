
# speedlight_plus_compat — optional wrapper if SpeedLightPlus vendored
try:
    from .speedlight_sidecar_plus import SpeedLightPlus
    HAVE_SPEEDLIGHT = True
except Exception:
    HAVE_SPEEDLIGHT = False
    SpeedLightPlus = None

SL_INSTANCE = None

def get_speedlight(mem_bytes=256_000_000, disk_dir=".sidecar/sl_cache", ledger_path=".sidecar/sl_ledger.jsonl"):
    global SL_INSTANCE
    if not HAVE_SPEEDLIGHT:
        return None
    if SL_INSTANCE is None:
        SL_INSTANCE = SpeedLightPlus(mem_bytes=mem_bytes, disk_dir=disk_dir, ledger_path=ledger_path)
    return SL_INSTANCE
