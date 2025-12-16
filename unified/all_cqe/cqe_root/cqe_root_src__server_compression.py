
from fastapi import APIRouter
import json, pathlib
router = APIRouter()
CONFIG_PATH = pathlib.Path(__file__).parent/'../../compression_config.json'
@router.post('/config/compression')
def set_compression(enable: bool):
    cfg=json.loads(CONFIG_PATH.read_text())
    cfg['compression_enabled']=enable
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    return {"status":"ok","enabled":enable}
