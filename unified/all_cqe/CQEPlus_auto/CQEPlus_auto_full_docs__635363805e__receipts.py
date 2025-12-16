from typing import Dict, Any
from .utils import now_receipt

def stamp(ctx: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    return now_receipt({"ctx": ctx, "payload": payload})
