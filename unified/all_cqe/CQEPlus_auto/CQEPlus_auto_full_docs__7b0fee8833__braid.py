from typing import Dict, Any
from .delta import run as delta_run
from .utils import now_receipt
def run(root: str, max_docs:int, out:str) -> Dict[str, Any]:
    res = delta_run(root, max_docs, out)
    return now_receipt({"stage":"braid.run","root":root,"max_docs":int(max_docs),"out":out})
