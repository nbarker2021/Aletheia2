from typing import Dict, Any
def e8_from_4d(seed:int=0) -> Dict[str, Any]:
    return {"type":"E8_slice","seed":int(seed)}
def niemeier_context(name:str='Leech') -> Dict[str, Any]:
    return {"type":"Niemeier","name":name}
def cartan_inward_projection(state: Dict[str, Any], negative_form: bool=True) -> Dict[str, Any]:
    s=dict(state); s['projection']='cartan_inward'; s['negative_form']=bool(negative_form); return s
