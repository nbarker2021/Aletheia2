from typing import Dict, Any, List
import numpy as np, json, math
from pathlib import Path
from .utils import now_receipt
import hashlib

def _blake(s: bytes, n=16):
    import hashlib
    return hashlib.blake2b(s, digest_size=n).digest()

def _read(p: Path, limit=200000) -> str:
    try:
        if p.suffix.lower() in {".txt",".md",".json",".csv",".py"}:
            return p.read_text(encoding="utf-8", errors="ignore")[:limit]
        return p.name
    except Exception:
        return p.name

def _views(texts: List[str], dim=128):
    def lex(t):
        v=np.zeros(dim, dtype=np.float32); b=t.encode('utf-8','ignore')
        for i in range(0,len(b)-3,3):
            h=int.from_bytes(_blake(b[i:i+3],4),'little')%dim; v[h]+=1.0
        s=v.sum(); return (v/s) if s>0 else v
    def spec(t):
        b=np.frombuffer(t.encode('utf-8','ignore'), dtype=np.uint8)
        if b.size==0: return np.zeros(dim, dtype=np.float32)
        n2=1<<(b.size-1).bit_length()
        pad=np.zeros(n2, dtype=np.float32); pad[:b.size]=b.astype(np.float32)
        mag=np.abs(np.fft.rfft(pad)).astype(np.float64)
        bins=np.zeros(dim, dtype=np.float64); idx=np.arange(mag.size)%dim
        np.add.at(bins, idx, mag); bins/= (bins.sum()+1e-12); return bins.astype(np.float32)
    def geom(t):
        v=np.zeros(dim, dtype=np.float32); acc=1469598103934665603; prime=1099511628211
        for j,ch in enumerate(t.encode('utf-8','ignore')):
            acc^=ch; acc=(acc*prime)&((1<<64)-1)
            idx=(acc^(j*0x9E3779B97F4A7C15))&0xffffffff; v[idx%dim]+=1.0
        s=v.sum(); return (v/s) if s>0 else v
    E_lex=np.stack([lex(t) for t in texts])
    E_spec=np.stack([spec(t) for t in texts])
    E_geom=np.stack([geom(t) for t in texts])
    return E_lex,E_spec,E_geom

def run(root: str, max_docs:int=128, out:str=".")->Dict[str,Any]:
    paths=sorted([p for p in Path(root).rglob("*") if p.is_file()])[:max_docs]
    docs=[{"id":p.name,"path":str(p),"text":_read(p)} for p in paths]
    texts=[d["text"] for d in docs]
    E_lex,E_spec,E_geom=_views(texts,128)
    # Distance mean
    def cos(E):
        X=E.astype(np.float64); nrm=np.linalg.norm(X,axis=1,keepdims=True)+1e-12; Xn=X/nrm; D=1.0-(Xn@Xn.T); np.fill_diagonal(D,0.0); return D
    def l2(E):
        X=E.astype(np.float64); G=X@X.T; nrm=np.sum(X*X,axis=1,keepdims=True); D2=nrm+nrm.T-2*G; D2[D2<0]=0; D=np.sqrt(D2); np.fill_diagonal(D,0.0); return D
    def wass(E):
        Cp=np.cumsum(E.astype(np.float64), axis=1); n=Cp.shape[0]; D=np.zeros((n,n),dtype=np.float64)
        for a in range(n): D[a]=np.abs(Cp[a][None,:]-Cp).mean(axis=1)
        np.fill_diagonal(D,0.0); return D
    D = cos(E_lex) + l2(E_spec) + wass(E_geom)
    S = np.exp(-D/(np.median(D)+1e-12))
    cent = S.sum(axis=1); order=np.argsort(-cent)
    ranking=[{"rank":int(i), "doc_id":docs[j]["id"], "path":docs[j]["path"], "centrality":float(cent[j])} for i,j in enumerate(order)]
    # write CSV
    import csv, os
    os.makedirs(out, exist_ok=True)
    with open(Path(out)/"delta_ranking.csv","w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=["rank","doc_id","path","centrality"]); w.writeheader(); w.writerows(ranking)
    return {"receipt": now_receipt({"stage":"delta.run","n":len(docs)}), "ranking": ranking}
