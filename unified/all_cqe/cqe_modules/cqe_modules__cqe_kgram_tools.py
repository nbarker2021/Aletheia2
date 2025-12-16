# cqe_kgram_tools.py
# Simple k-gram extraction to compare tokens vs snippets (shapes-first).

from collections import Counter

def kgrams(s: str, k: int = 5):
    s = s or ""
    s2 = "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace())
    s2 = " ".join(s2.split())
    return [s2[i:i+k] for i in range(max(0, len(s2)-k+1))]

def overlap(a: str, b: str, k: int = 5):
    A = Counter(kgrams(a, k))
    B = Counter(kgrams(b, k))
    keys = set(A) & set(B)
    common = sum(min(A[x], B[x]) for x in keys)
    total = sum(A.values()) + sum(B.values())
    score = (2*common) / total if total else 0.0
    return {"k": k, "common": common, "score": score, "keys": sorted(keys)}
