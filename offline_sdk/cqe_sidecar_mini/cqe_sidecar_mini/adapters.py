
# cqe_sidecar_mini.adapters — stdlib-only adapters (placeholders you can swap with your real ones)
import math, hashlib

def geotokenize(text: str):
    """Shape-ish features from text: counts by char class and run-length signatures."""
    vowels = set("aeiou"); cons=0; vow=0; digit=0; other=0
    last=None; runs=[]; rlen=0
    for ch in text or "":
        cl = 'v' if ch.lower() in vowels else ('d' if ch.isdigit() else ('c' if ch.isalpha() else 'x'))
        if cl=='v': vow+=1
        elif cl=='c': cons+=1
        elif cl=='d': digit+=1
        else: other+=1
        if cl==last: rlen+=1
        else:
            if last is not None: runs.append((last, rlen))
            last=cl; rlen=1
    if last is not None: runs.append((last, rlen))
    sig = hashlib.sha256(("".join(f"{a}{b}" for a,b in runs)).encode()).hexdigest()[:16]
    return {"counts":{"vowel":vow,"consonant":cons,"digit":digit,"other":other},"runs":runs[:16],"signature":sig}

def mdhg_signal(vec: dict, k:int=8):
    """Multi-D Hamiltonian-like toy: project frequency vector through golden-ratio phased oscillators."""
    phi = (1+5**0.5)/2
    keys = sorted(vec.keys())
    amps = [vec.get(k,0.0) for k in keys]
    total = sum(amps) or 1.0
    freqs = [(i+1)/len(keys or [1]) for i,_ in enumerate(keys)]
    coords = []
    e = 0.0
    for i in range(k):
        phase = (phi**i) % 1.0
        c = sum( (amps[j]/total) * math.sin(2*math.pi*(freqs[j]*phase)) for j in range(len(keys)) )
        coords.append(c); e += c*c
    return {"k":k,"coords":coords,"energy":e}

def moonshine_crosshit(vec: dict, mod:int=13):
    """Toy modular cross-hit score; inspired by modular/monster cross-correlations (no external math)."""
    keys = sorted(vec.keys()); s=0
    for i,k in enumerate(keys):
        v = vec[k]
        s = (s + (i+1)*v) % mod
    density = (sum(vec.values())/(len(keys) or 1)) % mod
    score = (s * (1+density)) % mod
    return {"mod":mod,"score":score,"density":density}


# --- AGRM / MDHG integration stubs (replace bodies with your real logic) ---
def agrm_hashcity(seed: str, ticks: int = 16, cities: int = 4, rooms_per_city: int = 8):
    """Legacy AGRM-ish 'hash-only' city evolution — deterministic, stdlib-only.
    Returns tiny summary you can receipt; expand as you like.
    """
    import hashlib
    def rnd(s):
        h = hashlib.sha256(s.encode()).hexdigest()
        # map hex chars to small ints
        return sum(int(h[i:i+2], 16) for i in range(0,len(h),2)) % 1000
    pop = [100 + rnd(f"{seed}:init:{i}")%50 for i in range(cities)]
    econ = [rnd(f"{seed}:econ:{i}")%100 for i in range(cities)]
    hist = []
    for t in range(1, ticks+1):
        for i in range(cities):
            # simple deterministic flows driven by hash; no RNG beyond hashlib
            delta = (rnd(f"{seed}:t{t}:c{i}") % 7) - 3
            econ[i] = max(0, econ[i] + delta)
            pop[i] = max(0, pop[i] + (delta - 1))
        hist.append({"t": t, "pop": pop[:], "econ": econ[:]})
    score = sum(econ) + sum(pop)
    return {"ticks": ticks, "cities": cities, "score": score, "tail": hist[-1] if hist else {}}

def mdhg_features(vec: dict, k:int=12):
    """Compatibility alias that wraps mdhg_signal with a different default k."""
    base = mdhg_signal(vec, k=k)
    base["alias"] = "mdhg_features"
    return base
