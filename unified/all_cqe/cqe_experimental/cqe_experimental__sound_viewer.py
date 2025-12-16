import hashlib, random, math
def analyze(form):
    h = int(hashlib.sha256(("snd"+form['form_id']).encode()).hexdigest(),16)
    rng = random.Random(h & 0xffffffff)
    echoes = []
    if rng.random() < 0.45: echoes.append("beats")
    if rng.random() < 0.35: echoes.append("harmonic_lock")
    if rng.random() < 0.25: echoes.append("subharmonic")
    features = {"band":"SOUND","octet_pass": int(44 + rng.random()*20)}
    return features, echoes
