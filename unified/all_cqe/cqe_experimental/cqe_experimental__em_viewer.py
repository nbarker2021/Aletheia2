import hashlib, random
def analyze(form):
    # Deterministic echo based on form_id
    h = int(hashlib.sha256(("em"+form['form_id']).encode()).hexdigest(),16)
    rng = random.Random(h & 0xffffffff)
    echoes = []
    if rng.random() < 0.5: echoes.append("cartan")
    if rng.random() < 0.4: echoes.append("subharmonic")
    if rng.random() < 0.3: echoes.append("hysteresis")
    features = {"band":"EM","octet_pass": int(48 + rng.random()*16)}
    return features, echoes
