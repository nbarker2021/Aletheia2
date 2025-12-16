import hashlib, random
def analyze(form):
    h = int(hashlib.sha256(("ax"+form['form_id']).encode()).hexdigest(),16)
    rng = random.Random(h & 0xffffffff)
    echoes = []
    if rng.random() < 0.2: echoes.append("axion_mix")
    if rng.random() < 0.25: echoes.append("dark_photon_mix")
    features = {"band":"AXION","octet_pass": int(40 + rng.random()*24)}
    return features, echoes
