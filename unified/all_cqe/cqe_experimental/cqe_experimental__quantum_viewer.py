import hashlib, random
def analyze(form):
    h = int(hashlib.sha256(("q"+form['form_id']).encode()).hexdigest(),16)
    rng = random.Random(h & 0xffffffff)
    echoes = []
    if rng.random() < 0.5: echoes.append("octet_cover")
    if rng.random() < 0.35: echoes.append("mirror_lock")
    features = {"band":"Q","octet_pass": int(50 + rng.random()*14)}
    return features, echoes
