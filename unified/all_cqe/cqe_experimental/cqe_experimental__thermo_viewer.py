import hashlib, random
def analyze(form):
    h = int(hashlib.sha256(("th"+form['form_id']).encode()).hexdigest(),16)
    rng = random.Random(h & 0x7fffffff)
    echoes = []
    if rng.random() < 0.4: echoes.append("entropy_flow")
    if rng.random() < 0.3: echoes.append("landauer")
    features = {"band":"THERMO","octet_pass": int(42 + rng.random()*22)}
    return features, echoes
