import hashlib, struct
def sha256d(b: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()
def varint(n: int) -> bytes:
    if n < 0xfd: return bytes([n])
    if n <= 0xffff: return b'\xfd' + struct.pack('<H', n)
    if n <= 0xffffffff: return b'\xfe' + struct.pack('<I', n)
    return b'\xff' + struct.pack('<Q', n)
def push(b: bytes) -> bytes:
    n = len(b)
    if n < 0x4c: return bytes([n]) + b
    if n <= 0xff: return b'\x4c' + bytes([n]) + b
    if n <= 0xffff: return b'\x4d' + n.to_bytes(2,'little') + b
    return b'\x4e' + n.to_bytes(4,'little') + b
def script_commit_session(session_root_hex: str, chamber_key: int, policy_code: int) -> bytes:
    root = bytes.fromhex(session_root_hex)
    ck8 = (chamber_key & ((1<<64)-1)).to_bytes(8,'little')
    pc1 = bytes([policy_code & 0xff])
    return push(root)+push(ck8)+push(pc1)
def build_coinbase(script_sig: bytes, value_sats: int=0, address_script: bytes=b'\x6a') -> bytes:
    version = struct.pack('<I', 1)
    vin_cnt = varint(1)
    prevout = b'\xff'*32 + struct.pack('<I', 0xffffffff)
    ss_len = varint(len(script_sig))
    seq = struct.pack('<I', 0xffffffff)
    vout_cnt = varint(1)
    val = struct.pack('<q', value_sats)
    pk_len = varint(len(address_script))
    locktime = struct.pack('<I', 0)
    return b''.join([version, vin_cnt, prevout, ss_len, script_sig, seq,
                     vout_cnt, val, pk_len, address_script, locktime])
def coinbase_txid_le(coinbase_tx: bytes) -> bytes:
    return sha256d(coinbase_tx)[::-1]
def merkle_root(txids_le: list) -> bytes:
    layer = txids_le[:]
    if not layer: return b'\x00'*32
    while len(layer) > 1:
        if len(layer) % 2 == 1: layer.append(layer[-1])
        nxt = []
        for i in range(0, len(layer), 2): nxt.append(sha256d(layer[i]+layer[i+1]))
        layer = nxt
    return layer[0]
def compute_merkle_with_coinbase(template: dict, coinbase_tx: bytes) -> str:
    txids = [bytes.fromhex(x['txid'])[::-1] for x in template.get('transactions', [])]
    cb_le = coinbase_txid_le(coinbase_tx)
    root_le = merkle_root([cb_le]+txids)
    return root_le[::-1].hex()
