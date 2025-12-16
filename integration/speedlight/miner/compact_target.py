def compact_to_target(bits):
    if isinstance(bits, str):
        if bits.startswith("0x"): bits = bits[2:]
        bits = int(bits, 16)
    exp = (bits >> 24) & 0xff
    mant = bits & 0x007fffff
    if exp <= 3:
        value = mant >> (8*(3-exp))
    else:
        value = mant << (8*(exp-3))
    value &= (1<<256)-1
    return f"{value:064x}"
