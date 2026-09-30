def mig_profile(slices, mem_gb, sms=132, bw_gbs=3350):
    return {"sms": round(sms * slices / 7), "mem_gb": mem_gb, "bw_gbs": bw_gbs * slices / 7}


def decode_ms(weight_gb, profile):
    if weight_gb > profile["mem_gb"]:
        return None
    return weight_gb * 1e9 / (profile["bw_gbs"] * 1e9) * 1e3


def best_split(weight_gb, budget_ms, options):
    best = None
    for name, slices, mem_gb in options:
        ms = decode_ms(weight_gb, mig_profile(slices, mem_gb))
        if ms is None or ms > budget_ms:
            continue
        n = 7 // slices
        if best is None or (n, -slices) > (best[1], -best[2]):
            best = (name, n, slices)
    return (best[0], best[1]) if best else None
