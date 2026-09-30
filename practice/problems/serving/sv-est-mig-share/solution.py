SLICES = {"1g.10gb": (1, 10), "2g.20gb": (2, 20), "3g.40gb": (3, 40), "7g.80gb": (7, 80)}
ORDER = ["1g.10gb", "2g.20gb", "3g.40gb", "7g.80gb"]


def mig_share(profile):
    slices, mem = SLICES[profile]
    return 7 // slices, mem, slices / 7


def fits(profile, weight_gb, kv_gb):
    _, mem, _ = mig_share(profile)
    return weight_gb + kv_gb <= mem


def decode_ms(weight_gb, bandwidth_gbs, share=1.0):
    return weight_gb / (bandwidth_gbs * share) * 1000


def choose(weight_gb, kv_gb, latency_budget_ms, tenants, bandwidth_gbs=3350):
    for profile in ORDER:                                  # 从切得最细的开始找
        count, _, share = mig_share(profile)
        if count < tenants:
            continue
        if not fits(profile, weight_gb, kv_gb):
            continue
        if decode_ms(weight_gb, bandwidth_gbs, share) <= latency_budget_ms:
            return profile
    return None
