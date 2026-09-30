SLICES = {"1g.10gb": (1, 10), "2g.20gb": (2, 20), "3g.40gb": (3, 40), "7g.80gb": (7, 80)}
ORDER = ["1g.10gb", "2g.20gb", "3g.40gb", "7g.80gb"]


def mig_share(profile):
    slices, mem = SLICES[profile]
    return 7 // slices, mem, 1.0                           # 以为算力不打折


def fits(profile, weight_gb, kv_gb):
    _, mem, _ = mig_share(profile)
    return weight_gb <= mem                                # 忘了 KV 也要显存


def decode_ms(weight_gb, bandwidth_gbs, share=1.0):
    return weight_gb / bandwidth_gbs * 1000                # 没按份额打折


def choose(weight_gb, kv_gb, latency_budget_ms, tenants, bandwidth_gbs=3350):
    for profile in ORDER:
        if fits(profile, weight_gb, kv_gb):
            return profile                                 # 没检查份数和延迟
    return None
