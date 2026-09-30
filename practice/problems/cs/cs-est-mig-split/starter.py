def mig_profile(slices, mem_gb, sms=132, bw_gbs=3350):
    return {"sms": round(sms * slices / 7), "mem_gb": mem_gb, "bw_gbs": bw_gbs}   # 带宽没有跟着切


def decode_ms(weight_gb, profile):
    return weight_gb * 1e9 / (profile["bw_gbs"] * 1e9) * 1e3                      # 没检查显存放不放得下


def best_split(weight_gb, budget_ms, options):
    for name, slices, mem_gb in options:
        ms = decode_ms(weight_gb, mig_profile(slices, mem_gb))
        if ms is not None and ms <= budget_ms:
            return name, 7 // slices                                              # 返回第一个满足的，不是实例最多的
    return None
