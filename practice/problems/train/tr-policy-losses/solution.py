import math


def token_weights(advantages, lengths, mode):
    if mode == "seq":
        n = len(advantages)
        return [a / (L * n) for a, L in zip(advantages, lengths)]
    if mode == "token":
        total = sum(lengths)
        return [a / total for a in advantages]
    raise ValueError(f"未知的 mode：{mode}")


def gspo_ratio(logp_new, logp_old):
    return math.exp(sum(a - b for a, b in zip(logp_new, logp_old)) / len(logp_new))


def cispo_weights(ratios, eps_low=0.2, eps_high=0.28):
    return [min(max(r, 1 - eps_low), 1 + eps_high) for r in ratios]


def k3(logp, logp_ref):
    out = []
    for lp, lr in zip(logp, logp_ref):
        log_r = lr - lp
        out.append(math.exp(log_r) - 1 - log_r)
    return out
