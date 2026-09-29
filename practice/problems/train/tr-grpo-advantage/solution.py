import math


def group_advantages(rewards, G, eps=1e-6):
    if G < 2 or len(rewards) % G:
        raise ValueError("G 至少为 2，且奖励个数要是 G 的整数倍")
    out = []
    for g in range(0, len(rewards), G):
        group = rewards[g:g + G]
        mean = sum(group) / G
        std = math.sqrt(sum((r - mean) ** 2 for r in group) / (G - 1))
        out += [(r - mean) / (std + eps) for r in group]
    return out


def clip_loss(ratios, advantages, eps=0.2):
    terms = [min(r * a, min(max(r, 1 - eps), 1 + eps) * a) for r, a in zip(ratios, advantages)]
    return -sum(terms) / len(terms)


def informative_groups(rewards, G):
    return [g // G for g in range(0, len(rewards), G) if len(set(rewards[g:g + G])) > 1]
