def replicate(load, n_phys):
    cnt = [1] * len(load)
    for _ in range(n_phys - len(load)):
        e = max(range(len(load)), key=lambda i: (load[i] / cnt[i], -i))
        cnt[e] += 1
    return cnt


def balanced_packing(weights, n_packs):
    per = len(weights) // n_packs
    packs, sums = [[] for _ in range(n_packs)], [0.0] * n_packs
    for i in sorted(range(len(weights)), key=lambda i: (-weights[i], i)):
        p = min((j for j in range(n_packs) if len(packs[j]) < per), key=lambda j: (sums[j], j))
        packs[p].append(i)
        sums[p] += weights[i]
    return packs


def imbalance(load, cnt, packs, phys):
    totals = [sum(load[phys[i]] / cnt[phys[i]] for i in p) for p in packs]
    return max(totals) / (sum(totals) / len(totals))
