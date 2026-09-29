import bisect


def pad_to(bs, sizes):
    i = bisect.bisect_left(sizes, bs)
    return sizes[i] if i < len(sizes) else None


def expected_waste(dist, sizes):
    total = sum(dist.values())
    if not total:
        return 0.0
    waste = 0
    for b, c in dist.items():
        p = pad_to(b, sizes)
        if p is not None:
            waste += c * (p - b)
    return waste / total


def best_sizes(dist, max_bs, k):
    n = max_bs
    cnt = [0] * (n + 1)
    for b, c in dist.items():
        if 1 <= b <= n:
            cnt[b] += c
    pc, pb = [0] * (n + 1), [0] * (n + 1)
    for b in range(1, n + 1):
        pc[b], pb[b] = pc[b - 1] + cnt[b], pb[b - 1] + cnt[b] * b

    def cost(t, s):                      # (t, s] 里的批次补齐到 s
        return s * (pc[s] - pc[t]) - (pb[s] - pb[t])

    INF = float("inf")
    # best[j][s]：选了 j 个、最大的是 s 时的 (代价, 方案)
    best = [[(INF, None)] * (n + 1) for _ in range(k + 1)]
    for s in range(1, n + 1):
        best[1][s] = (cost(0, s), [s])
    for j in range(2, k + 1):
        for s in range(j, n + 1):
            cand = (INF, None)
            for t in range(j - 1, s):
                c0, plan = best[j - 1][t]
                if plan is None:
                    continue
                c = c0 + cost(t, s)
                p = plan + [s]
                if c < cand[0] or (c == cand[0] and p < cand[1]):
                    cand = (c, p)
            best[j][s] = cand
    return best[k][n][1]
