def streaming_keep(n, sink, window):
    if n <= sink + window:
        return list(range(n))
    return list(range(sink)) + list(range(n - window, n))


def h2o_keep(acc, budget, recent):
    n = len(acc)
    if n <= budget:
        return list(range(n))
    keep = set(range(n - recent, n))
    others = sorted((i for i in range(n - recent)), key=lambda i: (-acc[i], i))
    keep.update(others[:budget - recent])
    return sorted(keep)


def simulate_h2o(attn_steps, budget, recent):
    cache, score = [], {}
    for t, row in enumerate(attn_steps):
        cache.append(t)
        score[t] = 0.0
        for pos, w in row.items():
            score[pos] += w
        if len(cache) > budget:
            kept = h2o_keep([score[p] for p in cache], budget, recent)
            new_cache = [cache[i] for i in kept]
            for p in set(cache) - set(new_cache):
                del score[p]
            cache = new_cache
    return cache
