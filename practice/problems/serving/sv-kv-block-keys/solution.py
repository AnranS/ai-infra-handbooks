def block_keys(tokens, block, seed=0):
    keys, h = [], seed
    for i in range(0, len(tokens) - len(tokens) % block, block):
        h = hash((h, tuple(tokens[i:i + block])))
        keys.append(h)
    return keys


def longest_hit(keys, tiers):
    hits = []
    for k in keys:
        where = next((t for t, tier in enumerate(tiers) if k in tier), None)
        if where is None:
            break
        hits.append(where)
    return hits


def plan_load(keys, tiers, block, cost_per_tier, recompute_cost):
    loaded = 0
    for t in longest_hit(keys, tiers):
        if cost_per_tier[t] >= recompute_cost:
            break
        loaded += block
    return loaded, len(keys) * block - loaded
