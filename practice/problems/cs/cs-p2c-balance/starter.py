def round_robin(i, n):
    return i % n


def least_conn(inflight):
    return inflight.index(min(inflight))


def p2c(rng, inflight):
    a, b = rng.randrange(len(inflight)), rng.randrange(len(inflight))
    return a if inflight[a] < inflight[b] else b     # 相等时选了 b，和题目要求相反


def simulate(policy, costs, n, seed=0):
    import random
    rng = random.Random(seed)
    busy_until = [0.0] * n
    queued = [[] for _ in range(n)]
    waits = []
    for i, cost in enumerate(costs):
        now = float(i)
        inflight = [len(q) for q in queued]          # 忘了把做完的请求去掉：负载只增不减
        if policy == "rr":
            s = round_robin(i, n)
        elif policy == "lc":
            s = least_conn(inflight)
        else:
            s = p2c(rng, inflight)
        start = max(now, busy_until[s])
        busy_until[s] = start + cost
        queued[s].append(busy_until[s])
        waits.append(start - now)
    return waits
