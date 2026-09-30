def round_robin(i, n):
    return i % n


def least_conn(inflight):
    return min(range(len(inflight)), key=lambda k: (inflight[k], k))


def p2c(rng, inflight):
    a, b = rng.randrange(len(inflight)), rng.randrange(len(inflight))
    return b if inflight[b] < inflight[a] else a


def simulate(policy, costs, n, seed=0):
    import random
    rng = random.Random(seed)
    busy_until = [0.0] * n
    queued = [[] for _ in range(n)]                  # 每台机器上还没做完的请求的完成时刻
    waits = []
    for i, cost in enumerate(costs):
        now = float(i)                               # 每毫秒到达一个请求
        for q in queued:
            q[:] = [t for t in q if t > now]         # 已经做完的不算在负载里
        inflight = [len(q) for q in queued]
        if policy == "rr":
            s = round_robin(i, n)
        elif policy == "lc":
            s = least_conn(inflight)
        else:
            s = p2c(rng, inflight)
        start = max(now, busy_until[s])              # 一台机器一次只处理一个请求
        busy_until[s] = start + cost
        queued[s].append(busy_until[s])
        waits.append(start - now)
    return waits
