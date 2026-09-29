import math


def rollout(n, load, cold_s, drain_s, max_unavailable, max_surge):
    step = max_unavailable + max_surge
    if step <= 0:
        raise ValueError("max_unavailable + max_surge must be positive")
    rounds = math.ceil(n / step)
    low = (n - max_unavailable) / n
    return {"rounds": rounds, "minutes": rounds * (cold_s + drain_s) / 60, "min_capacity": low,
            "extra_gpu": max_surge / n, "overloaded": low < load}


def max_safe_unavailable(n, load):
    k = 0
    while k + 1 <= n and (n - (k + 1)) / n >= load:
        k += 1
    return k


def fastest_plan(n, load, cold_s, drain_s, extra_budget):
    best = None
    for surge in range(0, extra_budget + 1):
        for unavail in range(0, max_safe_unavailable(n, load) + 1):
            if unavail + surge == 0:
                continue
            m = rollout(n, load, cold_s, drain_s, unavail, surge)["minutes"]
            key = (m, surge, unavail)
            if best is None or key < best[0]:
                best = (key, (unavail, surge))
    return best[1] if best else None
