def schedule_1f1b(p, m):
    plans = []
    for s in range(p):
        warmup = min(p - s - 1, m)
        ops = [("F", i) for i in range(warmup)]
        for i in range(m - warmup):                   # 稳定阶段：一个前向、一个反向交替
            ops += [("F", warmup + i), ("B", i)]
        ops += [("B", i) for i in range(m - warmup, m)]
        plans.append(ops)
    return plans


def simulate(p, m, plans, tf=1.0, tb=2.0):
    done = {}                                          # (阶段, 类型, micro-batch) -> 结束时刻
    free = [0.0] * p
    pos = [0] * p
    total = sum(len(x) for x in plans)
    while len(done) < total:
        progress = False
        for s in range(p):
            while pos[s] < len(plans[s]):
                kind, i = plans[s][pos[s]]
                if kind == "F":
                    deps = [(s - 1, "F", i)] if s > 0 else []
                else:
                    deps = [(s, "F", i)] + ([(s + 1, "B", i)] if s < p - 1 else [])
                if any(d not in done for d in deps):
                    break
                start = max([free[s]] + [done[d] for d in deps])
                free[s] = start + (tf if kind == "F" else tb)
                done[(s, kind, i)] = free[s]
                pos[s] += 1
                progress = True
        if not progress:
            raise RuntimeError("死锁：每个阶段的下一个操作都在等待别人")
    return max(free)


def peak_activations(plans):
    peaks = []
    for ops in plans:
        live = peak = 0
        for kind, _ in ops:
            live += 1 if kind == "F" else -1
            peak = max(peak, live)
        peaks.append(peak)
    return peaks
