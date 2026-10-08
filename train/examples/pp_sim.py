def schedule(kind, p, m):
    """每个 stage 按什么顺序执行前向 (F, i) 和反向 (B, i)"""
    orders = []
    for s in range(p):
        if kind == "gpipe":
            orders.append([("F", i) for i in range(m)] + [("B", i) for i in range(m)])
        else:                                   # 1F1B：先做 p-s-1 个前向"预热"，之后一前一后交替
            warm = min(p - s - 1, m)
            order = [("F", i) for i in range(warm)]
            for i in range(m - warm):
                order += [("F", warm + i), ("B", i)]
            order += [("B", i) for i in range(m - warm, m)]
            orders.append(order)
    return orders


def simulate(kind, p, m, tf=1, tb=2):
    orders, done, t_free = schedule(kind, p, m), {}, [0] * p
    pos, rows = [0] * p, [[] for _ in range(p)]
    live, peak = [0] * p, [0] * p
    while any(pos[s] < len(orders[s]) for s in range(p)):
        for s in range(p):
            if pos[s] == len(orders[s]):
                continue
            op, i = orders[s][pos[s]]
            dep = ("F", s - 1, i) if op == "F" else (("B", s + 1, i) if s < p - 1 else ("F", s, i))
            if op == "F" and s == 0:
                dep = None
            if dep is not None and dep not in done:
                continue
            start = max(t_free[s], done.get(dep, 0))
            end = start + (tf if op == "F" else tb)
            rows[s] += ["."] * (start - len(rows[s])) + [str(i) if op == "F" else chr(ord("a") + i)] * (end - start)
            done[(op, s, i)], t_free[s] = end, end
            live[s] += 1 if op == "F" else -1
            peak[s] = max(peak[s], live[s])
            pos[s] += 1
    total = max(t_free)
    busy = p * m * (tf + tb)
    return rows, total, 1 - busy / (p * total), peak


for kind in ("gpipe", "1f1b"):
    rows, total, bubble, peak = simulate(kind, p=4, m=8)
    print(f"{kind}：总时间 {total}，气泡占比 {bubble:.1%}，每个 stage 同时保存的激活份数 {peak}")
    for s, r in enumerate(rows):
        print(f"  stage {s} |{''.join(r).ljust(total, '.')}|")
print("理论气泡占比 (p-1)/(m+p-1) =", f"{3 / 11:.1%}")
