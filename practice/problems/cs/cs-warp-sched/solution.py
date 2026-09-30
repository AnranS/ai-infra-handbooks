def utilization(warps, ilp, compute, latency, rounds):
    phase = ["load"] * warps                       # load: 发访存；wait: 等数据；compute: 发计算
    left = [ilp] * warps                           # 本阶段还剩几条指令
    ready_at = [0] * warps
    done = [0] * warps
    cycle = busy = 0
    cur = 0
    while max(done) < rounds:
        def ready(w):
            return phase[w] != "wait" or cycle >= ready_at[w]

        if not ready(cur):
            cur = next((w for w in range(warps) if ready(w)), -1)
        if cur < 0:                                # 所有 warp 都在等数据
            cycle, cur = cycle + 1, 0
            continue
        if phase[cur] == "wait":
            phase[cur], left[cur] = "compute", compute
        left[cur] -= 1
        busy += 1
        if phase[cur] == "load" and left[cur] == 0:
            phase[cur], ready_at[cur] = "wait", cycle + latency
        elif phase[cur] == "compute" and left[cur] == 0:
            phase[cur], left[cur], done[cur] = "load", ilp, done[cur] + 1
        cycle += 1
    return busy / cycle
