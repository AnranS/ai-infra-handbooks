def utilization(warps, ilp, compute, latency, rounds):
    phase = ["load"] * warps
    left = [ilp] * warps
    ready_at = [0] * warps
    done = [0] * warps
    cycle = busy = 0
    cur = 0
    while max(done) < rounds:
        def ready(w):
            return phase[w] != "wait" or cycle >= ready_at[w]

        picked = next((w for w in range((cur + 1) % warps, warps) if ready(w)),
                      next((w for w in range(warps) if ready(w)), -1))   # 每个周期换下一个 warp
        if picked < 0:
            cycle += 1
            continue
        cur = picked
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
