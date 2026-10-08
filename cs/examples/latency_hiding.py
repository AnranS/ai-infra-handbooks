# 一个 warp 调度器每周期最多发射一条指令。每个 warp 反复做：连续发出 ilp 条互不依赖的访存，
# 等数据回来（latency 个周期），再发出 compute 条计算指令。模拟调度器有多少比例的周期在发射指令。
# 调度策略是"贪心再取最老"（GTO）：一直发射同一个 warp，直到它要等数据，再换编号最小的就绪 warp。
# 只统计稳定阶段：有任何一个 warp 做完 iters 轮就停止
def simulate(warps, ilp, compute, latency=500, iters=100):
    phase = ["load"] * warps                 # load：发访存；wait：等数据；compute：发计算
    left = [ilp] * warps                     # 本阶段还剩几条指令
    ready_at = [0] * warps                   # 数据到达的周期
    done = [0] * warps                       # 已完成几轮
    cycle = busy = 0
    cur = 0
    while max(done) < iters:
        def ready(w):
            return phase[w] != "wait" or cycle >= ready_at[w]
        if not ready(cur):
            cur = next((w for w in range(warps) if ready(w)), None)
        if cur is None:                      # 所有 warp 都在等数据：这个周期空转
            cycle, cur = cycle + 1, 0
            continue
        w = cur
        if phase[w] == "wait":
            phase[w], left[w] = "compute", compute
        left[w] -= 1                         # 发射一条指令
        busy += 1
        if phase[w] == "load" and left[w] == 0:
            phase[w], ready_at[w] = "wait", cycle + latency
        elif phase[w] == "compute" and left[w] == 0:
            phase[w], left[w], done[w] = "load", ilp, done[w] + 1
        cycle += 1
    return busy / cycle


print("每个调度器上的 warp 数       1     2     4     8    16")
for ilp, compute in [(1, 16), (4, 64)]:
    row = [f"{simulate(w, ilp, compute):5.0%}" for w in (1, 2, 4, 8, 16)]
    print(f"每轮 {ilp} 条访存、{compute:2d} 条计算：" + " ".join(row))
