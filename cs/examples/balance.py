# 四种负载均衡策略在"服务时间长尾"下的表现：轮询、随机、最少连接、二选一（P2C）
import heapq
import random


def simulate(policy, n_servers=8, n_reqs=20000, seed=7):
    rng = random.Random(seed)
    busy_until = [0.0] * n_servers        # 每台服务器忙到什么时候
    inflight = [0] * n_servers            # 正在处理的请求数
    finish = []                           # 已安排的完成事件（时间, 服务器）
    waits = []
    t = 0.0
    for i in range(n_reqs):
        t += rng.expovariate(1 / 1.0)     # 请求到达间隔平均 1 ms
        while finish and finish[0][0] <= t:
            _, s = heapq.heappop(finish)
            inflight[s] -= 1
        if policy == "轮询":
            s = i % n_servers
        elif policy == "随机":
            s = rng.randrange(n_servers)
        elif policy == "最少连接":
            s = min(range(n_servers), key=lambda k: (inflight[k], k))
        else:                             # 二选一：随机挑两台，选正在处理的请求少的那台
            a, b = rng.randrange(n_servers), rng.randrange(n_servers)
            s = a if inflight[a] <= inflight[b] else b
        service = rng.expovariate(1 / 4.0) if rng.random() > 0.05 else rng.expovariate(1 / 40.0)
        start = max(t, busy_until[s])     # 这台机器一次只处理一个请求
        busy_until[s] = start + service
        inflight[s] += 1
        heapq.heappush(finish, (busy_until[s], s))
        waits.append(start - t)           # 排队等待时间
    waits.sort()
    return waits[len(waits) // 2], waits[int(len(waits) * 0.99)]


print("到达 1000 请求/秒，8 台服务器，平均服务 5.8 ms（5% 的慢请求平均 40 ms），利用率约 72%\n")
print("策略      排队 p50   排队 p99")
for policy in ["轮询", "随机", "最少连接", "二选一"]:
    p50, p99 = simulate(policy)
    print(f"{policy:6s} {p50:7.2f} ms {p99:7.1f} ms")
