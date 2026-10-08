# CFS 带宽控制：每个周期（100 ms）里，整个容器最多用 quota 毫秒的 CPU 时间，用完就被挂起到下一个周期
PERIOD, QUOTA = 100, 200             # 相当于 limits.cpu = 2


def finish_time(threads, work_ms):
    """threads 个线程同时开工，每个要 work_ms 毫秒 CPU（机器上核足够多），返回全部完成的时刻"""
    left = [work_ms] * threads
    t = 0.0
    while True:
        budget = QUOTA                       # 新周期开始，配额重置
        start = t
        while budget > 1e-9 and any(left):
            running = [i for i, x in enumerate(left) if x > 0]
            step = min(min(left[i] for i in running), budget / len(running))   # 大家并行跑，直到有人跑完或配额用光
            for i in running:
                left[i] -= step
            budget -= step * len(running)
            t += step
        if not any(x > 1e-9 for x in left):
            return t
        t = start + PERIOD                   # 配额用光：整个容器被挂起，直到下一个周期开始


for threads in (1, 4, 8, 16):
    print(f"{threads} 个线程各算 40 ms：不限配额时 40 ms 完成，limits.cpu=2 时 {finish_time(threads, 40):.0f} ms 完成")
