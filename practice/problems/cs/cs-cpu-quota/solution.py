def usable_cpus(affinity, quota_us, period_us):
    if quota_us == -1:
        return float(affinity)
    return min(float(affinity), quota_us / period_us)


def finish_ms(threads, work_ms, quota_ms, period_ms, cores):
    left = [float(work_ms)] * threads              # 每个线程剩下的工作量（大家进度一样，按平均分配）
    t, eps = 0.0, 1e-9
    while True:
        period_end = (int(t // period_ms + eps) + 1) * period_ms
        budget = float("inf") if quota_ms is None else quota_ms
        while budget > eps:
            running = [i for i, x in enumerate(left) if x > eps]
            if not running:
                return round(t, 6)
            share = min(len(running), cores) / len(running)   # 每个线程每毫秒能跑多少毫秒
            rate = share * len(running)                        # 整个容器每毫秒消耗多少配额
            dt = min(min(left[i] for i in running) / share,    # 有线程做完
                     budget / rate,                             # 配额用光
                     period_end - t)                            # 周期结束
            for i in running:
                left[i] -= dt * share
            budget -= dt * rate
            t += dt
            if t >= period_end - eps:
                break
        if not any(x > eps for x in left):
            return round(t, 6)
        t = period_end if t < period_end - eps else t          # 配额用光：等到下一个周期
