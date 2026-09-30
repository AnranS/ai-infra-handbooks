def usable_cpus(affinity, quota_us, period_us):
    if quota_us == -1:
        return float(affinity)
    return min(float(affinity), quota_us / period_us)


def finish_ms(threads, work_ms, quota_ms, period_ms, cores):
    parallel = min(threads, cores)
    if quota_ms is not None:
        parallel = min(parallel, quota_ms / period_ms)     # 把配额当成"平均能用几个核"
    return threads * work_ms / parallel
