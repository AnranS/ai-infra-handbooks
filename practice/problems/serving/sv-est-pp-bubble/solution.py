def bubble_fraction(p, m, v=1):
    return (p - 1) / (v * m + p - 1)


def bubble_overhead(p, m, v=1):
    return (p - 1) / (v * m)


def min_microbatches(p, target, v=1):
    m = 1
    while bubble_fraction(p, m, v) > target:
        m += 1
    return m


def peak_inflight(p, m, schedule):
    if schedule == "gpipe":
        return m
    if schedule == "1f1b":
        return min(p, m)
    raise ValueError(f"未知的调度：{schedule}")
