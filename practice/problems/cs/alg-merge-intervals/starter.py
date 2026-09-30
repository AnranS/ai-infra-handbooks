def merge(intervals):
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = end                   # 直接覆盖：被包含的区间会把右端点改小
        else:
            out.append([start, end])
    return out


def min_remove(intervals):
    kept, last_end = 0, float("-inf")
    for start, end in sorted(intervals):       # 按起点排序：长区间会挡住后面的
        if start >= last_end:
            kept += 1
            last_end = end
    return len(intervals) - kept


def max_concurrent(intervals):
    events = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    events.sort(key=lambda t: (t[0], -t[1]))   # 同一时刻先处理开始：会多算一个
    cur = best = 0
    for _, delta in events:
        cur += delta
        best = max(best, cur)
    return best
