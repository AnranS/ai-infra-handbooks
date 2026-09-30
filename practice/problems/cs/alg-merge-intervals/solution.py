def merge(intervals):
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def min_remove(intervals):
    kept, last_end = 0, float("-inf")
    for start, end in sorted(intervals, key=lambda t: t[1]):   # 按终点排序
        if start >= last_end:
            kept += 1
            last_end = end
    return len(intervals) - kept


def max_concurrent(intervals):
    events = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    events.sort(key=lambda t: (t[0], t[1]))    # 同一时刻先处理结束（-1 < 1）
    cur = best = 0
    for _, delta in events:
        cur += delta
        best = max(best, cur)
    return best
