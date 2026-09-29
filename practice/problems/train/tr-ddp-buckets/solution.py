def make_buckets(sizes, cap):
    buckets, cur, total = [], [], 0
    for i in reversed(range(len(sizes))):
        if cur and total + sizes[i] > cap:
            buckets.append(cur)
            cur, total = [], 0
        cur.append(i)
        total += sizes[i]
    if cur:
        buckets.append(cur)
    return buckets


def comm_timeline(ready, buckets, sizes, bw):
    timeline, free = [], 0.0
    for b in buckets:
        start = max(free, max(ready[i] for i in b))
        free = start + sum(sizes[i] for i in b) / bw
        timeline.append((start, free))
    return timeline


def exposed_comm(ready, buckets, sizes, bw):
    end = comm_timeline(ready, buckets, sizes, bw)[-1][1]
    return max(0.0, end - max(ready))
