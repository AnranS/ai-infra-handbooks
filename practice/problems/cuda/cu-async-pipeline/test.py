import heapq

from checker import check
from solution import pipeline_time, schedule


def event_sim(K, L, C, S):
    """独立的离散事件模拟，用来对拍"""
    t = 0
    copy_queue, copy_busy_until, copying = [], 0, None
    copied, computed = set(), {}
    compute_busy_until, computing = 0, None
    for k in range(min(S, K)):
        copy_queue.append(k)
    next_compute = 0
    events = []
    while len(computed) < K:
        if copying is None and copy_queue and t >= copy_busy_until:
            copying = copy_queue.pop(0)
            copy_busy_until = t + L
            heapq.heappush(events, (copy_busy_until, "copy", copying))
        if computing is None and next_compute in copied:
            computing = next_compute
            compute_busy_until = t + C
            heapq.heappush(events, (compute_busy_until, "compute", computing))
        t, kind, k = heapq.heappop(events)
        if kind == "copy":
            copied.add(k)
            copying = None
        else:
            computed[k] = t
            computing = None
            next_compute += 1
            if k + S < K:
                copy_queue.append(k + S)
        while events and events[0][0] == t:
            t2, kind2, k2 = heapq.heappop(events)
            if kind2 == "copy":
                copied.add(k2)
                copying = None
            else:
                computed[k2] = t2
                computing = None
                next_compute += 1
                if k2 + S < K:
                    copy_queue.append(k2 + S)
    return max(computed.values()) if computed else 0


def test_example():
    check(pipeline_time(4, 10, 10, 1), 80, "S=1：没有重叠")
    check(pipeline_time(4, 10, 10, 2), 50, "S=2")
    check(pipeline_time(8, 4, 10, 3), 84, "计算是瓶颈")


def test_schedule_details():
    s = schedule(3, 5, 2, 2)
    check(s, [(0, 5, 5, 7), (5, 10, 10, 12), (10, 15, 15, 17)], "拷贝是瓶颈时的时间线")
    s = schedule(3, 2, 5, 1)
    check(s, [(0, 2, 2, 7), (7, 9, 9, 14), (14, 16, 16, 21)], "S=1 时第 k 块的拷贝要等第 k-1 块算完")


def test_against_event_simulation():
    for K in [1, 2, 5, 9]:
        for L in [1, 3, 7]:
            for C in [1, 4, 6]:
                for S in [1, 2, 3, 4]:
                    check(pipeline_time(K, L, C, S), event_sim(K, L, C, S), f"K={K}, L={L}, C={C}, S={S}")


def test_edge():
    check(pipeline_time(0, 5, 5, 2), 0, "K=0")
    check(len(schedule(6, 1, 1, 10)), 6, "S 比 K 大")
