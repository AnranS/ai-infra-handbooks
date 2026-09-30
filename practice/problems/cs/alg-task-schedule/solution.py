import heapq
from math import ceil


def _simulate(tasks, machines):
    if machines <= 0:
        return float("inf")
    heap = [0] * machines
    for cost in tasks:
        free_at = heapq.heappop(heap)          # 最早空闲的机器
        heapq.heappush(heap, free_at + cost)
    return max(heap) if heap else 0


def finish_time(tasks, machines):
    return _simulate(tasks, machines)


def lpt_finish(tasks, machines):
    return _simulate(sorted(tasks, reverse=True), machines)


def min_machines(tasks, deadline):
    if not tasks:
        return 0
    if max(tasks) > deadline:
        return -1
    m = max(1, ceil(sum(tasks) / deadline))    # 理论下界
    while lpt_finish(tasks, m) > deadline:
        m += 1
    return m
