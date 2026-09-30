import heapq
from collections import defaultdict


def find_order(n, prerequisites):
    graph = defaultdict(list)
    indeg = [0] * n
    for a, b in prerequisites:
        graph[a].append(b)                     # 边的方向反了
        indeg[b] += 1
    heap = [i for i in range(n) if indeg[i] == 0]
    heapq.heapify(heap)
    out = []
    while heap:
        cur = heapq.heappop(heap)
        out.append(cur)
        for nxt in graph[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                heapq.heappush(heap, nxt)
    return out


def can_finish(n, prerequisites):
    return len(find_order(n, prerequisites)) == n
