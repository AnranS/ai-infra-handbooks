import heapq
from collections import defaultdict


def dijkstra(n, edges, start):
    graph = defaultdict(list)
    for a, b, w in edges:
        graph[a].append((b, w))
    INF = float("inf")
    dist = [INF] * n
    if 0 <= start < n:
        dist[start] = 0
    heap = [(0, start)] if 0 <= start < n else []
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node]:                     # 惰性删除：这条记录已经过期
            continue
        for nxt, w in graph[node]:
            nd = d + w
            if nd < dist[nxt]:
                dist[nxt] = nd
                heapq.heappush(heap, (nd, nxt))
    return [None if d == INF else d for d in dist]


def network_delay(n, edges, start):
    dist = dijkstra(n, edges, start)
    return -1 if any(d is None for d in dist) else max(dist)
