import heapq
from collections import defaultdict


def _indeg(n, edges):
    graph = defaultdict(list)
    indeg = [0] * n
    for a, b in edges:
        graph[a].append(b)
        indeg[b] += 1
    return graph, indeg


def topo_order(n, edges):
    graph, indeg = _indeg(n, edges)
    heap = [i for i in range(n) if indeg[i] == 0]
    heapq.heapify(heap)                        # 用堆保证字典序最小
    out = []
    while heap:
        cur = heapq.heappop(heap)
        out.append(cur)
        for nxt in graph[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                heapq.heappush(heap, nxt)
    return out if len(out) == n else None      # 少了节点说明有环


def longest_path(n, edges):
    order = topo_order(n, edges)
    if order is None:
        return -1
    graph, _ = _indeg(n, edges)
    level = [1] * n
    for cur in order:                          # 按拓扑序做 DP
        for nxt in graph[cur]:
            level[nxt] = max(level[nxt], level[cur] + 1)
    return max(level) if n else 0
