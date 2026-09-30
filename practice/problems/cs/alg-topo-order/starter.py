from collections import defaultdict, deque


def topo_order(n, edges):
    graph = defaultdict(list)
    indeg = [0] * n
    for a, b in edges:
        graph[a].append(b)
        indeg[b] += 1
    q = deque(i for i in range(n) if indeg[i] == 0)
    out = []
    while q:
        cur = q.popleft()
        out.append(cur)
        for nxt in graph[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)
    return out                                 # 有环时也返回部分结果，没有判环


def longest_path(n, edges):
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
    level = [1] * n
    for a, b in edges:                         # 不按拓扑序：一遍 DP 算不对
        level[b] = max(level[b], level[a] + 1)
    return max(level) if n else 0
