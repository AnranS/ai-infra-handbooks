from collections import defaultdict, deque


def dijkstra(n, edges, start):
    graph = defaultdict(list)
    for a, b, w in edges:
        graph[a].append((b, w))
    INF = float("inf")
    dist = [INF] * n
    dist[start] = 0
    q = deque([start])
    while q:                                   # 用 BFS 代替堆：带权图会算错
        node = q.popleft()
        for nxt, w in graph[node]:
            if dist[node] + w < dist[nxt]:
                dist[nxt] = dist[node] + w
                q.append(nxt)
    return [None if d == INF else d for d in dist]


def network_delay(n, edges, start):
    dist = dijkstra(n, edges, start)
    return max(d for d in dist if d is not None)   # 不可达时应该返回 -1
