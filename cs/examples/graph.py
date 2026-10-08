# 图的三个基本算法：BFS 最短路（无权）、拓扑排序、并查集
from collections import defaultdict, deque

# 一个小的依赖图：算子 -> 依赖它的算子（计算图的样子）
edges = [("embed", "attn"), ("attn", "norm1"), ("norm1", "mlp"), ("mlp", "norm2"),
         ("embed", "residual"), ("residual", "norm2"), ("norm2", "logits")]
graph = defaultdict(list)
indeg = defaultdict(int)
nodes = set()
for a, b in edges:
    graph[a].append(b)
    indeg[b] += 1
    nodes |= {a, b}


def bfs_dist(start):
    dist = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt in graph[cur]:
            if nxt not in dist:                # 第一次访问就是最短距离
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return dist


def topo_sort():
    deg = {n: indeg[n] for n in nodes}
    q = deque(sorted(n for n in nodes if deg[n] == 0))   # 排序只为输出稳定
    out = []
    while q:
        cur = q.popleft()
        out.append(cur)
        for nxt in graph[cur]:
            deg[nxt] -= 1
            if deg[nxt] == 0:
                q.append(nxt)
    return out if len(out) == len(nodes) else None       # 少了节点说明有环


class DSU:
    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.size = {x: 1 for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # 路径压缩
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:                  # 按大小合并
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


print("从 embed 出发的层数：", dict(sorted(bfs_dist("embed").items())))
print("拓扑序（一种合法的执行顺序）：", topo_sort())

graph["logits"].append("embed")                # 加一条回边，制造环
indeg["embed"] += 1
print("加一条 logits -> embed 之后：", topo_sort(), "（None 表示有环，计算图非法）")
graph["logits"].pop()
indeg["embed"] -= 1

dsu = DSU(nodes)
merged = [dsu.union(a, b) for a, b in edges]
print("\n并查集：把所有有依赖关系的算子并起来后，连通块个数 =",
      len({dsu.find(n) for n in nodes}), "，合并成功的边数 =", sum(merged), "（其余是成环的边）")
