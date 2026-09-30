class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.sz = [1] * n
        self.groups = n

    def find(self, x):
        if self.parent[x] == x:
            return x
        return self.find(self.parent[x])                   # 没有路径压缩，深链会爆栈

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb                               # 不按大小合并：树会长成一条链
        self.sz[rb] += self.sz[ra]
        self.groups -= 1
        return True

    def count(self):
        return self.groups

    def size(self, x):
        return self.sz[x]                                  # 没有取根：非代表元的大小是错的


def count_components(n, edges):
    d = DSU(n)
    for a, b in edges:
        d.union(a, b)
    return d.count()


def has_cycle(n, edges):
    d = DSU(n)
    return any(not d.union(a, b) for a, b in edges)
