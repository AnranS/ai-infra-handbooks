class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.sz = [1] * n
        self.groups = n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # 路径压缩
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.sz[ra] < self.sz[rb]:                      # 按大小合并
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.sz[ra] += self.sz[rb]
        self.groups -= 1
        return True

    def count(self):
        return self.groups

    def size(self, x):
        return self.sz[self.find(x)]


def count_components(n, edges):
    d = DSU(n)
    for a, b in edges:
        d.union(a, b)
    return d.count()


def has_cycle(n, edges):
    d = DSU(n)
    return any(not d.union(a, b) for a, b in edges)        # 合并失败说明已经连通
