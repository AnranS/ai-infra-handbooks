# 循环变换：同一个矩阵乘，换一种循环顺序、加一层分块，访存量差几十倍
def traffic(m, n, k, cache_lines, tile=None, order="ijk"):
    """用一个全相联 LRU 缓存模拟这段矩阵乘要从内存搬多少条缓存行（每行装 8 个 float64）"""
    from collections import OrderedDict
    cache, misses = OrderedDict(), 0

    def touch(addr):
        nonlocal misses
        line = addr // 8
        if line in cache:
            cache.move_to_end(line)
            return
        misses += 1
        if len(cache) == cache_lines:
            cache.popitem(last=False)
        cache[line] = True

    def body(i, j, p):
        touch(0 + i * k + p)                   # A[i][p]
        touch(m * k + p * n + j)               # B[p][j]
        touch(m * k + k * n + i * n + j)       # C[i][j]

    if tile is None:
        loops = {"ijk": lambda: ((i, j, p) for i in range(m) for j in range(n) for p in range(k)),
                 "ikj": lambda: ((i, j, p) for i in range(m) for p in range(k) for j in range(n))}[order]
        for i, j, p in loops():
            body(i, j, p)
    else:
        for i0 in range(0, m, tile):           # 分块：先把一小块的数据全用完再换
            for j0 in range(0, n, tile):
                for p0 in range(0, k, tile):
                    for i in range(i0, min(i0 + tile, m)):
                        for p in range(p0, min(p0 + tile, k)):
                            for j in range(j0, min(j0 + tile, n)):
                                body(i, j, p)
    return misses


n = 48
lines = 16                                     # 很小的缓存：16 条行 = 128 个元素
print(f"{n}x{n} 的矩阵乘，缓存只有 {lines} 条行（{lines * 8} 个元素）")
base = traffic(n, n, n, lines, order="ijk")
for name, misses in [("i,j,p 顺序（B 按列访问）", base),
                     ("i,p,j 顺序（B 按行访问）", traffic(n, n, n, lines, order="ikj")),
                     ("分块 8x8", traffic(n, n, n, lines, tile=8)),
                     ("分块 16x16", traffic(n, n, n, lines, tile=16))]:
    print(f"  {name:24s} 缓存缺失 {misses:7d} 次，相对第一种 {misses / base:5.2f}x")
print()
print("循环交换让 B 变成按行访问（连续），分块让一小块数据留在缓存里被反复使用。")
print("注意 16x16 反而比 8x8 差：块太大，一块的工作集装不进这个缓存了。")
print("块大小要和缓存（或共享内存、寄存器）的容量匹配——这正是自动调优要搜的参数。")
