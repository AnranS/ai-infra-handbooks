def unique_paths(m, n):
    f = [1] * n
    for _ in range(m):                         # 多算了一行
        for j in range(1, n):
            f[j] += f[j - 1]
    return f[n - 1]


def paths_with_obstacles(grid):
    n = len(grid[0])
    f = [0] * n
    f[0] = 1
    for row in grid:
        for j in range(n):
            if j > 0:
                f[j] += f[j - 1]               # 没处理障碍
    return f[n - 1]


def min_path_sum(grid):
    n = len(grid[0])
    f = [0] * n                                # 第一行没有按前缀和初始化
    for row in grid:
        for j in range(n):
            f[j] = min(f[j], f[j - 1] if j else f[j]) + row[j]
    return f[n - 1]
