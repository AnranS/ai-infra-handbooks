def unique_paths(m, n):
    if m <= 0 or n <= 0:
        return 0
    f = [1] * n                                # 第一行都是 1
    for _ in range(1, m):
        for j in range(1, n):
            f[j] += f[j - 1]                   # 上面（旧值）+ 左边（新值）
    return f[n - 1]


def paths_with_obstacles(grid):
    if not grid or not grid[0] or grid[0][0] == 1:
        return 0
    n = len(grid[0])
    f = [0] * n
    f[0] = 1
    for row in grid:
        for j in range(n):
            if row[j] == 1:
                f[j] = 0                       # 障碍：走不通
            elif j > 0:
                f[j] += f[j - 1]
    return f[n - 1]


def min_path_sum(grid):
    if not grid or not grid[0]:
        return 0
    n = len(grid[0])
    INF = float("inf")
    f = [INF] * n
    f[0] = 0
    for row in grid:
        f[0] += row[0]
        for j in range(1, n):
            f[j] = min(f[j], f[j - 1]) + row[j]
    return f[n - 1]
