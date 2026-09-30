from collections import deque


def _scan(grid, visit_island):
    if not grid or not grid[0]:
        return
    rows, cols = len(grid), len(grid[0])
    seen = [[False] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 1 or seen[r][c]:
                continue
            area, q = 0, deque([(r, c)])
            while q:
                x, y = q.popleft()
                if seen[x][y]:                 # 出队时才标记：同一个格子会被重复入队并重复计数
                    continue
                seen[x][y] = True
                area += 1
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 1:
                        q.append((nx, ny))
            visit_island(area)


def num_islands(grid):
    count = []
    _scan(grid, lambda area: count.append(area))
    return len(count)


def max_area(grid):
    areas = []
    _scan(grid, areas.append)
    return max(areas)                          # 没有岛屿时会抛异常
