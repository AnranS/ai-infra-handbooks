from collections import deque


def shortest_path(grid, start, end):
    rows, cols = len(grid), len(grid[0])
    q = deque([(start[0], start[1], 0)])
    seen = set()
    while q:
        r, c, d = q.popleft()
        if (r, c) == end:
            return d
        if (r, c) in seen:                     # 出队时才标记：同一个格子被反复入队
            continue
        seen.add((r, c))
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                q.append((nr, nc, d + 1))
    return -1


def rotting_time(grid):
    rows, cols = len(grid), len(grid[0])
    q = deque((r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 2)
    minutes = 0
    while q:
        minutes += 1                           # 没检查是否还有新鲜的：会多算一分钟
        for _ in range(len(q)):
            r, c = q.popleft()
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                    grid[nr][nc] = 2
                    q.append((nr, nc))
    return minutes
