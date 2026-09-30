from collections import deque


def shortest_path(grid, start, end):
    if not grid or not grid[0]:
        return -1
    rows, cols = len(grid), len(grid[0])
    sr, sc = start
    er, ec = end
    if grid[sr][sc] == 1 or grid[er][ec] == 1:
        return -1
    if start == end:
        return 0
    seen = {start}
    q = deque([(sr, sc, 0)])
    while q:
        r, c, d = q.popleft()
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if not (0 <= nr < rows and 0 <= nc < cols) or grid[nr][nc] == 1 or (nr, nc) in seen:
                continue
            if (nr, nc) == end:
                return d + 1
            seen.add((nr, nc))                 # 入队时标记
            q.append((nr, nc, d + 1))
    return -1


def rotting_time(grid):
    rows, cols = len(grid), len(grid[0]) if grid else 0
    q = deque((r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 2)
    fresh = sum(row.count(1) for row in grid)
    seen = {(r, c) for r, c in q}
    minutes = 0
    while q and fresh:
        minutes += 1
        for _ in range(len(q)):                # 一次处理一整层 = 一分钟
            r, c = q.popleft()
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1 and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    fresh -= 1
                    q.append((nr, nc))
    return -1 if fresh else minutes
