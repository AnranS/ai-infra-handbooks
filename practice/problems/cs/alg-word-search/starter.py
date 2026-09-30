def exist(board, word):
    if not word:
        return True
    rows, cols = len(board), len(board[0])
    seen = set()

    def dfs(r, c, i):
        if board[r][c] != word[i]:
            return False
        if i == len(word) - 1:
            return True
        seen.add((r, c))
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen and dfs(nr, nc, i + 1):
                return True
        return False                                       # 忘了撤销标记：走过的格子永久不可用

    return any(dfs(r, c, 0) for r in range(rows) for c in range(cols))
