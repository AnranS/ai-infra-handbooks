from collections import Counter


def exist(board, word):
    if not word:
        return True
    if not board or not board[0]:
        return False
    rows, cols = len(board), len(board[0])
    if len(word) > rows * cols:
        return False
    counts = Counter(ch for row in board for ch in row)
    need = Counter(word)
    if any(counts[ch] < n for ch, n in need.items()):      # 字符不够，直接判否
        return False
    if counts[word[-1]] < counts[word[0]]:                 # 从更稀有的一端搜
        word = word[::-1]
    seen = set()

    def dfs(r, c, i):
        if board[r][c] != word[i]:
            return False
        if i == len(word) - 1:
            return True
        seen.add((r, c))
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen and dfs(nr, nc, i + 1):
                seen.discard((r, c))
                return True
        seen.discard((r, c))                               # 撤销标记
        return False

    return any(dfs(r, c, 0) for r in range(rows) for c in range(cols))
