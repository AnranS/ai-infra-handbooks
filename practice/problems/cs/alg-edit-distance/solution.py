def edit_distance(a, b):
    m, n = len(a), len(b)
    f = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        f[i][0] = i                            # 全删
    for j in range(n + 1):
        f[0][j] = j                            # 全插
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                f[i][j] = f[i - 1][j - 1]
            else:
                f[i][j] = 1 + min(f[i - 1][j], f[i][j - 1], f[i - 1][j - 1])
    return f[m][n]


def lcs(a, b):
    m, n = len(a), len(b)
    f = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                f[i][j] = f[i - 1][j - 1] + 1
            else:
                f[i][j] = max(f[i - 1][j], f[i][j - 1])
    return f[m][n]
