def word_break(s, words):
    if not s:
        return True
    ws = set(words)
    if not ws:
        return False
    maxlen = max(len(w) for w in ws)
    f = [False] * (len(s) + 1)
    f[0] = True
    for i in range(1, len(s) + 1):
        for j in range(max(0, i - maxlen), i):
            if f[j] and s[j:i] in ws:
                f[i] = True
                break
    return f[len(s)]


def min_cuts(s, words):
    if not s:
        return 0
    ws = set(words)
    if not ws:
        return -1
    maxlen = max(len(w) for w in ws)
    INF = float("inf")
    g = [0] + [INF] * len(s)
    for i in range(1, len(s) + 1):
        for j in range(max(0, i - maxlen), i):
            if g[j] + 1 < g[i] and s[j:i] in ws:
                g[i] = g[j] + 1
    return -1 if g[len(s)] == INF else g[len(s)]
