def schedule(K, L, C, S):
    out, t = [], 0
    for k in range(K):                       # 没有流水：拷完一块算一块
        out.append((t, t + L, t + L, t + L + C))
        t += L + C
    return out


def pipeline_time(K, L, C, S):
    s = schedule(K, L, C, S)
    return s[-1][3] if s else 0
