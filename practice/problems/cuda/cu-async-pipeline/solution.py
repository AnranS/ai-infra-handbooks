def schedule(K, L, C, S):
    out = []
    copy_free = 0
    compute_end = []
    for k in range(K):
        request = 0 if k < S else compute_end[k - S]
        cs = max(copy_free, request)
        ce = cs + L
        copy_free = ce
        ps = max(ce, compute_end[k - 1] if k else 0)
        pe = ps + C
        compute_end.append(pe)
        out.append((cs, ce, ps, pe))
    return out


def pipeline_time(K, L, C, S):
    s = schedule(K, L, C, S)
    return s[-1][3] if s else 0
