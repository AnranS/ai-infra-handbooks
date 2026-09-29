def ring_allreduce(data):
    n = len(data)
    size = len(data[0]) // n
    buf = [list(x) for x in data]
    sent = [0] * n

    def block(i):
        return slice(i * size, (i + 1) * size)

    for k in range(n - 1):                        # reduce-scatter
        msgs = [(r, (r - k) % n, buf[r][block((r - k) % n)]) for r in range(n)]   # 切片是拷贝：先把这一步要发的都取出来
        for r, i, vals in msgs:
            dst = buf[(r + 1) % n]
            dst[block(i)] = [a + b for a, b in zip(dst[block(i)], vals)]
            sent[r] += len(vals)
    for k in range(n - 1):                        # all-gather
        msgs = [(r, (r + 1 - k) % n, buf[r][block((r + 1 - k) % n)]) for r in range(n)]
        for r, i, vals in msgs:
            buf[(r + 1) % n][block(i)] = vals
            sent[r] += len(vals)
    return buf, sent
