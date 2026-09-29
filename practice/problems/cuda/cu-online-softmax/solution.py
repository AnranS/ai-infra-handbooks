import math

import gpusim as gs

BLOCK = 128


def combine(m1, s1, m2, s2):
    m = max(m1, m2)
    if m == -math.inf:
        return m, 0.0
    return m, s1 * math.exp(m1 - m) + s2 * math.exp(m2 - m)


def warp_combine(t, m, s):
    off = 16
    while off > 0:
        m2 = yield t.shfl_down(m, off)
        s2 = yield t.shfl_down(s, off)
        m, s = combine(m, s, m2, s2)
        off //= 2
    return m, s


@gs.kernel
def softmax_rows(t, x, y, rows, cols):
    sm = t.shared("sm", 32)
    ss = t.shared("ss", 32)
    r, tid = t.blockIdx.x, t.threadIdx.x
    m, s = -math.inf, 0.0
    for c in range(tid, cols, BLOCK):              # 第一遍：online softmax
        v = float(x[r, c])
        m, s = combine(m, s, v, 1.0)
    m, s = yield from warp_combine(t, m, s)
    if t.lane == 0:
        sm[t.warp_id], ss[t.warp_id] = m, s
    yield t.syncthreads()
    if t.warp_id == 0:
        nw = BLOCK // 32
        m, s = (float(sm[t.lane]), float(ss[t.lane])) if t.lane < nw else (-math.inf, 0.0)
        m, s = yield from warp_combine(t, m, s)
    yield t.syncthreads()                          # 第 0 个 warp 读完 sm/ss 之后才能覆盖
    if tid == 0:
        sm[0], ss[0] = m, s
    yield t.syncthreads()
    m, s = float(sm[0]), float(ss[0])
    for c in range(tid, cols, BLOCK):              # 第二遍：写结果
        y[r, c] = math.exp(float(x[r, c]) - m) / s
