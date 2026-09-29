import math

import gpusim as gs

BLOCK = 128


def combine(m1, s1, m2, s2):
    m = max(m1, m2)
    if m == -math.inf:
        return m, 0.0
    return m, s1 * math.exp(m1 - m) + s2 * math.exp(m2 - m)


@gs.kernel
def softmax_rows(t, x, y, rows, cols):
    # 三遍法（每遍都只由线程 0 串行完成）：结果对，但读了三遍输入，也没有并行
    r = t.blockIdx.x
    if t.threadIdx.x == 0:
        m = -math.inf
        for c in range(cols):
            m = max(m, float(x[r, c]))
        s = 0.0
        for c in range(cols):
            s += math.exp(float(x[r, c]) - m)
        for c in range(cols):
            y[r, c] = math.exp(float(x[r, c]) - m) / s
