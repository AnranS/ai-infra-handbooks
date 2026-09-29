import numpy as np


def a_coord(lane, i):
    g, t = lane // 4, lane % 4
    return g + 8 * ((i // 2) % 2), 2 * t + (i % 2) + 8 * (i // 4)


def b_coord(lane, i):
    g, t = lane // 4, lane % 4
    return 2 * t + (i % 2) + 8 * (i // 2), g


def c_coord(lane, i):
    g, t = lane // 4, lane % 4
    return g + 8 * (i // 2), 2 * t + (i % 2)


def load_fragments(A, B, C):
    fa = np.array([[A[a_coord(l, i)] for i in range(8)] for l in range(32)])
    fb = np.array([[B[b_coord(l, i)] for i in range(4)] for l in range(32)])
    fc = np.array([[C[c_coord(l, i)] for i in range(4)] for l in range(32)])
    return fa, fb, fc


def _gather(frag, coord, shape):
    M = np.zeros(shape, dtype=np.float64)
    for l in range(32):
        for i in range(frag.shape[1]):
            M[coord(l, i)] = frag[l, i]
    return M


def mma_sync(fa, fb, fc):
    A = _gather(fa, a_coord, (16, 16))
    B = _gather(fb, b_coord, (16, 8))
    C = _gather(fc, c_coord, (16, 8))
    D = A @ B + C
    return np.array([[D[c_coord(l, i)] for i in range(4)] for l in range(32)])


def store_fragment(fd):
    return _gather(fd, c_coord, (16, 8))


def lanes_holding_row_of_c(r):
    return sorted({l for l in range(32) for i in range(4) if c_coord(l, i)[0] == r})
