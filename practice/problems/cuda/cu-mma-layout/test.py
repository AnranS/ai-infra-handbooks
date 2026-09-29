import numpy as np

from checker import check, check_close
from solution import a_coord, b_coord, c_coord, lanes_holding_row_of_c, load_fragments, mma_sync, store_fragment


def test_example():
    check([a_coord(0, i) for i in range(8)], [(0, 0), (0, 1), (8, 0), (8, 1), (0, 8), (0, 9), (8, 8), (8, 9)], "lane 0 的 A 元素")
    check([b_coord(5, i) for i in range(4)], [(2, 1), (3, 1), (10, 1), (11, 1)], "lane 5 的 B 元素")
    check([c_coord(31, i) for i in range(4)], [(7, 6), (7, 7), (15, 6), (15, 7)], "lane 31 的 C 元素")


def test_layouts_are_bijections():
    for name, fn, n, shape in [("A", a_coord, 8, (16, 16)), ("B", b_coord, 4, (16, 8)), ("C", c_coord, 4, (16, 8))]:
        cells = [fn(l, i) for l in range(32) for i in range(n)]
        check(len(set(cells)), shape[0] * shape[1], f"{name} 的每个元素恰好被覆盖一次")
        assert all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells), f"{name} 的坐标越界"


def test_mma_roundtrip():
    rng = np.random.default_rng(0)
    A, B, C = rng.standard_normal((16, 16)), rng.standard_normal((16, 8)), rng.standard_normal((16, 8))
    fa, fb, fc = load_fragments(A, B, C)
    check((fa.shape, fb.shape, fc.shape), ((32, 8), (32, 4), (32, 4)), "片段形状")
    check_close(store_fragment(mma_sync(fa, fb, fc)), A @ B + C, rtol=1e-12, atol=1e-12, what="D = A @ B + C")


def test_fragment_values():
    A = np.arange(256, dtype=np.float64).reshape(16, 16)
    B = np.arange(128, dtype=np.float64).reshape(16, 8)
    fa, fb, fc = load_fragments(A, B, np.zeros((16, 8)))
    check(fa[6].tolist(), [16 * 1 + 4, 16 * 1 + 5, 16 * 9 + 4, 16 * 9 + 5, 16 * 1 + 12, 16 * 1 + 13, 16 * 9 + 12, 16 * 9 + 13],
          "lane 6（g=1, t=2）的 A 片段")
    check(fb[6].tolist(), [8 * 4 + 1, 8 * 5 + 1, 8 * 12 + 1, 8 * 13 + 1], "lane 6 的 B 片段")


def test_rows_of_c():
    check(lanes_holding_row_of_c(0), [0, 1, 2, 3], "C 第 0 行")
    check(lanes_holding_row_of_c(13), [20, 21, 22, 23], "C 第 13 行")
