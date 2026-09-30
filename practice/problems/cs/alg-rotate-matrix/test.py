from checker import check
from solution import rotate, spiral


def run_rotate(m):
    rotate(m)
    return m


def test_example():
    check(run_rotate([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), [[7, 4, 1], [8, 5, 2], [9, 6, 3]], "3x3 旋转")
    check(spiral([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), [1, 2, 3, 6, 9, 8, 7, 4, 5], "螺旋")


def test_rotate_edges():
    check(run_rotate([]), [], "空矩阵")
    check(run_rotate([[1]]), [[1]], "1x1")
    check(run_rotate([[1, 2], [3, 4]]), [[3, 1], [4, 2]], "2x2")


def test_rotate_four_times():
    m = [[1, 2], [3, 4]]
    original = [row[:] for row in m]
    for _ in range(4):
        rotate(m)
    check(m, original, "转四次回到原样")


def test_rotate_in_place():
    m = [[1, 2], [3, 4]]
    before = id(m)
    rotate(m)
    check(id(m), before, "必须原地")


def test_spiral_edges():
    check(spiral([]), [], "空")
    check(spiral([[]]), [], "空行")
    check(spiral([[1]]), [1], "单元素")
    check(spiral([[1, 2, 3]]), [1, 2, 3], "单行")
    check(spiral([[1], [2], [3]]), [1, 2, 3], "单列")


def test_spiral_rectangle():
    check(spiral([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
          [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7], "3x4")
    check(spiral([[1, 2], [3, 4], [5, 6]]), [1, 2, 4, 6, 5, 3], "3x2")


def test_large():
    n = 200
    m = [[i * n + j for j in range(n)] for i in range(n)]
    got = spiral(m)
    check(len(got), n * n, "全部元素都遍历到")
    check(len(set(got)), n * n, "没有重复")
    check(got[0], 0, "从左上角开始")
    rotate(m)
    check(m[0][0], (n - 1) * n, "旋转后左上角是原来的左下角")
