from checker import check
from solution import max_area, num_islands


def test_example():
    g = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    check(num_islands(g), 2, "两座岛")
    check(max_area(g), 3, "最大面积")


def test_edges():
    check(num_islands([]), 0, "空网格")
    check(num_islands([[]]), 0, "空行")
    check(num_islands([[0]]), 0, "全是水")
    check(max_area([[0, 0], [0, 0]]), 0, "没有岛屿时返回 0")
    check(num_islands([[1]]), 1, "一个格子")


def test_all_land():
    g = [[1, 1], [1, 1]]
    check(num_islands(g), 1, "连成一片")
    check(max_area(g), 4, "面积是全部")


def test_diagonal_not_connected():
    g = [[1, 0], [0, 1]]
    check(num_islands(g), 2, "对角线不算相邻")


def test_does_not_modify():
    g = [[1, 1], [0, 1]]
    copy = [row[:] for row in g]
    num_islands(g)
    max_area(g)
    check(g, copy, "不能修改传入的网格")


def test_snake():
    g = [[1, 1, 1, 1, 1],
         [0, 0, 0, 0, 1],
         [1, 1, 1, 0, 1],
         [1, 0, 1, 0, 1],
         [1, 0, 1, 1, 1]]
    check(num_islands(g), 1, "蛇形连通")
    check(max_area(g), 17, "面积")


def test_large():
    n = 150
    g = [[1] * n for _ in range(n)]            # 全是陆地：递归写法会爆栈
    check(num_islands(g), 1, "一大片陆地")
    check(max_area(g), n * n, "面积是整张图")
