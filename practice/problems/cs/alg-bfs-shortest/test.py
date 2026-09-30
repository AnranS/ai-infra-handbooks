from checker import check
from solution import rotting_time, shortest_path


def test_example():
    check(shortest_path([[0, 0], [1, 0]], (0, 0), (1, 1)), 2, "绕开障碍")
    check(rotting_time([[2, 1, 1], [1, 1, 0], [0, 1, 1]]), 4, "四分钟")


def test_path_edges():
    check(shortest_path([[0]], (0, 0), (0, 0)), 0, "起点即终点")
    check(shortest_path([[0, 1], [1, 0]], (0, 0), (1, 1)), -1, "不可达")
    check(shortest_path([[1]], (0, 0), (0, 0)), -1, "起点是障碍")
    check(shortest_path([], (0, 0), (0, 0)), -1, "空网格")


def test_path_straight():
    grid = [[0] * 10]
    check(shortest_path(grid, (0, 0), (0, 9)), 9, "一条直线")
    grid = [[0] for _ in range(10)]
    check(shortest_path(grid, (0, 0), (9, 0)), 9, "竖着走")


def test_path_detour():
    grid = [[0, 0, 0],
            [1, 1, 0],
            [0, 0, 0]]
    check(shortest_path(grid, (0, 0), (2, 0)), 6, "要绕一圈")


def test_rot_edges():
    check(rotting_time([[0]]), 0, "全是空格")
    check(rotting_time([[2]]), 0, "本来就没有新鲜的")
    check(rotting_time([[1]]), -1, "没有腐烂源")
    check(rotting_time([[2, 1]]), 1, "一分钟")


def test_rot_unreachable():
    check(rotting_time([[2, 1, 0], [0, 0, 0], [0, 0, 1]]), -1, "有一个传染不到")


def test_rot_multi_source():
    check(rotting_time([[2, 1, 1, 1, 2]]), 2, "两个源同时扩散")


def test_does_not_modify():
    g = [[2, 1], [1, 1]]
    copy = [row[:] for row in g]
    rotting_time(g)
    check(g, copy, "不能修改传入的网格")


def test_large():
    n = 120
    grid = [[0] * n for _ in range(n)]
    check(shortest_path(grid, (0, 0), (n - 1, n - 1)), 2 * (n - 1), "大网格的曼哈顿距离")
    rot = [[1] * n for _ in range(n)]
    rot[0][0] = 2
    check(rotting_time(rot), 2 * (n - 1), "从角落扩散到全图")
