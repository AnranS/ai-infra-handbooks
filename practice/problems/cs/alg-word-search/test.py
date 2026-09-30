from checker import check
from solution import exist


def board():
    return [["A", "B", "C", "E"],
            ["S", "F", "C", "S"],
            ["A", "D", "E", "E"]]


def test_example():
    check(exist(board(), "ABCCED"), True, "能找到")
    check(exist(board(), "SEE"), True, "另一条路径")
    check(exist(board(), "ABCB"), False, "格子不能重复使用")


def test_edges():
    check(exist(board(), ""), True, "空单词")
    check(exist([], "A"), False, "空网格")
    check(exist([["A"]], "A"), True, "单格匹配")
    check(exist([["A"]], "B"), False, "单格不匹配")
    check(exist([["A"]], "AA"), False, "单词比网格长")


def test_backtrack_needed():
    # 走错一条路之后必须能退回来重试
    b = [["A", "A", "A"],
         ["A", "B", "A"],
         ["A", "A", "A"]]
    check(exist(b, "AAB"), True, "要试多条路")
    check(exist(b, "ABA"), True, "经过中间的 B")
    check(exist(b, "BB"), False, "只有一个 B")


def test_snake_path():
    b = [["A", "B", "C"],
         ["F", "E", "D"],
         ["G", "H", "I"]]
    check(exist(b, "ABCDEF"), True, "蛇形路径")
    check(exist(b, "ABCDEFGHI"), True, "整张表刚好是一条蛇形路径")
    check(exist(b, "ABCDEFGHIA"), False, "A 不能重复使用")


def test_does_not_modify():
    b = board()
    copy = [row[:] for row in b]
    exist(b, "ABCCED")
    exist(b, "NOTFOUND")
    check(b, copy, "不能修改传入的网格")


def test_large():
    n = 40
    b = [["A"] * n for _ in range(n)]          # 全是 A：没有剪枝会跑很久
    check(exist(b, "A" * 10), True, "十个 A")
    check(exist(b, "A" * (n * n + 1)), False, "比格子总数还长")
    check(exist(b, "A" * 20 + "B"), False, "没有 B")
