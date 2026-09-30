from checker import check
from solution import sort_colors


def run(a):
    sort_colors(a)
    return a


def test_example():
    check(run([2, 0, 2, 1, 1, 0]), [0, 0, 1, 1, 2, 2], "经典例子")


def test_edges():
    check(run([]), [], "空数组")
    check(run([1]), [1], "单元素")
    check(run([2, 0]), [0, 2], "两个元素")


def test_already_sorted():
    check(run([0, 1, 2]), [0, 1, 2], "已经有序")
    check(run([2, 1, 0]), [0, 1, 2], "完全逆序")


def test_single_color():
    check(run([2, 2, 2]), [2, 2, 2], "全是 2")
    check(run([0, 0]), [0, 0], "全是 0")


def test_in_place():
    a = [2, 1, 0, 2, 1, 0]
    before = id(a)
    sort_colors(a)
    check(id(a), before, "必须原地")
    check(a, [0, 0, 1, 1, 2, 2], "结果正确")


def test_large():
    import random
    rng = random.Random(0)
    a = [rng.randrange(3) for _ in range(30000)]
    counts = [a.count(0), a.count(1), a.count(2)]
    sort_colors(a)
    check(a, [0] * counts[0] + [1] * counts[1] + [2] * counts[2], "大数组")
