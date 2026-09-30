from checker import check
from solution import max_area


def test_example():
    check(max_area([1, 8, 6, 2, 5, 4, 8, 3, 7]), 49, "经典例子")
    check(max_area([1, 1]), 1, "两根等高")


def test_edges():
    check(max_area([]), 0, "空数组")
    check(max_area([5]), 0, "只有一根线")
    check(max_area([0, 0]), 0, "高度都是 0")


def test_increasing():
    check(max_area([1, 2, 3, 4, 5]), 6, "递增：取中间两根")
    check(max_area([5, 4, 3, 2, 1]), 6, "递减")


def test_tall_edges():
    check(max_area([2, 3, 4, 5, 18, 17, 6]), 17, "高线在中间")


def test_large():
    h = [1] * 20000
    h[0] = h[-1] = 10000
    check(max_area(h), 19999 * 10000, "两端最高时是最宽也最高")
