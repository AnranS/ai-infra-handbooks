from checker import check
from solution import merge


def run(a, m, b, n):
    merge(a, m, b, n)
    return a


def test_example():
    check(run([1, 2, 3, 0, 0, 0], 3, [2, 5, 6], 3), [1, 2, 2, 3, 5, 6], "交错合并")


def test_edges():
    check(run([1], 1, [], 0), [1], "b 为空")
    check(run([0], 0, [1], 1), [1], "a 为空")
    check(run([], 0, [], 0), [], "都为空")


def test_disjoint():
    check(run([4, 5, 6, 0, 0, 0], 3, [1, 2, 3], 3), [1, 2, 3, 4, 5, 6], "b 全在 a 前面")
    check(run([1, 2, 3, 0, 0, 0], 3, [4, 5, 6], 3), [1, 2, 3, 4, 5, 6], "b 全在 a 后面")


def test_duplicates():
    check(run([2, 2, 0, 0], 2, [2, 2], 2), [2, 2, 2, 2], "全相同")


def test_in_place():
    a = [1, 3, 5, 0, 0, 0]
    before = id(a)
    merge(a, 3, [2, 4, 6], 3)
    check(id(a), before, "必须原地")
    check(a, [1, 2, 3, 4, 5, 6], "结果正确")


def test_large():
    m = n = 10000
    a = list(range(0, 2 * m, 2)) + [0] * n
    b = list(range(1, 2 * n, 2))
    merge(a, m, b, n)
    check(a, list(range(2 * m)), "两万个元素交错合并")
