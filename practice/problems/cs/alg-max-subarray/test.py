from checker import check
from solution import max_subarray


def test_example():
    check(max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]), (6, 3, 7), "经典例子")


def test_all_negative():
    check(max_subarray([-3, -1, -2]), (-1, 1, 2), "全负时取最大的那个")
    check(max_subarray([-5]), (-5, 0, 1), "只有一个负数")


def test_all_positive():
    check(max_subarray([1, 2, 3]), (6, 0, 3), "全正时是整个数组")


def test_single():
    check(max_subarray([7]), (7, 0, 1), "单元素")
    check(max_subarray([0]), (0, 0, 1), "只有一个零")


def test_tie_breaking():
    check(max_subarray([2, -1, 2]), (3, 0, 3), "起点最小")
    check(max_subarray([1, 0]), (1, 0, 1), "相同的和取更短的")


def test_large():
    a = [(-1) ** i * (i % 7) for i in range(50000)] + [1000]
    total, s, e = max_subarray(a)
    check(e, len(a), "最大和的区间以最后一个大数结尾")
    check(total >= 1000, True, "至少包含那个大数")
