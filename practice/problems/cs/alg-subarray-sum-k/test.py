from checker import check
from solution import subarray_sum


def test_example():
    check(subarray_sum([1, 1, 1], 2), 2, "两个")
    check(subarray_sum([1, 2, 3], 3), 2, "[1,2] 和 [3]")
    check(subarray_sum([1, -1, 0], 0), 3, "有负数和零")


def test_from_start():
    check(subarray_sum([3, 4, 7], 3), 1, "从下标 0 开始的子数组")
    check(subarray_sum([5], 5), 1, "整个数组")
    check(subarray_sum([5], 3), 0, "无解")


def test_zeros():
    check(subarray_sum([0, 0, 0], 0), 6, "三个零有 6 个子数组")
    check(subarray_sum([1, 0, -1], 0), 2, "[0] 和 [1,0,-1] 两个")


def test_negative_k():
    check(subarray_sum([-1, -1, 1], -2), 1, "k 是负数")


def test_large():
    a = [1] * 20000
    check(subarray_sum(a, 20000), 1, "只有整个数组")
    check(subarray_sum(a, 1), 20000, "每个单元素")
