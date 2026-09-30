from checker import check
from solution import two_sum


def test_example():
    check(two_sum([2, 7, 11, 15], 9), (0, 1), "前两个")
    check(two_sum([3, 2, 4], 6), (1, 2), "不是第一个元素")


def test_same_value():
    check(two_sum([3, 3], 6), (0, 1), "两个相同的值")
    check(two_sum([0, 4, 3, 0], 0), (0, 3), "两个 0")


def test_negative():
    check(two_sum([-3, 4, 3, 90], 0), (0, 2), "负数")


def test_order():
    nums = [1, 5, 9, 13]
    i, j = two_sum(nums, 14)
    check(i < j, True, "下标从小到大")
    check(nums[i] + nums[j], 14, "两个下标对应的值加起来等于目标")


def test_large():
    n = 20000
    nums = list(range(n))
    check(two_sum(nums, 2 * n - 3), (n - 2, n - 1), "两万个元素也要很快")
