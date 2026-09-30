from checker import check
from solution import max_product


def test_example():
    check(max_product([2, 3, -2, 4]), 6, "前两个")
    check(max_product([-2, 0, -1]), 0, "零最大")
    check(max_product([-2, 3, -4]), 24, "两个负数相乘")


def test_edges():
    check(max_product([]), 0, "空数组")
    check(max_product([0]), 0, "单个零")
    check(max_product([-3]), -3, "单个负数")
    check(max_product([5]), 5, "单个正数")


def test_all_negative():
    check(max_product([-1, -2, -3]), 6, "偶数个负数相乘")
    check(max_product([-1, -2, -3, -4]), 24, "去掉一个负数")


def test_zeros_split():
    check(max_product([2, -5, 0, 3, 4]), 12, "零把数组分成两段")
    check(max_product([0, 0, 0]), 0, "全是零")
    check(max_product([-2, 0, -3]), 0, "零比任何一段都大")


def test_long_negative_run():
    check(max_product([-1, -1, -1, -1, -1]), 1, "五个 -1")
    check(max_product([2, -1, 3, -1, 2]), 12, "跨过两个负数")


def test_large():
    nums = [2 if i % 3 else -1 for i in range(2000)]
    got = max_product(nums)
    check(got > 0, True, "结果为正")
    nums = [1] * 30000
    check(max_product(nums), 1, "全是 1")
