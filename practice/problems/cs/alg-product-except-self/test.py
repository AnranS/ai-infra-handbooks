from checker import check
from solution import product_except_self


def test_example():
    check(product_except_self([1, 2, 3, 4]), [24, 12, 8, 6], "基本例子")
    check(product_except_self([-1, 1, 0, -3, 3]), [0, 0, 9, 0, 0], "有一个零")


def test_zeros():
    check(product_except_self([0, 0]), [0, 0], "两个零")
    check(product_except_self([0, 5]), [5, 0], "一个零")
    check(product_except_self([0, 1, 2]), [2, 0, 0], "零在开头")


def test_small():
    check(product_except_self([2, 3]), [3, 2], "两个元素")
    check(product_except_self([7]), [1], "单元素：空积是 1")


def test_negative():
    check(product_except_self([-1, -2, -3]), [6, 3, 2], "全负")


def test_large():
    n = 20000
    nums = [1] * n
    nums[123] = 2
    out = product_except_self(nums)
    check(out[123], 1, "自己不算进去")
    check(out[0], 2, "别的位置包含那个 2")
