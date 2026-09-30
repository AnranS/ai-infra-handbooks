from checker import check
from solution import longest_ones


def test_example():
    check(longest_ones([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], 2), 6, "翻两个 0")
    check(longest_ones([0, 0, 0], 0), 0, "不能翻")


def test_edges():
    check(longest_ones([], 3), 0, "空数组")
    check(longest_ones([1], 0), 1, "单个 1")
    check(longest_ones([0], 1), 1, "翻掉唯一的 0")


def test_all_ones():
    check(longest_ones([1, 1, 1], 0), 3, "本来就全是 1")
    check(longest_ones([1, 1, 1], 5), 3, "k 比数组还大")


def test_consecutive_zeros():
    check(longest_ones([1, 0, 0, 0, 1], 1), 2, "连续三个 0，只能翻一个")
    check(longest_ones([0, 0, 1, 1, 0, 0], 2), 4, "只能翻两个 0，最多连 4 个")


def test_large():
    nums = [1 if i % 5 else 0 for i in range(30000)]
    check(longest_ones(nums, 3), 19, "每五个里有一个 0，翻三个能连 19 个")
