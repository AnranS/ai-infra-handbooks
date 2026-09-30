from checker import check
from solution import longest_unique


def test_example():
    check(longest_unique("abcabcbb"), 3, "abc")
    check(longest_unique("bbbbb"), 1, "全相同")
    check(longest_unique("pwwkew"), 3, "wke")


def test_empty_and_single():
    check(longest_unique(""), 0, "空串")
    check(longest_unique("a"), 1, "单字符")


def test_backward_jump():
    check(longest_unique("abba"), 2, "左边界不能回退")
    check(longest_unique("tmmzuxt"), 5, "mzuxt")


def test_all_unique():
    check(longest_unique("abcdef"), 6, "整串都不重复")


def test_large():
    s = "abcdefghij" * 5000
    check(longest_unique(s), 10, "周期为 10 的长串")
