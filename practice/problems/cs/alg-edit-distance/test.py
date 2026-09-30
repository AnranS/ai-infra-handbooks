from checker import check
from solution import edit_distance, lcs


def test_example():
    check(edit_distance("horse", "ros"), 3, "经典例子")
    check(edit_distance("intention", "execution"), 5, "另一个经典例子")
    check(lcs("abcde", "ace"), 3, "最长公共子序列")


def test_empty():
    check(edit_distance("", ""), 0, "两个空串")
    check(edit_distance("", "abc"), 3, "全插入")
    check(edit_distance("abc", ""), 3, "全删除")
    check(lcs("", "abc"), 0, "空串没有公共子序列")


def test_identical():
    check(edit_distance("same", "same"), 0, "完全相同")
    check(lcs("same", "same"), 4, "公共子序列就是自己")


def test_single_ops():
    check(edit_distance("cat", "cut"), 1, "替换一个")
    check(edit_distance("cat", "cats"), 1, "插入一个")
    check(edit_distance("cats", "cat"), 1, "删除一个")


def test_no_common():
    check(edit_distance("abc", "xyz"), 3, "全部替换")
    check(lcs("abc", "xyz"), 0, "没有公共部分")


def test_lcs_order_matters():
    check(lcs("abc", "cba"), 1, "顺序相反只能取一个")
    check(lcs("aab", "aba"), 2, "有多种取法")


def test_large():
    a = "ab" * 300
    b = "ba" * 300
    check(edit_distance(a, a), 0, "六百个字符完全相同")
    check(lcs(a, b) >= 599, True, "长串的公共子序列")
    check(edit_distance("x" * 500, "y" * 500), 500, "全部替换")
