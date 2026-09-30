from checker import check
from solution import min_window


def test_example():
    check(min_window("ADOBECODEBANC", "ABC"), "BANC", "经典例子")
    check(min_window("a", "aa"), "", "凑不齐")


def test_edges():
    check(min_window("", "a"), "", "空串")
    check(min_window("a", ""), "", "t 为空")
    check(min_window("a", "a"), "a", "正好相等")


def test_duplicates():
    check(min_window("aaab", "aab"), "aab", "重复字符要数够")
    check(min_window("bba", "ab"), "ba", "最短的那个")


def test_leftmost():
    check(min_window("abab", "ab"), "ab", "多个最短答案取最左")
    check(min_window("abba", "ab"), "ab", "同样长度时取左边那个，而不是右边的 ba")


def test_large():
    s = "x" * 20000 + "abc" + "x" * 20000
    check(min_window(s, "abc"), "abc", "长串里找一小段")
