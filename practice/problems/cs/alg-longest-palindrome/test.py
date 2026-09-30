from checker import check
from solution import count_palindromes, longest_palindrome


def test_example():
    check(longest_palindrome("babad"), "bab", "最靠左的答案")
    check(longest_palindrome("cbbd"), "bb", "偶数长度")
    check(count_palindromes("aaa"), 6, "三个单字符 + 两个 aa + 一个 aaa")


def test_edges():
    check(longest_palindrome(""), "", "空串")
    check(longest_palindrome("a"), "a", "单字符")
    check(count_palindromes(""), 0, "空串没有回文")
    check(count_palindromes("a"), 1, "单字符算一个")


def test_no_long_palindrome():
    check(longest_palindrome("abcd"), "a", "只有单字符是回文")
    check(count_palindromes("abcd"), 4, "四个单字符")


def test_whole_string():
    check(longest_palindrome("racecar"), "racecar", "整串是回文")
    check(longest_palindrome("abba"), "abba", "偶数长度的整串")


def test_leftmost():
    check(longest_palindrome("abacdc"), "aba", "两个长度 3 的回文取左边的")


def test_large():
    s = "a" * 1000
    check(longest_palindrome(s), s, "全相同")
    check(count_palindromes(s), 1000 * 1001 // 2, "所有子串都是回文")
    s = "ab" * 500
    check(len(longest_palindrome(s)), 999, "abab… 的最长回文")
