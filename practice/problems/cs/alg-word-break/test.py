from checker import check
from solution import min_cuts, word_break


def test_example():
    check(word_break("leetcode", ["leet", "code"]), True, "能拆")
    check(word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]), False, "拆不了")
    check(min_cuts("catsanddog", ["cat", "cats", "and", "sand", "dog"]), 3, "最少三段")


def test_edges():
    check(word_break("", ["a"]), True, "空串算能拆")
    check(word_break("a", []), False, "空字典")
    check(word_break("a", ["a"]), True, "正好一个词")
    check(min_cuts("", ["a"]), 0, "空串零段")
    check(min_cuts("abc", []), -1, "空字典")


def test_reuse():
    check(word_break("aaaa", ["a"]), True, "重复使用同一个词")
    check(min_cuts("aaaa", ["a", "aa"]), 2, "用两个 aa 更少")


def test_greedy_fails():
    # 贪心先切长的会失败，但实际上是可以拆的
    check(word_break("aaaaab", ["aaaa", "aaa", "b", "aa"]), True, "需要回退的情形")
    check(min_cuts("aaaaab", ["aaaa", "aaa", "b", "aa"]), 3, "aaa + aa + b")


def test_no_solution():
    check(word_break("abcd", ["a", "bc"]), False, "剩下的 d 拆不了")
    check(min_cuts("abcd", ["a", "bc"]), -1, "无解")


def test_large():
    s = "a" * 300 + "b"
    words = ["a", "aa", "aaa"]
    check(word_break(s, words), False, "最后的 b 拆不了")
    check(word_break(s + "b", words + ["bb"]), True, "补上 bb 就能拆")
    check(min_cuts("a" * 300, words), 100, "全用 aaa")
