from checker import check
from solution import group_anagrams, is_anagram


def test_example_is_anagram():
    check(is_anagram("anagram", "nagaram"), True, "是异位词")
    check(is_anagram("rat", "car"), False, "不是")
    check(is_anagram("", ""), True, "两个空串")
    check(is_anagram("a", "ab"), False, "长度不同")
    check(is_anagram("aab", "abb"), False, "字符相同但个数不同")


def test_example_group():
    check(group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]),
          [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]], "经典例子")


def test_group_edges():
    check(group_anagrams([]), [], "空列表")
    check(group_anagrams([""]), [[""]], "空字符串")
    check(group_anagrams(["a"]), [["a"]], "单个词")


def test_group_same_length_not_anagram():
    check(group_anagrams(["abc", "def", "cba"]), [["abc", "cba"], ["def"]], "长度相同但不是异位词")


def test_group_order():
    got = group_anagrams(["bat", "tab", "cat", "act", "dog"])
    check(got, [["bat", "tab"], ["cat", "act"], ["dog"]], "组的顺序按第一次出现")


def test_large():
    words = [f"{'a' * (i % 7)}{'b' * (i % 5)}" for i in range(20000)]
    got = group_anagrams(words)
    check(len(got), len({"".join(sorted(w)) for w in words}), "组数等于不同规范形式的个数")
    check(sum(len(g) for g in got), len(words), "每个词都被分到某一组")
