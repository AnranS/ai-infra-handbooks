from checker import check
from solution import Trie


def test_example():
    t = Trie()
    for w in ["cat", "car", "card"]:
        t.insert(w)
    check(t.contains("car"), True, "完整插入过")
    check(t.contains("ca"), False, "只是前缀")
    check(t.starts_with("ca"), True, "有以它开头的")
    check(t.count_prefix("car"), 2, "car 和 card")
    check(t.longest_match("cards"), 4, "最长匹配 card")


def test_empty():
    t = Trie()
    check(t.contains(""), False, "空树里什么都没有")
    check(t.starts_with(""), True, "空前缀总是存在")
    check(t.longest_match("abc"), 0, "匹配不到")
    check(t.count_prefix("a"), 0, "计数为 0")


def test_insert_empty():
    t = Trie()
    t.insert("")
    check(t.contains(""), True, "插入了空序列")
    check(t.longest_match("abc"), 0, "空序列匹配长度为 0")


def test_duplicate_insert():
    t = Trie()
    t.insert("ab")
    t.insert("ab")
    check(t.count_prefix("a"), 2, "插入两次算两个")
    check(t.contains("ab"), True, "仍然存在")


def test_no_partial_match():
    t = Trie()
    t.insert("abcdef")
    check(t.longest_match("abcxyz"), 3, "匹配到分叉处为止")
    check(t.contains("abc"), False, "abc 没被完整插入")
    check(t.starts_with("abcd"), True, "但它是某个词的前缀")


def test_token_sequences():
    cache = Trie()
    system = (101, 102, 103, 104, 105)
    cache.insert(system + (201, 202))
    check(cache.longest_match(system + (301,)), 5, "共享系统提示词：前 5 个 token 可复用")
    check(cache.longest_match((999,) + system), 0, "换了开头就全部失效")
    check(cache.longest_match(system + (201, 202)), 7, "完全相同")


def test_large():
    t = Trie()
    for i in range(3000):
        t.insert(f"prefix-{i:05d}")
    check(t.count_prefix("prefix-"), 3000, "共同前缀")
    check(t.count_prefix("prefix-0"), 3000, "五位补零，全部以 0 开头")
    check(t.longest_match("prefix-00123-extra"), 12, "匹配到已有序列的末尾")
    check(t.contains("prefix-00123"), True, "完整存在")
