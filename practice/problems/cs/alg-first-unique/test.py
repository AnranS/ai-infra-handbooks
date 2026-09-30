from checker import check
from solution import first_unique_char, top_k_frequent


def test_example():
    check(first_unique_char("leetcode"), 0, "第一个字符")
    check(first_unique_char("loveleetcode"), 2, "第三个字符")
    check(top_k_frequent([1, 1, 1, 2, 2, 3], 2), [1, 2], "前两名")
    check(top_k_frequent([4, 4, 5, 5, 6], 2), [4, 5], "次数相同按先出现的")


def test_unique_edges():
    check(first_unique_char(""), -1, "空串")
    check(first_unique_char("aabb"), -1, "全都重复")
    check(first_unique_char("a"), 0, "单字符")
    check(first_unique_char("aab"), 2, "答案在末尾")


def test_topk_edges():
    check(top_k_frequent([], 3), [], "空数组")
    check(top_k_frequent([1], 1), [1], "单元素")
    check(top_k_frequent([1, 2, 3], 5), [1, 2, 3], "k 比种类还多")
    check(top_k_frequent([1, 2, 3], 0), [], "k 为 0")


def test_all_same_count():
    check(top_k_frequent([3, 1, 2], 3), [3, 1, 2], "次数都是 1，按出现顺序")
    check(top_k_frequent([2, 2, 1, 1, 3, 3], 2), [2, 1], "次数都是 2")


def test_large():
    s = "a" * 50000 + "b" + "a" * 50000
    check(first_unique_char(s), 50000, "十万个字符里找唯一的那个")
    nums = [i % 100 for i in range(50000)] + [7] * 1000
    check(top_k_frequent(nums, 1), [7], "出现最多的")
