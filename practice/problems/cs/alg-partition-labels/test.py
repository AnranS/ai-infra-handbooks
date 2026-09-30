from checker import check
from solution import merge_ranges, partition_labels


def test_example():
    check(partition_labels("ababcbacadefegdehijhklij"), [9, 7, 8], "三段")
    check(merge_ranges([(1, 3), (2, 4), (6, 8)]), [(1, 4), (6, 8)], "合并重叠")


def test_partition_edges():
    check(partition_labels(""), [], "空串")
    check(partition_labels("a"), [1], "单字符")
    check(partition_labels("abc"), [1, 1, 1], "每个字母一段")
    check(partition_labels("aaa"), [3], "全相同")


def test_partition_spread():
    check(partition_labels("abac"), [3, 1], "a 跨越前三个")
    check(partition_labels("eccbbbbdec"), [10], "整串是一段")


def test_merge_edges():
    check(merge_ranges([]), [], "空")
    check(merge_ranges([(1, 2)]), [(1, 2)], "单个区间")
    check(merge_ranges([(3, 4), (1, 2)]), [(1, 2), (3, 4)], "输入无序且不相交")
    check(merge_ranges([(1, 2), (2, 3)]), [(1, 3)], "端点相接要合并")
    check(merge_ranges([(1, 5), (2, 3)]), [(1, 5)], "包含关系")


def test_merge_chain():
    check(merge_ranges([(1, 2), (2, 3), (3, 4), (10, 11)]), [(1, 4), (10, 11)], "连锁合并")


def test_sum_of_partitions():
    s = "abacdcefghgf"
    parts = partition_labels(s)
    check(sum(parts), len(s), "所有片段加起来是原串长度")
    # 每个字母只能出现在一个片段里
    pos, seen = 0, []
    for length in parts:
        seen.append(set(s[pos:pos + length]))
        pos += length
    for i, a in enumerate(seen):
        for b in seen[i + 1:]:
            check(a & b, set(), "不同片段之间没有共同字母")


def test_large():
    s = "abcdefghij" * 2000
    check(partition_labels(s), [len(s)], "每个字母跨越整串")
    ranges = [(i, i + 1) for i in range(0, 20000, 2)]
    check(len(merge_ranges(ranges)), 10000, "互不相接的区间不合并")
