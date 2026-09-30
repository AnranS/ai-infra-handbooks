from checker import check
from solution import three_sum


def test_example():
    check(three_sum([-1, 0, 1, 2, -1, -4]), [[-1, -1, 2], [-1, 0, 1]], "经典例子")
    check(three_sum([0, 0, 0, 0]), [[0, 0, 0]], "全零只算一组")
    check(three_sum([1, 2, 3]), [], "无解")


def test_small():
    check(three_sum([]), [], "空数组")
    check(three_sum([0, 0]), [], "不足三个")


def test_many_duplicates():
    check(three_sum([-2, 0, 0, 2, 2]), [[-2, 0, 2]], "重复元素只出一组")
    check(three_sum([-1, -1, 2, 2, 0, 1]), [[-1, -1, 2], [-1, 0, 1]], "多组且有重复")


def test_sorted_output():
    got = three_sum([3, -2, -1, 0, 1, 2, -3])
    check(got == sorted(got), True, "结果按字典序")
    check(all(t == sorted(t) for t in got), True, "每个三元组内部升序")
    check(all(sum(t) == 0 for t in got), True, "每组和为 0")


def test_no_duplicate_triples():
    got = three_sum([-4, -2, -2, -2, 0, 1, 2, 2, 2, 3, 3, 4, 4, 6, 6])
    check(len(got), len({tuple(t) for t in got}), "没有重复的三元组")


def test_large():
    a = list(range(-300, 300))
    got = three_sum(a)
    check(all(sum(t) == 0 for t in got), True, "都为 0")
    check(len(got), len({tuple(t) for t in got}), "无重复")
