from checker import check
from solution import combination_sum, permute, subsets


def test_example():
    check(subsets([1, 2, 3]), [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]], "子集")
    check(permute([1, 2, 3]),
          [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]], "全排列")
    check(combination_sum([2, 3, 6, 7], 7), [[2, 2, 3], [7]], "组合总和")


def test_edges():
    check(subsets([]), [[]], "空数组只有空集")
    check(subsets([1]), [[], [1]], "一个元素")
    check(permute([]), [[]], "空排列")
    check(permute([1]), [[1]], "单元素")
    check(combination_sum([2], 1), [], "凑不出")
    check(combination_sum([2], 0), [[]], "凑 0 有一种方法：什么都不选")


def test_counts():
    check(len(subsets([1, 2, 3, 4, 5])), 32, "2^5 个子集")
    check(len(permute([1, 2, 3, 4])), 24, "4! 个排列")


def test_copies_are_independent():
    got = subsets([1, 2])
    got[0].append(99)                          # 改动一个答案不应该影响别的
    check(subsets([1, 2]), [[], [1], [1, 2], [2]], "每个答案是独立的列表")


def test_permute_with_duplicates():
    got = permute([1, 1])
    check(len(got), 2, "两个相同元素仍然有 2 个排列（按位置区分）")
    check(got, [[1, 1], [1, 1]], "结果")


def test_combination_reuse():
    check(combination_sum([2, 3], 6), [[2, 2, 2], [3, 3]], "可以重复使用同一个数")
    check(combination_sum([7, 3, 2], 7), [[2, 2, 3], [7]], "输入无序时先排序")


def test_larger():
    got = combination_sum([2, 3, 5], 20)
    check(all(sum(c) == 20 for c in got), True, "每个组合都等于目标")
    check(all(c == sorted(c) for c in got), True, "组合内部升序")
    check(len(got), len({tuple(c) for c in got}), "没有重复组合")
