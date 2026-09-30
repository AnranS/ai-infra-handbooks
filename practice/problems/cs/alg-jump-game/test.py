from checker import check
from solution import can_jump, min_jumps


def test_example():
    check(can_jump([2, 3, 1, 1, 4]), True, "能到")
    check(can_jump([3, 2, 1, 0, 4]), False, "被 0 挡住")
    check(min_jumps([2, 3, 1, 1, 4]), 2, "两跳")


def test_edges():
    check(can_jump([0]), True, "起点就是终点")
    check(min_jumps([0]), 0, "不用跳")
    check(can_jump([1, 0]), True, "一步到")
    check(min_jumps([1, 0]), 1, "一跳")
    check(can_jump([0, 1]), False, "第一步就卡住")


def test_zeros():
    check(can_jump([2, 0, 0]), True, "跳过零")
    check(can_jump([1, 0, 0]), False, "跳不过去")
    check(can_jump([0, 0, 0]), False, "全是零")


def test_min_jumps_counts():
    check(min_jumps([1, 1, 1, 1]), 3, "只能一步一步")
    check(min_jumps([4, 1, 1, 1]), 1, "一跳到底")
    check(min_jumps([2, 1, 1, 1, 1]), 3, "中间要落一次")


def test_long_reach():
    nums = [100] + [0] * 99
    check(can_jump(nums), True, "第一步就能到底")
    check(min_jumps(nums), 1, "一跳")


def test_large():
    n = 50000
    nums = [1] * n
    check(can_jump(nums), True, "一步一步也能到")
    check(min_jumps(nums), n - 1, "跳 n-1 次")
    nums = [2] * n
    check(min_jumps(nums), (n - 1 + 1) // 2, "每次跳两格")
