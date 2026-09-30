from checker import check
from solution import climb_stairs, rob, rob_circle


def test_example():
    check(climb_stairs(4), 5, "四级楼梯")
    check(rob([2, 7, 9, 3, 1]), 12, "隔家偷")
    check(rob_circle([2, 3, 2]), 3, "首尾不能同时偷")


def test_climb_edges():
    check(climb_stairs(0), 1, "站着不动也算一种")
    check(climb_stairs(1), 1, "一种")
    check(climb_stairs(2), 2, "两种")
    check(climb_stairs(10), 89, "斐波那契")


def test_rob_edges():
    check(rob([]), 0, "没有房子")
    check(rob([5]), 5, "只有一家")
    check(rob([2, 1]), 2, "两家取大的")
    check(rob([1, 2, 3]), 4, "偷首尾")


def test_rob_not_alternating():
    check(rob([2, 1, 1, 2]), 4, "偷第一和第四家，不是隔一个偷一个")


def test_circle_edges():
    check(rob_circle([]), 0, "空")
    check(rob_circle([5]), 5, "一家时可以偷")
    check(rob_circle([2, 3]), 3, "两家取大的")
    check(rob_circle([1, 2, 3, 1]), 4, "偷第二和第四家")


def test_large():
    check(climb_stairs(50), 20365011074, "第 50 级")
    nums = [i % 7 for i in range(100000)]
    got = rob(nums)
    check(got > sum(nums) // 3, True, "十万家也要能算")
