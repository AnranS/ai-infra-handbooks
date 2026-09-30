from checker import check
from solution import coin_change, count_ways


def test_example():
    check(coin_change([1, 3, 4], 6), 2, "贪心在这里是错的")
    check(coin_change([2], 3), -1, "凑不出")
    check(count_ways([1, 2, 5], 5), 4, "四种组合")


def test_change_edges():
    check(coin_change([1], 0), 0, "凑 0 元")
    check(coin_change([], 5), -1, "没有硬币")
    check(coin_change([], 0), 0, "凑 0 元不需要硬币")
    check(coin_change([5], 5), 1, "正好一枚")
    check(coin_change([2, 4], 7), -1, "全是偶数凑不出奇数")


def test_change_classic():
    check(coin_change([1, 2, 5], 11), 3, "5+5+1")
    check(coin_change([186, 419, 83, 408], 6249), 20, "大面额")


def test_ways_edges():
    check(count_ways([1], 0), 1, "凑 0 有一种方法")
    check(count_ways([], 5), 0, "没有硬币")
    check(count_ways([2], 3), 0, "凑不出")
    check(count_ways([1], 5), 1, "只有一种")


def test_ways_is_combination():
    check(count_ways([1, 2], 3), 2, "[1,1,1] 和 [1,2]，不算 [2,1]")
    check(count_ways([2, 3, 5], 8), 3, "[3,5]、[2,3,3] 和 [2,2,2,2]")


def test_large():
    check(coin_change([1, 2, 5], 10000), 2000, "两千枚 5 元")
    check(count_ways([1, 2, 5], 500) > 0, True, "大金额也能算")
