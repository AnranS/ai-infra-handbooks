from checker import check
from solution import max_profit_cooldown, max_profit_many, max_profit_once


def test_example():
    check(max_profit_once([7, 1, 5, 3, 6, 4]), 5, "1 买 6 卖")
    check(max_profit_many([7, 1, 5, 3, 6, 4]), 7, "多次交易")
    check(max_profit_cooldown([1, 2, 3, 0, 2]), 3, "带冷冻期")


def test_edges():
    check(max_profit_once([]), 0, "空")
    check(max_profit_once([5]), 0, "只有一天")
    check(max_profit_many([]), 0, "空")
    check(max_profit_cooldown([]), 0, "空")
    check(max_profit_cooldown([5]), 0, "只有一天")


def test_decreasing():
    check(max_profit_once([7, 6, 4, 3, 1]), 0, "一直跌，不交易")
    check(max_profit_many([7, 6, 4, 3, 1]), 0, "一直跌")
    check(max_profit_cooldown([7, 6, 4, 3, 1]), 0, "一直跌")


def test_order_matters():
    check(max_profit_once([5, 1]), 0, "最高价在最低价之前")
    check(max_profit_once([1, 5]), 4, "正常顺序")


def test_increasing():
    check(max_profit_once([1, 2, 3, 4]), 3, "一直涨：一次买卖")
    check(max_profit_many([1, 2, 3, 4]), 3, "一直涨：多次等于一次")
    check(max_profit_cooldown([1, 2, 3, 4]), 3, "一直涨时冷冻期没影响")


def test_cooldown_matters():
    check(max_profit_many([1, 4, 1, 4]), 6, "两次交易")
    check(max_profit_cooldown([1, 4, 1, 4]), 3, "冷冻期让第二次错过")


def test_large():
    prices = [(i * 7919) % 1000 for i in range(50000)]
    a = max_profit_once(prices)
    b = max_profit_many(prices)
    c = max_profit_cooldown(prices)
    check(a <= b, True, "多次不会比一次差")
    check(c <= b, True, "有冷冻期不会比没有好")
    check(a >= 0 and c >= 0, True, "利润非负")
