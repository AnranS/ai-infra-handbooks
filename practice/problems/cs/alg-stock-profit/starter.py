def max_profit_once(prices):
    return max(prices) - min(prices) if prices else 0   # 没考虑顺序：最高价可能在最低价之前


def max_profit_many(prices):
    return max(prices) - min(prices) if prices else 0


def max_profit_cooldown(prices):
    hold, cash = float("-inf"), 0
    for p in prices:
        hold = max(hold, cash - p)             # 忘了冷冻期：卖出当天就又买了
        cash = max(cash, hold + p)
    return cash
