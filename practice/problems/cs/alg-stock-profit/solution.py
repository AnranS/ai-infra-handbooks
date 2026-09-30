def max_profit_once(prices):
    best, low = 0, float("inf")
    for p in prices:
        low = min(low, p)
        best = max(best, p - low)
    return best


def max_profit_many(prices):
    return sum(max(0, b - a) for a, b in zip(prices, prices[1:]))


def max_profit_cooldown(prices):
    hold, cash, prev_cash = float("-inf"), 0, 0
    for p in prices:
        new_hold = max(hold, prev_cash - p)    # 买入要从前天的 cash 转移
        prev_cash = cash
        cash = max(cash, hold + p)
        hold = new_hold
    return cash
