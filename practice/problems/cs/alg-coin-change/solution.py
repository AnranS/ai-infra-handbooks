def coin_change(coins, amount):
    INF = float("inf")
    f = [0] + [INF] * amount
    for coin in coins:
        for total in range(coin, amount + 1):  # 正序：完全背包，每种硬币可以用多次
            if f[total - coin] + 1 < f[total]:
                f[total] = f[total - coin] + 1
    return -1 if f[amount] == INF else f[amount]


def count_ways(coins, amount):
    f = [1] + [0] * amount                     # 凑出 0 有一种方法：什么都不拿
    for coin in coins:                         # 外层硬币：数的是组合
        for total in range(coin, amount + 1):
            f[total] += f[total - coin]
    return f[amount]
