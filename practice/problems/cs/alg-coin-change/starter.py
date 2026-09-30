def coin_change(coins, amount):
    count, remain = 0, amount
    for coin in sorted(coins, reverse=True):   # 贪心：面额 [1,3,4] 凑 6 会答错
        while remain >= coin:
            remain -= coin
            count += 1
    return count if remain == 0 else -1


def count_ways(coins, amount):
    f = [1] + [0] * amount
    for total in range(1, amount + 1):         # 外层金额：数的是排列，不是组合
        for coin in coins:
            if total >= coin:
                f[total] += f[total - coin]
    return f[amount]
