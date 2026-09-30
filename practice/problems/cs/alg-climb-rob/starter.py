def climb_stairs(n):
    if n <= 2:
        return n                               # n == 0 时应该是 1
    a, b = 1, 2
    for _ in range(n - 2):
        a, b = b, a + b
    return b


def rob(nums):
    total = 0
    for i in range(0, len(nums), 2):           # 隔一个偷一个：不是最优
        total += nums[i]
    return total


def rob_circle(nums):
    return rob(nums)                           # 没处理首尾相邻
