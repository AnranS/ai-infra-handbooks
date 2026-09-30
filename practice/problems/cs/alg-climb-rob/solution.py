def climb_stairs(n):
    a, b = 1, 1                                # f(0) = 1, f(1) = 1
    for _ in range(n):
        a, b = b, a + b
    return a


def rob(nums):
    prev, cur = 0, 0                           # prev = f(i-2), cur = f(i-1)
    for x in nums:
        prev, cur = cur, max(cur, prev + x)
    return cur


def rob_circle(nums):
    if len(nums) <= 1:
        return sum(nums)
    return max(rob(nums[:-1]), rob(nums[1:]))  # 不偷最后一家 / 不偷第一家
