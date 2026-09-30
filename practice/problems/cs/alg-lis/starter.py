def lis_dp(nums):
    f = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                f[i] = f[j] + 1                # 没取最大值：会被后面的短序列覆盖
    return f[-1]                               # 答案不是最后一个，而是最大值


def lis_fast(nums):
    tails = []
    for x in nums:
        if not tails or x > tails[-1]:
            tails.append(x)                    # 只追加，不替换：结尾没保持最小
    return len(tails)
