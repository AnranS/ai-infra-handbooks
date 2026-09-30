def lis_dp(nums):
    if not nums:
        return 0
    f = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i] and f[j] + 1 > f[i]:
                f[i] = f[j] + 1
    return max(f)


def lis_fast(nums):
    tails = []                                 # tails[k]：长度 k+1 的递增子序列里最小的结尾
    for x in nums:
        lo, hi = 0, len(tails)                 # 手写 lower_bound
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(tails):
            tails.append(x)
        else:
            tails[lo] = x                      # 用更小的结尾替换
    return len(tails)
