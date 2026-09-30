def product_except_self(nums):
    n = len(nums)
    out = [1] * n
    for i in range(1, n):
        out[i] = out[i - 1] * nums[i - 1]      # out[i] 先放前缀积
    suffix = 1
    for i in range(n - 1, -1, -1):
        out[i] *= suffix                       # 再乘上后缀积
        suffix *= nums[i]
    return out
