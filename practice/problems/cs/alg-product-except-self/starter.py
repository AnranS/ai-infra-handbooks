def product_except_self(nums):
    total = 1
    for x in nums:
        total *= x
    return [total // x for x in nums]          # 用了除法：有 0 时直接崩
