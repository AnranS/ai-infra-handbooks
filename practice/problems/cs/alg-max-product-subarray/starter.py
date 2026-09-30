def max_product(nums):
    if not nums:
        return 0
    best = cur_max = cur_min = nums[0]
    for x in nums[1:]:
        cur_max = max(x, cur_max * x, cur_min * x)
        cur_min = min(x, cur_max * x, cur_min * x)   # 用了刚更新过的 cur_max
        best = max(best, cur_max)
    return best
