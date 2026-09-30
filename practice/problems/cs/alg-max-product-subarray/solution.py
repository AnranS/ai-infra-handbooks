def max_product(nums):
    if not nums:
        return 0
    best = cur_max = cur_min = nums[0]
    for x in nums[1:]:
        prev_max = cur_max                     # 先存下旧值
        cur_max = max(x, prev_max * x, cur_min * x)
        cur_min = min(x, prev_max * x, cur_min * x)
        best = max(best, cur_max)
    return best
