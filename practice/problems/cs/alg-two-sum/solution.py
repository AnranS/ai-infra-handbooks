def two_sum(nums, target):
    seen = {}                                  # 值 -> 下标
    for i, x in enumerate(nums):
        if target - x in seen:                 # 先查：配对的是前面出现过的元素
            return seen[target - x], i
        seen[x] = i
    return None
