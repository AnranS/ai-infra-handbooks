def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        seen[x] = i                            # 先存后查：x + x == target 时会和自己配对
        if target - x in seen:
            return seen[target - x], i
    return None
