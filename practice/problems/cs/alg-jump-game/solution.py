def can_jump(nums):
    reach = 0
    for i, step in enumerate(nums):
        if i > reach:                          # 走不到这里，断了
            return False
        reach = max(reach, i + step)
    return True


def min_jumps(nums):
    jumps = end = farthest = 0
    for i in range(len(nums) - 1):             # 到倒数第二个就够了
        farthest = max(farthest, i + nums[i])
        if i == end:                           # 当前这一跳的边界到了
            jumps += 1
            end = farthest
    return jumps
