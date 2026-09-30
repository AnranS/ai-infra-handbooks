def can_jump(nums):
    reach = 0
    for i, step in enumerate(nums):
        reach = max(reach, i + step)           # 没检查 i 是否已经超出可达范围
    return reach >= len(nums) - 1


def min_jumps(nums):
    jumps = end = farthest = 0
    for i in range(len(nums)):                 # 遍历到最后一个：会多跳一次
        farthest = max(farthest, i + nums[i])
        if i == end:
            jumps += 1
            end = farthest
    return jumps
