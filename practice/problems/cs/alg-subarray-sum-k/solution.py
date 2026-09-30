from collections import defaultdict


def subarray_sum(a, k):
    seen = defaultdict(int)
    seen[0] = 1                                # 空前缀：让从头开始的子数组也能被数到
    cur = count = 0
    for x in a:
        cur += x
        count += seen[cur - k]                 # 先查：前面有几个前缀和等于 cur - k
        seen[cur] += 1
    return count
