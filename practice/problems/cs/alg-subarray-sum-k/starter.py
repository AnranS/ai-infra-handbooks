from collections import defaultdict


def subarray_sum(a, k):
    seen = defaultdict(int)                    # 少了 seen[0] = 1
    cur = count = 0
    for x in a:
        cur += x
        seen[cur] += 1                         # 先存后查：k == 0 时会把空子数组数进去
        count += seen[cur - k]
    return count
