from collections import deque


def sliding_max(nums, k):
    if not nums or k <= 0:
        return []
    k = min(k, len(nums))
    dq, out = deque(), []                      # 队列里存下标，值单调递减
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] <= x:
            dq.pop()                           # 比新元素小的都没用了
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()                       # 队首滑出窗口
        if i >= k - 1:
            out.append(nums[dq[0]])
    return out
