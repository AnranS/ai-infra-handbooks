from collections import deque


def sliding_max(nums, k):
    dq, out = deque(), []
    for i, x in enumerate(nums):
        if dq and nums[dq[-1]] <= x:           # 用 if 只弹一个：队列不再单调
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            out.append(nums[dq[0]])
    return out
