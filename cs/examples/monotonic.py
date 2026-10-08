# 单调栈与单调队列：一遍扫描解决"下一个更大元素"和"滑动窗口最大值"
from collections import deque


def next_greater(a):
    """每个元素右边第一个比它大的元素，没有就是 -1"""
    out = [-1] * len(a)
    stack = []                                 # 存下标，对应的值从栈底到栈顶递减
    for i, x in enumerate(a):
        while stack and a[stack[-1]] < x:      # 新元素比栈顶大：栈顶找到了答案
            out[stack.pop()] = x
        stack.append(i)
    return out


def largest_rectangle(heights):
    """柱状图里最大的矩形：每根柱子向左右扩展到第一个更矮的柱子"""
    stack, best = [], 0
    for i, h in enumerate(heights + [0]):      # 末尾补一个 0，把栈里剩下的全部弹出
        while stack and heights[stack[-1]] >= h:
            height = heights[stack.pop()]
            left = stack[-1] + 1 if stack else 0
            best = max(best, height * (i - left))
        stack.append(i)
    return best


def sliding_max(a, k):
    """滑动窗口最大值：双端队列里存下标，值单调递减"""
    dq, out = deque(), []
    for i, x in enumerate(a):
        while dq and a[dq[-1]] <= x:           # 比新元素小的都不可能再当最大值
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:                     # 队首滑出窗口
            dq.popleft()
        if i >= k - 1:
            out.append(a[dq[0]])
    return out


a = [2, 1, 2, 4, 3]
print("数组：", a)
print("下一个更大元素：", next_greater(a))
print("最大矩形（柱状图 [2,1,5,6,2,3]）：", largest_rectangle([2, 1, 5, 6, 2, 3]))
print("滑动窗口最大值（k=3）：", sliding_max([1, 3, -1, -3, 5, 3, 6, 7], 3))
print()
print("共同点：每个下标进栈/进队一次、出一次，所以是 O(n)，哪怕代码里有嵌套的 while。")
