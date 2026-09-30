def daily_temperatures(t):
    out = [0] * len(t)
    stack = []
    for i, x in enumerate(t):
        if stack and t[stack[-1]] < x:         # 用 if 只弹一个：更早的元素永远等不到答案
            j = stack.pop()
            out[j] = i - j
        stack.append(i)
    return out


def trap(height):
    stack, total = [], 0
    for i, h in enumerate(height):
        while stack and height[stack[-1]] < h:
            bottom = height[stack.pop()]
            width = i - (stack[-1] if stack else 0) - 1
            total += width * (min(height[stack[-1]] if stack else h, h) - bottom)
        stack.append(i)
    return total
