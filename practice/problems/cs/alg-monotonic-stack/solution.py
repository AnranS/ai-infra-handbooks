def daily_temperatures(t):
    out = [0] * len(t)
    stack = []                                 # 下标，对应温度单调不增
    for i, x in enumerate(t):
        while stack and t[stack[-1]] < x:
            j = stack.pop()
            out[j] = i - j
        stack.append(i)
    return out


def trap(height):
    stack, total = [], 0
    for i, h in enumerate(height):
        while stack and height[stack[-1]] < h:
            bottom = height[stack.pop()]
            if not stack:                      # 左边没有墙，接不住水
                break
            width = i - stack[-1] - 1
            total += width * (min(height[stack[-1]], h) - bottom)
        stack.append(i)
    return total
