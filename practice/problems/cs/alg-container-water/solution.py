def max_area(height):
    left, right = 0, len(height) - 1
    best = 0
    while left < right:
        best = max(best, (right - left) * min(height[left], height[right]))
        if height[left] < height[right]:       # 移动矮的那边
            left += 1
        else:
            right -= 1
    return best
