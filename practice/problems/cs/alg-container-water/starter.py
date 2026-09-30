def max_area(height):
    left, right = 0, len(height) - 1
    best = 0
    while left < right:
        best = max(best, (right - left) * min(height[left], height[right]))
        if height[left] > height[right]:       # 移动高的那边：会错过最优解
            left += 1
        else:
            right -= 1
    return best
