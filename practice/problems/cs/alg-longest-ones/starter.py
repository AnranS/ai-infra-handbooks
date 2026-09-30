def longest_ones(nums, k):
    left = best = zeros = 0
    for right, x in enumerate(nums):
        zeros += x == 0
        while zeros > k:
            zeros -= 1                         # 不管左边是不是 0 都减：窗口里的 0 被少算了
            left += 1
        best = max(best, right - left + 1)
    return best
