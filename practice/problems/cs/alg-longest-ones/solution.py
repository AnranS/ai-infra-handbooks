def longest_ones(nums, k):
    left = best = zeros = 0
    for right, x in enumerate(nums):
        zeros += x == 0
        while zeros > k:                       # 0 太多了，收缩
            zeros -= nums[left] == 0
            left += 1
        best = max(best, right - left + 1)
    return best
