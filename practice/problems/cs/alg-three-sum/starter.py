def three_sum(nums):
    a = sorted(nums)
    n = len(a)
    out = []
    for i in range(n - 2):
        if i > 0 and a[i] == a[i - 1]:
            continue
        left, right = i + 1, n - 1
        while left < right:
            total = a[i] + a[left] + a[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                out.append([a[i], a[left], a[right]])   # 只在第一层去重：会产生重复的三元组
                left += 1
                right -= 1
    return out
