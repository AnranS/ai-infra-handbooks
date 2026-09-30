def three_sum(nums):
    a = sorted(nums)
    n = len(a)
    out = []
    for i in range(n - 2):
        if a[i] > 0:                           # 剪枝：后面都是正数
            break
        if i > 0 and a[i] == a[i - 1]:         # 去重：第一个数
            continue
        left, right = i + 1, n - 1
        while left < right:
            total = a[i] + a[left] + a[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                out.append([a[i], a[left], a[right]])
                left += 1
                right -= 1
                while left < right and a[left] == a[left - 1]:     # 去重：第二个数
                    left += 1
                while left < right and a[right] == a[right + 1]:   # 去重：第三个数
                    right -= 1
    return out
