def search_rotated(a, target):
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if a[mid] == target:
            return mid
        if a[lo] < a[mid]:                     # 少了等号：只有两个元素时判断错
            if a[lo] <= target < a[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            if a[mid] < target < a[hi]:        # 右端点该取等号
                lo = mid + 1
            else:
                hi = mid - 1
    return -1
