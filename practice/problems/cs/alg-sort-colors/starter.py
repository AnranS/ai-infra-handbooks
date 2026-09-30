def sort_colors(a):
    low, i, high = 0, 0, len(a) - 1
    while i <= high:
        if a[i] == 0:
            a[low], a[i] = a[i], a[low]
            low += 1
            i += 1
        elif a[i] == 2:
            a[high], a[i] = a[i], a[high]
            high -= 1
            i += 1                             # 换过来的元素被跳过了
        else:
            i += 1
