def merge(a, m, b, n):
    i = j = k = 0
    while j < n:
        if i < m and a[i] <= b[j]:             # 从前往后写：会覆盖掉 a 里还没处理的元素
            k += 1
            i += 1
        else:
            a[k] = b[j]
            j += 1
            k += 1
