def merge(a, m, b, n):
    i, j, k = m - 1, n - 1, m + n - 1
    while j >= 0:                              # b 放完就结束
        if i >= 0 and a[i] > b[j]:
            a[k] = a[i]
            i -= 1
        else:
            a[k] = b[j]
            j -= 1
        k -= 1
