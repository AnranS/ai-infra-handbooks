def move_zeroes(a):
    slow = 0                                   # 下一个非零元素该放的位置
    for fast in range(len(a)):
        if a[fast] != 0:
            a[slow], a[fast] = a[fast], a[slow]
            slow += 1
