def move_zeroes(a):
    for i in range(len(a)):
        if a[i] == 0:
            a.pop(i)                           # 一边遍历一边删除：下标会错位，而且 pop 是 O(n)
            a.append(0)
