# 前缀和与差分：把"每次都重新求区间和"降成 O(1)，把"区间批量加"降成 O(1)
def prefix_sums(a):
    ps = [0] * (len(a) + 1)
    for i, x in enumerate(a):
        ps[i + 1] = ps[i] + x                   # ps[i] 是前 i 个元素的和
    return ps


a = [3, 1, 4, 1, 5, 9, 2, 6]
ps = prefix_sums(a)
print("数组：", a)
print("前缀和：", ps)
print("区间 [2, 5) 的和 =", ps[5] - ps[2], "（直接算是", sum(a[2:5]), "）")
print()

# 差分数组：把"给 [l, r) 每个元素加 v"变成两次单点修改，最后做一次前缀和还原
n = 8
diff = [0] * (n + 1)
for l, r, v in [(0, 3, 5), (2, 6, -2), (5, 8, 10)]:
    diff[l] += v
    diff[r] -= v
restored, cur = [], 0
for i in range(n):
    cur += diff[i]
    restored.append(cur)
print("三次区间加之后的数组：", restored)
print()

# 二维前缀和：统计一块矩形区域的和，常用于"KV 块的占用统计""注意力掩码的分块统计"
grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
rows, cols = len(grid), len(grid[0])
ps2 = [[0] * (cols + 1) for _ in range(rows + 1)]
for i in range(rows):
    for j in range(cols):
        ps2[i + 1][j + 1] = grid[i][j] + ps2[i][j + 1] + ps2[i + 1][j] - ps2[i][j]


def rect_sum(r1, c1, r2, c2):                   # 左闭右开
    return ps2[r2][c2] - ps2[r1][c2] - ps2[r2][c1] + ps2[r1][c1]


print("矩阵：", grid)
print("左上 2x2 的和 =", rect_sum(0, 0, 2, 2), "，右下 2x2 的和 =", rect_sum(1, 1, 3, 3))
print()
print("规律：前缀和把区间查询变成 O(1)，差分把区间修改变成 O(1)；两者互为逆运算。")
print("需要同时支持区间查询和单点/区间修改时，就该上树状数组或线段树了。")
