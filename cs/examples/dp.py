# 从"暴力递归"到"记忆化"再到"递推"：同一道题的三种写法
import time
from functools import lru_cache


def fib_naive(n):
    return n if n < 2 else fib_naive(n - 1) + fib_naive(n - 2)


@lru_cache(maxsize=None)
def fib_memo(n):
    return n if n < 2 else fib_memo(n - 1) + fib_memo(n - 2)


def fib_iter(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


t0 = time.perf_counter()
fib_naive(28)
t_naive = time.perf_counter() - t0
t0 = time.perf_counter()
fib_memo(28)
t_memo = time.perf_counter() - t0
print(f"fib(28)：暴力递归 {t_naive * 1e3:.1f} ms，记忆化 {t_memo * 1e3:.4f} ms，"
      f"快 {t_naive / max(t_memo, 1e-9):.0f} 倍")
print(f"三种写法结果一致：{fib_naive(20) == fib_memo(20) == fib_iter(20)}")
print()

# 0-1 背包：二维表 -> 一维滚动数组（注意内层要倒序）
def knapsack_2d(weights, values, cap):
    n = len(weights)
    f = [[0] * (cap + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for c in range(cap + 1):
            f[i][c] = f[i - 1][c]                                   # 不拿第 i 件
            if c >= weights[i - 1]:
                f[i][c] = max(f[i][c], f[i - 1][c - weights[i - 1]] + values[i - 1])
    return f[n][cap]


def knapsack_1d(weights, values, cap):
    f = [0] * (cap + 1)
    for w, v in zip(weights, values):
        for c in range(cap, w - 1, -1):                             # 倒序：保证每件只用一次
            f[c] = max(f[c], f[c - w] + v)
    return f[cap]


def knapsack_wrong(weights, values, cap):
    f = [0] * (cap + 1)
    for w, v in zip(weights, values):
        for c in range(w, cap + 1):                                 # 正序：同一件会被用多次（完全背包）
            f[c] = max(f[c], f[c - w] + v)
    return f[cap]


weights, values, cap = [2, 3, 4, 5], [3, 4, 5, 6], 8
print(f"0-1 背包（容量 {cap}）：二维 {knapsack_2d(weights, values, cap)}，"
      f"一维倒序 {knapsack_1d(weights, values, cap)}，一维正序 {knapsack_wrong(weights, values, cap)}（错：同一件拿了多次）")
print()

# 编辑距离：二维 DP 的标准模板
def edit_distance(a, b):
    m, n = len(a), len(b)
    f = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        f[i][0] = i                                                 # 全删
    for j in range(n + 1):
        f[0][j] = j                                                 # 全插
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                f[i][j] = f[i - 1][j - 1]
            else:
                f[i][j] = 1 + min(f[i - 1][j], f[i][j - 1], f[i - 1][j - 1])   # 删、插、替
    return f[m][n]


for a, b in [("horse", "ros"), ("intention", "execution"), ("", "abc")]:
    print(f"编辑距离('{a}', '{b}') = {edit_distance(a, b)}")
