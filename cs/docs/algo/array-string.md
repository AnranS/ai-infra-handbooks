# 数组与字符串：双指针、滑动窗口、前缀和与二分

<p class="lead">面试里一半以上的题目都落在这一章：给一个数组或字符串，求某种最优的子数组、子串或下标对。它们的共同套路是——暴力解有两层循环，而问题里藏着某种单调性，能让内层循环变成"指针只往前走"或"二分查找"，把 O(n²) 降到 O(n) 或 O(n log n)。这一章把四个模板（双指针、滑动窗口、前缀和与差分、二分）写成可以直接套用的形式，并且用分块 prefill 的分块策略作为二分答案的例子。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 滑动窗口的三种形态（定长、最长满足、最短满足）有什么共同框架？区别在哪？
    2. 为什么滑动窗口是 O(n) 而不是 O(n²)？
    3. `lower_bound` 和 `upper_bound` 分别返回什么？怎么用它们数一个值出现了几次？
    4. 什么时候能用"二分答案"？举一个推理系统里的例子。
    5. 前缀和与差分各把什么操作变成了 O(1)？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 框架都是"右指针每次前进一格，把新元素加入窗口；然后按条件收缩左边界"。区别只在收缩时机：定长窗口是右进一个、左出一个；求最长时是**不满足条件才收缩**；求最短时是**一满足条件就收缩并记录答案**。
    2. 因为左右两个指针都只往右走，各自最多走 n 步，总共 2n 次移动。虽然代码里有嵌套的 `while`，但它执行的总次数受"左指针一共能走多远"约束，不是每个右指针位置都要走 n 步。
    3. `lower_bound(a, x)` 是第一个 ≥ x 的位置，`upper_bound(a, x)` 是第一个 > x 的位置（对应 Python 的 `bisect_left` / `bisect_right`）。出现次数 = `upper_bound - lower_bound`。
    4. 答案落在一个区间里、并且"某个答案可行 ⇒ 所有更大（或更小）的答案都可行"时，就可以在答案空间上二分，把"求最优值"变成"判定可行性"。推理系统里的例子：分块 prefill 的每块 token 上限、给定延迟目标反推最大 batch、给定显存反推能放多少并发。
    5. 前缀和把"区间求和"变成 O(1)（预处理 O(n)）；差分把"区间整体加"变成 O(1)（最后一次前缀和还原）。两者互为逆运算。同时要求区间查询和区间修改时，上树状数组或线段树。

## 双指针与滑动窗口

先一步一步看窗口怎么走：

<div class="aig-widget" data-widget="window-step"></div>

双指针有两种用法：**相向**（左右两端往中间走，用于有序数组的两数之和、盛水容器、回文判断）和**同向**（快慢指针，用于原地删除、去重、移动零）。滑动窗口是同向双指针的一个特例，也是这一章最重要的模板：

```python title="windows.py"
# 滑动窗口的三种形态：定长、最长满足条件、最短满足条件。三者共用同一套框架
from collections import Counter


def fixed_window_max_sum(a, k):
    """定长窗口：长度为 k 的子数组里最大的和"""
    cur = sum(a[:k])
    best = cur
    for right in range(k, len(a)):
        cur += a[right] - a[right - k]          # 进一个、出一个
        best = max(best, cur)
    return best


def longest_at_most_k_distinct(s, k):
    """最长窗口：最多含 k 种字符的最长子串长度"""
    cnt = Counter()
    left = best = 0
    for right, ch in enumerate(s):
        cnt[ch] += 1
        while len(cnt) > k:                     # 不满足条件就收缩左边界
            cnt[s[left]] -= 1
            if cnt[s[left]] == 0:
                del cnt[s[left]]
            left += 1
        best = max(best, right - left + 1)
    return best


def shortest_sum_at_least(a, target):
    """最短窗口：和 >= target 的最短子数组长度（全为正数）"""
    left = cur = 0
    best = len(a) + 1
    for right, x in enumerate(a):
        cur += x
        while cur >= target:                    # 满足了就尽量收缩
            best = min(best, right - left + 1)
            cur -= a[left]
            left += 1
    return best if best <= len(a) else 0


a = [2, 1, 5, 1, 3, 2]
print("定长窗口 k=3 的最大和：", fixed_window_max_sum(a, 3))
print("最长窗口（最多 2 种字符）：", longest_at_most_k_distinct("eceba", 2))
print("最短窗口（和 >= 7）：", shortest_sum_at_least(a, 7))
print()
print("三者的共同框架：右指针每步进一个元素，左指针只会往右走，所以总共 O(n)。")
print("区别只在收缩的时机：定长是右进左出，最长是不满足时收缩，最短是满足时收缩。")
```

```text title="输出"
定长窗口 k=3 的最大和： 9
最长窗口（最多 2 种字符）： 3
最短窗口（和 >= 7）： 3

三者的共同框架：右指针每步进一个元素，左指针只会往右走，所以总共 O(n)。
区别只在收缩的时机：定长是右进左出，最长是不满足时收缩，最短是满足时收缩。
```

套用时问自己三个问题：**窗口里维护什么状态**（和、计数、最大值）、**什么时候收缩**、**答案在哪一步记录**。三者定下来，代码就是模板。

常见变体：

| 题型 | 窗口状态 | 收缩条件 |
| --- | --- | --- |
| 最长无重复子串 | 字符 → 最后出现的位置 | 出现重复时 |
| 最小覆盖子串 | 还差几种字符没凑齐 | 已经凑齐时 |
| 和 ≥ target 的最短子数组 | 窗口和 | 和已经够时 |
| 至多 k 个 0 的最长全 1 段 | 窗口里 0 的个数 | 0 的个数超过 k 时 |
| 定长窗口的最大平均值 | 窗口和 | 每步右进左出 |

窗口里要取最大值或最小值时（比如滑动窗口最大值），需要单调队列，见[链表、栈与哈希](linked-stack-hash.md)。

## 前缀和与差分

```python title="prefix.py"
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
```

```text title="输出"
数组： [3, 1, 4, 1, 5, 9, 2, 6]
前缀和： [0, 3, 4, 8, 9, 14, 23, 25, 31]
区间 [2, 5) 的和 = 10 （直接算是 10 ）

三次区间加之后的数组： [5, 5, 3, -2, -2, 8, 10, 10]

矩阵： [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
左上 2x2 的和 = 12 ，右下 2x2 的和 = 28

规律：前缀和把区间查询变成 O(1)，差分把区间修改变成 O(1)；两者互为逆运算。
需要同时支持区间查询和单点/区间修改时，就该上树状数组或线段树了。
```

前缀和还有两个常见变体：

- **前缀和 + 哈希表**：求"和为 k 的子数组个数"。遍历时维护当前前缀和 `cur`，查 `cur - k` 出现过几次——把两层循环降成一层。这是"子数组和"类题目的标准解法。
- **前缀异或、前缀最大值、前缀乘积**：同一个思路换运算。乘积要小心 0（用左右两次遍历代替除法）。

推理系统里的对应：KV 块的占用统计、每层激活的累计显存、按 token 位置统计的注意力掩码，都是"区间查询"；而"给某段位置整体加一个偏移"（比如批量更新一批请求的槽位计数）就是差分。

## 二分：在下标上，也在答案上

```python title="binary.py"
# 二分的三种用法：找插入位置、在答案空间上二分、以及和标准库对拍
import bisect
import random


def lower_bound(a, x):
    """第一个 >= x 的位置（等价于 bisect.bisect_left）"""
    lo, hi = 0, len(a)                          # 半开区间 [lo, hi)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def upper_bound(a, x):
    """第一个 > x 的位置（等价于 bisect.bisect_right）"""
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


rng = random.Random(0)
for _ in range(2000):                           # 和标准库随机对拍
    a = sorted(rng.randrange(10) for _ in range(rng.randrange(8)))
    x = rng.randrange(10)
    assert lower_bound(a, x) == bisect.bisect_left(a, x), (a, x)
    assert upper_bound(a, x) == bisect.bisect_right(a, x), (a, x)
print("和 bisect 对拍 2000 组：完全一致")
print("出现次数 = upper_bound - lower_bound，例如 [1,2,2,2,3] 里 2 出现",
      upper_bound([1, 2, 2, 2, 3], 2) - lower_bound([1, 2, 2, 2, 3], 2), "次")
print()


def min_chunk_size(lengths, budget, max_chunks):
    """二分答案：把一串请求按顺序切成若干块，每块的 token 数不超过 size，
    问 size 最小取多少才能不超过 max_chunks 块（这就是分块 prefill 的分块策略）"""
    def chunks_needed(size):
        used = cur = 0
        for x in lengths:
            if x > size:
                return float("inf")             # 单个请求就超了，这个 size 不可行
            if cur + x > size:
                used, cur = used + 1, 0
            cur += x
        return used + (1 if cur else 0)

    lo, hi = max(lengths), sum(lengths)         # 答案一定在这个范围里
    while lo < hi:
        mid = (lo + hi) // 2
        if chunks_needed(mid) <= max_chunks:
            hi = mid                            # 可行，试试更小的
        else:
            lo = mid + 1
    return lo if lo <= budget else -1


reqs = [512, 1024, 256, 2048, 128, 768]
for k in (2, 3, 4, 6):
    print(f"把 {len(reqs)} 个请求切成不超过 {k} 块时，每块至少要能装 {min_chunk_size(reqs, 99999, k)} 个 token")
```

```text title="输出"
和 bisect 对拍 2000 组：完全一致
出现次数 = upper_bound - lower_bound，例如 [1,2,2,2,3] 里 2 出现 3 次

把 6 个请求切成不超过 2 块时，每块至少要能装 2944 个 token
把 6 个请求切成不超过 3 块时，每块至少要能装 2048 个 token
把 6 个请求切成不超过 4 块时，每块至少要能装 2048 个 token
把 6 个请求切成不超过 6 块时，每块至少要能装 2048 个 token
```

两件事：

- **模板要统一**。用半开区间 `[lo, hi)`，循环条件 `while lo < hi`，`lo = mid + 1` / `hi = mid`，循环结束时 `lo == hi` 就是答案。这样写不会死循环，也不用纠结 `+1`、`-1`。Python 里直接用 `bisect` 更省事，但面试常要求手写。
- **二分答案更常考**。条件是"答案有单调性"：`size` 可行 ⇒ 所有更大的 `size` 也可行。于是求最优值变成了写一个 `check(x)` 判定函数，再在答案空间里二分。上面的例子是分块 prefill 的分块策略：把一串请求按顺序切块、每块不超过 `size` 个 token，问 `size` 最小取多少能不超过 `max_chunks` 块（见[分块 prefill](minisgl://schedule/chunked-prefill/)）。

同类的二分答案题：最小化最大值（分割数组、运送包裹、跳石头）、最大化最小值（爱吃香蕉的珂珂、制作花束）、以及工程上的"给定 SLO 反推最大并发"。

## 这一章的题在考什么

把常见题型和对应的模板对上号：

| 看到这些字眼 | 往哪个模板想 |
| --- | --- |
| 连续子数组、子串、窗口 | 滑动窗口 |
| 有序数组、两数之和、首尾 | 相向双指针 |
| 原地、O(1) 空间、保持相对顺序 | 快慢指针 |
| 区间和、区间查询 | 前缀和（+ 哈希表） |
| 区间整体修改 | 差分 |
| 有序、找位置、第一个满足 | 二分 |
| 最小化最大值、最大化最小值、能否在 X 内完成 | 二分答案 |
| 第 k 大、Top-K | 堆或快速选择（见[排序、堆与贪心](sort-heap-greedy.md)） |

!!! interview "面试怎么答"
    这一章的题几乎都可以用同一句话开头："暴力解是两层循环 O(n²)，但这里有单调性，可以把内层变成指针移动或二分。"然后说清三件事：窗口/指针里维护什么状态、什么时候移动或收缩、答案在哪一步更新。二分要主动说"我用半开区间的模板，循环结束时 lo 就是答案"，并说明为什么不会死循环。碰到"最小化最大值"这类，直接点出"这是二分答案：写一个 check(x) 判定可行性，答案有单调性所以可以二分"，再分析 check 的复杂度乘上 log。最后主动给复杂度和空间，并说边界：空数组、全相同、窗口比数组长。

## 小结

- [x] 滑动窗口三形态共用一个框架，区别只在收缩时机；左右指针各走一遍所以是 O(n)。
- [x] 前缀和把区间查询变 O(1)，差分把区间修改变 O(1)；前缀和 + 哈希表解决"和为 k 的子数组"。
- [x] 二分用半开区间模板，`lower_bound` / `upper_bound` 的差就是出现次数。
- [x] 二分答案适用于"答案有单调性"的最优化问题，分块 prefill 的块大小就是一个例子。
- [x] 看到"连续子数组"想窗口，看到"原地 O(1)"想快慢指针，看到"最小化最大值"想二分答案。
