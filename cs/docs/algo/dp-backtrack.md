# 动态规划与回溯：把重复计算消掉

<p class="lead">动态规划是面试里最让人紧张的一类，但它的套路其实很固定：定义状态、写转移方程、确定初始值和遍历顺序。回溯则是"把所有可能都试一遍"，胜负手在剪枝。这一章按"暴力递归 → 记忆化 → 递推 → 滚动数组"的顺序把 DP 讲清楚，再用全排列、子集、组合总和、n 皇后把回溯的模板固定下来，最后把它们和投机解码的验证树、分块策略联系起来。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 一道题能用动态规划，需要满足什么条件？
    2. 写 DP 要定下哪四件事？
    3. 0-1 背包压缩成一维数组时，内层循环为什么必须倒序？
    4. 回溯的模板是什么？`out.append(path)` 有什么坑？
    5. 记忆化递归和递推（自底向上）各有什么优缺点？

??? success "自测参考答案（先自己答，再展开对照）"
    1. **最优子结构**（大问题的最优解由子问题的最优解组成）和**重叠子问题**（同一个子问题会被反复求解）。只有最优子结构没有重叠，那是分治；贪心还要额外满足"局部最优导致全局最优"。
    2. 状态的定义（`f[i]` 表示什么，说清楚"以 i 结尾"还是"前 i 个"）、转移方程、初始值（含边界）、遍历顺序（保证用到的子问题已经算好）。写之前先把这四句话说出来，代码就只是翻译。
    3. 倒序时 `f[c - w]` 还是"上一轮（不含当前物品）"的值，保证每件物品只用一次；正序会用到本轮已经更新过的值，等于允许同一件物品拿多次——那正好是**完全背包**的写法。
    4. "做选择 → 递归 → 撤销选择"。坑在于 `path` 是同一个列表对象，一路被修改，所以收集答案时必须拷贝一份（`path[:]`），否则最后得到的全是空列表或同一个结果。
    5. 记忆化递归贴近"暴力解 + 缓存"，写起来直观、只计算真正用到的状态；缺点是递归深度受限（Python 1000 层）、常数大。递推没有递归开销、可以滚动数组压缩空间，但要自己想清楚遍历顺序。面试时可以先写记忆化说明思路，再改写成递推。

## 从暴力递归到递推

```python title="dp.py"
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
```

```text title="输出（本机示例）"
fib(28)：暴力递归 46.1 ms，记忆化 0.0239 ms，快 1930 倍
三种写法结果一致：True

0-1 背包（容量 8）：二维 10，一维倒序 10，一维正序 12（错：同一件拿了多次）

编辑距离('horse', 'ros') = 3
编辑距离('intention', 'execution') = 5
编辑距离('', 'abc') = 3
```

三点：

- **记忆化是最省事的入口**：写出暴力递归，加一个 `@lru_cache`，指数级立刻变成多项式级。斐波那契 fib(28) 从 58 毫秒降到 0.04 毫秒。
- **一维滚动数组的方向**决定了语义：倒序是 0-1 背包（每件一次），正序是完全背包（每件无限次）。上面的对照里，正序版本算出 12 而正确答案是 10。
- **二维 DP 的模板**（编辑距离）值得背下来：状态是"a 的前 i 个变成 b 的前 j 个最少要几步"，初始值是全删和全插，转移是"相等就继承对角线，不等就取删/插/替三者最小加一"。最长公共子序列、正则匹配、通配符匹配都是同一个骨架。

常见题型和状态定义：

| 题型 | 状态定义 | 转移 |
| --- | --- | --- |
| 爬楼梯、打家劫舍 | `f[i]`：到第 i 级/前 i 家的最优 | `f[i] = max(f[i-1], f[i-2] + a[i])` |
| 最长递增子序列 | `f[i]`：以 i 结尾的最长长度 | `f[i] = max(f[j] + 1)`，O(n²)；贪心+二分 O(n log n) |
| 0-1 背包 | `f[i][c]`：前 i 件、容量 c 的最大价值 | 拿或不拿 |
| 编辑距离、LCS | `f[i][j]`：两个前缀之间的最优 | 三种操作取最优 |
| 区间 DP（戳气球） | `f[i][j]`：区间 `[i, j]` 的最优 | 枚举分割点 |
| 状压 DP（旅行商） | `f[mask][i]`：访问过 mask、停在 i | 枚举下一个 |

**怎么定状态**是唯一需要练的地方。两个经验：（1）先写出暴力递归函数的参数，参数就是状态；（2）"以 i 结尾"和"前 i 个"是两种不同的定义，前者通常用于子数组/子串（答案要取所有 `f[i]` 的最大值），后者用于"前缀决策"（答案就是 `f[n]`）。

## 回溯：做选择、递归、撤销

```python title="backtrack.py"
# 回溯：做选择 -> 递归 -> 撤销选择。剪枝决定了它能不能跑得动
def permutations(nums):
    out, path, used = [], [], [False] * len(nums)

    def dfs():
        if len(path) == len(nums):
            out.append(path[:])                # 一定要拷贝：path 后面还会被修改
            return
        for i, x in enumerate(nums):
            if used[i]:
                continue
            used[i], _ = True, path.append(x)
            dfs()
            used[i] = False                    # 撤销选择
            path.pop()

    dfs()
    return out


def subsets(nums):
    out, path = [], []

    def dfs(start):
        out.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            dfs(i + 1)                         # 从 i+1 开始：不重复选同一个
            path.pop()

    dfs(0)
    return out


def combination_sum(candidates, target):
    """每个数可以用多次，结果不能重复"""
    out, path = [], []
    nums = sorted(candidates)

    def dfs(start, remain):
        if remain == 0:
            out.append(path[:])
            return
        for i in range(start, len(nums)):
            if nums[i] > remain:               # 剪枝：排序后，后面的只会更大
                break
            path.append(nums[i])
            dfs(i, remain - nums[i])           # 还是 i：允许重复使用
            path.pop()

    dfs(0, target)
    return out


def n_queens(n):
    """统计 n 皇后的解数：用三个集合记录被占用的列和两条对角线"""
    cols, diag1, diag2 = set(), set(), set()
    count = 0

    def dfs(row):
        nonlocal count
        if row == n:
            count += 1
            return
        for col in range(n):
            if col in cols or row - col in diag1 or row + col in diag2:
                continue
            cols.add(col), diag1.add(row - col), diag2.add(row + col)
            dfs(row + 1)
            cols.remove(col), diag1.remove(row - col), diag2.remove(row + col)

    dfs(0)
    return count


print("全排列 [1,2,3]：", permutations([1, 2, 3]))
print("子集 [1,2,3]：", subsets([1, 2, 3]))
print("组合总和（[2,3,6,7] 凑 7）：", combination_sum([2, 3, 6, 7], 7))
print("n 皇后的解数：", {n: n_queens(n) for n in range(4, 10)})
print()
print("模板都是：做选择 -> 递归 -> 撤销。区别在于起点（start）怎么传、什么时候剪枝。")
```

```text title="输出"
全排列 [1,2,3]： [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
子集 [1,2,3]： [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]
组合总和（[2,3,6,7] 凑 7）： [[2, 2, 3], [7]]
n 皇后的解数： {4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352}

模板都是：做选择 -> 递归 -> 撤销。区别在于起点（start）怎么传、什么时候剪枝。
```

三个模板的区别只在"下一层从哪开始"：

- **全排列**：用 `used` 数组标记，每层都从头扫（顺序不同算不同答案）；
- **子集/组合**：传 `start`，下一层从 `i + 1` 开始（不重复选、不考虑顺序）；
- **可重复使用的组合**：下一层还是从 `i` 开始。

有重复元素时要**排序 + 同层去重**（`if i > start and nums[i] == nums[i-1]: continue`），否则会产生重复答案。

剪枝决定了回溯能不能跑动：组合总和里"排序后 `nums[i] > remain` 就 break"、n 皇后用三个集合 O(1) 判断冲突、数独先填候选最少的格子。复杂度通常是 O(解的个数 × 每个解的长度)，加上被剪掉的分支。

**回溯和 DP 的关系**：都是在一棵决策树上搜索。DP 记住子问题的答案、只算一次；回溯遍历所有路径。当"要枚举所有方案"时只能回溯，当"只要最优值或方案数"时用 DP。

## 在推理系统里

- **投机解码的验证树**：草稿模型生成一棵候选树，目标模型一次前向验证所有分支，接受最长的正确前缀。构造树和计算接受长度期望就是树形 DP（见[投机解码进阶](serving://topics/speculative/)）。
- **分块与批次规划**：给定显存和延迟约束选择每步的批次组成，本质是背包/装箱。工程上用贪心近似，但分析上界时用 DP。
- **动态规划在算子里**：FlashAttention 的在线 softmax 是一种"增量维护状态"的思路，和 DP 的滚动数组同源；beam search 是带剪枝的搜索树。
- **编辑距离**：评测里算 WER/CER、投机解码里比较草稿和目标序列的差异，用的都是它。

!!! interview "面试怎么答"
    DP 题先把四件事说出来："状态 `f[i][j]` 表示……，转移是……，初始值是……，遍历顺序是……"，然后再写代码。说不清状态就先写暴力递归、加记忆化，再翻译成递推——这条路径本身就是很好的展示。空间优化要主动提："这里只依赖上一行，可以压成一维，注意 0-1 背包要倒序"。回溯题给出模板（做选择、递归、撤销）并强调两点：收集答案要拷贝 `path[:]`，剪枝在哪一步做。最后说复杂度：DP 是 O(状态数 × 每个状态的转移数)，回溯是 O(解的个数 × 解的长度)。能联系到工程场景（投机解码的验证树、批次规划的背包模型）会加分。

## 小结

- [x] DP 需要最优子结构 + 重叠子问题；写之前先定下状态、转移、初始值、遍历顺序。
- [x] 路径：暴力递归 → 加记忆化 → 改递推 → 滚动数组；0-1 背包一维要倒序，正序是完全背包。
- [x] 二维 DP 的骨架（编辑距离、LCS）值得背下来。
- [x] 回溯 = 做选择 → 递归 → 撤销；收集答案要拷贝，重复元素要排序 + 同层去重，剪枝决定能否跑动。
- [x] 要所有方案用回溯，只要最优值或计数用 DP；投机解码的验证树、批次规划都能落到这两套方法上。
