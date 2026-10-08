# 算法面试怎么准备：复杂度、题型地图与 Python 的坑

<p class="lead">推理岗的面试照样考算法题：一轮手写代码几乎是标配，ACM/NOI 背景还是加分项。但准备方式和刷题网站上的"每日一题"不一样——目标是在 40 分钟里把题读懂、说清思路、写出没有边界错误的代码，并且能分析复杂度。这一章讲清楚怎么估复杂度（含本机实测的常数）、题型地图长什么样、用 Python 写题有哪些坑、以及一套能在面试现场复用的解题流程。后面五章按题型展开，每章都配可在浏览器里判题的练习。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `x in list` 和 `x in set` 差多少？把 10 万个元素当队列用 `list.pop(0)` 取出来，比 `deque.popleft` 慢多少？
    2. 一道题给了 n ≤ 1e5，什么复杂度能过？n ≤ 1e3 呢？n ≤ 20 呢？
    3. `[[0] * 3] * 2` 有什么问题？
    4. 为什么链表题在 Python 里最好写成迭代？
    5. 面试时先写代码还是先说思路？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 本机实测：10 万个元素的列表里查一个不存在的值要 1.1 毫秒，集合里只要 0.05 微秒，差两万多倍（一个 O(n) 一个 O(1)）。队列那题：`list.pop(0)` 每次都要把后面的元素整体前移，是 O(n²)，3 万个元素就要 1.2 秒；`deque.popleft` 是 O(1)，只要 1 毫秒。
    2. 粗略的经验值（Python 每秒大约能做一千万次简单操作）：n ≤ 1e5 时要 O(n) 或 O(n log n)；n ≤ 1e3 时 O(n²) 可以；n ≤ 20 时可以枚举子集 O(2ⁿ)；n ≤ 10 时可以全排列 O(n!)。反过来看数据范围就能猜到出题人想要什么复杂度。
    3. 外层的 `* 2` 复制的是同一个内层列表的引用，改 `grid[0][0]` 会让两行一起变。要用 `[[0] * 3 for _ in range(2)]`。
    4. Python 默认的递归深度上限是 1000，链表长度上万就会 `RecursionError`；而且函数调用的开销不小。树的题目如果树退化成链，也有同样的问题。
    5. 先说思路。把题目复述一遍、确认边界和数据范围、说出暴力解法和它的复杂度、再说优化方向，得到面试官认可后再动手。直接写代码的风险是写到一半发现方向错了，时间也不够了。

## 复杂度：增长与常数

先把几条曲线放在一张图上，拉一拉 n 和常数：

<div class="aig-widget" data-widget="complexity"></div>

复杂度分析看的是"n 变大时增长得多快"，但真实耗时还取决于常数。先看本机的几个数：

```python title="complexity.py"
# 复杂度讲的是"增长"，但常数也要心里有数：本机上几种常见操作的实测耗时
import random
import time
from collections import deque

random.seed(0)
N = 100_000
data = list(range(N))
random.shuffle(data)
s, d = set(data), {x: x for x in data}
dq = deque(data)


def per_call(fn, reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


rows = [
    ("x in list（未命中）", "O(n)", per_call(lambda: -1 in data, 20)),
    ("x in set（未命中）", "O(1)", per_call(lambda: -1 in s, 200000)),
    ("d[x]", "O(1)", per_call(lambda: d[12345], 200000)),
    ("list.append", "O(1) 摊还", per_call(lambda: data.append(0) or data.pop(), 200000)),
    ("deque.popleft", "O(1)", per_call(lambda: dq.appendleft(dq.popleft()), 200000)),
    ("sorted(data)", "O(n log n)", per_call(lambda: sorted(data), 20)),
]
print(f"n = {N:,} 时单次操作的耗时")
for name, big_o, sec in rows:
    print(f"  {name:22s} {big_o:12s} {sec * 1e6:9.2f} 微秒")
print()
M = 30_000
print(f"把 {M:,} 个元素当队列全部取出来：")
q = data[:M]
t0 = time.perf_counter()
while q:
    q.pop(0)                                          # 每次都要把后面的元素整体前移
t_list = time.perf_counter() - t0
q = deque(data[:M])
t0 = time.perf_counter()
while q:
    q.popleft()
t_deque = time.perf_counter() - t0
print(f"  list.pop(0)：  {t_list * 1e3:8.1f} ms（O(n²)）")
print(f"  deque.popleft：{t_deque * 1e3:8.1f} ms（O(n)），快 {t_list / t_deque:.0f} 倍")
print()
print("用错数据结构差多少：求两个 20000 元素列表的交集")
a, b = list(range(20000)), list(range(10000, 30000))
t0 = time.perf_counter()
naive = [x for x in a if x in b]                      # O(n × m)：每个元素都扫一遍另一个列表
t_naive = time.perf_counter() - t0
t0 = time.perf_counter()
fast = sorted(set(a) & set(b))                        # O(n + m)
t_fast = time.perf_counter() - t0
print(f"  列表里查找：{t_naive * 1e3:8.1f} ms")
print(f"  转成集合：  {t_fast * 1e3:8.1f} ms，快 {t_naive / t_fast:.0f} 倍；结果一致：{sorted(naive) == fast}")
```

```text title="输出（本机示例）"
n = 100,000 时单次操作的耗时
  x in list（未命中）         O(n)           1172.54 微秒
  x in set（未命中）          O(1)              0.05 微秒
  d[x]                   O(1)              0.06 微秒
  list.append            O(1) 摊还           0.07 微秒
  deque.popleft          O(1)              0.07 微秒
  sorted(data)           O(n log n)    19676.73 微秒

把 30,000 个元素当队列全部取出来：
  list.pop(0)：    1236.6 ms（O(n²)）
  deque.popleft：     1.2 ms（O(n)），快 1003 倍

用错数据结构差多少：求两个 20000 元素列表的交集
  列表里查找：  1806.4 ms
  转成集合：       2.9 ms，快 630 倍；结果一致：True
```

三个要记住的事实：

- **数据结构选错，差几百上千倍**。列表查找是 O(n)、集合和字典是 O(1)；`list.pop(0)` 是 O(n)，`deque.popleft` 是 O(1)。面试里"用哈希表把 O(n²) 降到 O(n)"是最常见的优化。
- **Python 每秒大约能做一千万次简单操作**。由此得到一张对照表：

| 数据范围 | 可行的复杂度 | 典型解法 |
| --- | --- | --- |
| n ≤ 10 | O(n!) | 全排列、回溯 |
| n ≤ 20 | O(2ⁿ) | 枚举子集、状态压缩 DP |
| n ≤ 100 | O(n³) | 区间 DP、Floyd |
| n ≤ 1000 | O(n²) | 二维 DP、朴素两层循环 |
| n ≤ 1e5 | O(n log n) | 排序、堆、二分、并查集 |
| n ≤ 1e7 | O(n) | 双指针、滑动窗口、前缀和 |
| n ≥ 1e9 | O(log n) 或 O(1) | 二分答案、数学推导 |

- **空间复杂度也要说**。递归的栈深度算在里面；`O(1) 额外空间`是很多题的加分要求（原地交换、双指针、位运算）。

## 题型地图

面试里高频出现的题型就那么几类，后面五章按这个顺序展开：

| 章节 | 题型 | 核心套路 | 推理系统里的对应 |
| --- | --- | --- | --- |
| [数组与字符串](array-string.md) | 双指针、滑动窗口、前缀和、二分 | 用单调性把两层循环降成一层 | 分块 prefill 的窗口、KV 块的区间统计 |
| [链表、栈与哈希](linked-stack-hash.md) | 反转、快慢指针、单调栈、LRU | 用额外结构换时间 | 块分配器的空闲链表、LRU 淘汰 |
| [树与图](tree-graph.md) | 遍历、BFS/DFS、拓扑排序、并查集、最短路 | 状态转移 + 访问标记 | 前缀树（Radix Cache）、计算图的拓扑序 |
| [排序、堆与贪心](sort-heap-greedy.md) | 自定义排序、Top-K、区间合并、调度 | 先排序再贪心 | 请求调度、Top-K 采样、EPLB 装箱 |
| [动态规划与回溯](dp-backtrack.md) | 一维/二维 DP、背包、字符串 DP、剪枝 | 定义状态 + 转移方程 | 投机解码的验证树、分块策略 |

每章末尾都有可以直接在浏览器里判题的练习，按难度排好；把每一类的"标准写法"练到不用想就能写对，比刷题数量重要得多。

## Python 写题的坑

```python title="pitfalls.py"
# 面试写题时 Python 特有的几个坑，每个都能在现场把人绊倒
import sys

# 1. 默认参数是在定义时求值的，可变默认参数会被所有调用共享
def collect(x, acc=[]):
    acc.append(x)
    return acc


print("可变默认参数：", collect(1), collect(2), "  <- 第二次调用带着上一次的结果")


def collect_ok(x, acc=None):
    acc = [] if acc is None else acc
    acc.append(x)
    return acc


print("正确写法：    ", collect_ok(1), collect_ok(2))

# 2. 二维数组不能用 [[0] * n] * m：外层是同一个列表的 m 个引用
grid = [[0] * 3] * 2
grid[0][0] = 1
print("\n[[0]*3]*2 改一个元素：", grid, " <- 两行都变了")
grid = [[0] * 3 for _ in range(2)]
grid[0][0] = 1
print("列表推导式：          ", grid)

# 3. 切片是拷贝：在循环里切片会让复杂度多一个 n
def has_dup_slow(a):
    return any(x in a[i + 1:] for i, x in enumerate(a))    # 每次切片都拷贝一份


def has_dup_fast(a):
    return len(set(a)) != len(a)


print("\n切片会拷贝：两种写法结果一致 =", has_dup_slow([1, 2, 3, 2]) == has_dup_fast([1, 2, 3, 2]))

# 4. 递归深度：默认上限 1000，链表和树的题目很容易超
def depth(n):
    return 0 if n == 0 else 1 + depth(n - 1)


print("默认递归上限：", sys.getrecursionlimit())
try:
    depth(2000)
except RecursionError:
    print("递归 2000 层：RecursionError —— 链表和退化成链的树要用迭代写法")

# 5. 整数除法与负数取整
print("\n-7 // 2 =", -7 // 2, "（向下取整），int(-7 / 2) =", int(-7 / 2), "（向零取整）")
print("-7 % 3 =", -7 % 3, "（Python 的余数跟除数符号），C 语言里是 -1")

# 6. 排序的稳定性与自定义键
words = [("b", 2), ("a", 2), ("c", 1)]
print("\n按第二个元素排序（稳定，相等的保持原序）：", sorted(words, key=lambda t: t[1]))
print("按第二个降序、第一个升序：", sorted(words, key=lambda t: (-t[1], t[0])))
```

```text title="输出"
可变默认参数： [1, 2] [1, 2]   <- 第二次调用带着上一次的结果
正确写法：     [1] [2]

[[0]*3]*2 改一个元素： [[1, 0, 0], [1, 0, 0]]  <- 两行都变了
列表推导式：           [[1, 0, 0], [0, 0, 0]]

切片会拷贝：两种写法结果一致 = True
默认递归上限： 1000
递归 2000 层：RecursionError —— 链表和退化成链的树要用迭代写法

-7 // 2 = -4 （向下取整），int(-7 / 2) = -3 （向零取整）
-7 % 3 = 2 （Python 的余数跟除数符号），C 语言里是 -1

按第二个元素排序（稳定，相等的保持原序）： [('c', 1), ('b', 2), ('a', 2)]
按第二个降序、第一个升序： [('a', 2), ('b', 2), ('c', 1)]
```

除此之外还有几条：

- **用对标准库**：`collections.deque`（双端队列）、`heapq`（小顶堆，要大顶堆就取负）、`bisect`（二分插入位置）、`collections.Counter`（计数）、`defaultdict`（自动初始化）、`itertools`（排列组合）。面试允许用这些，它们能省掉大量样板代码。
- **字符串拼接要用 `"".join()`**：循环里 `s += x` 是 O(n²)。
- **不要在遍历时修改容器**：需要删除元素时，构造新列表或倒着遍历。
- **注意整数除法**：`//` 向下取整，负数时和 C/Java 的 `/`（向零取整）不一样；取模同理。
- **`sort` 是稳定的**，多关键字排序直接写 `key=lambda x: (-x[1], x[0])`。

## 一套解题流程

面试里的 40 分钟怎么用：

1. **复述题目（1～2 分钟）**：用自己的话讲一遍，确认输入输出的类型、数据范围、是否有重复元素、是否要求原地、结果是否唯一。这一步常常能问出关键信息。
2. **举例子（2 分钟）**：手写一个小例子和一个边界例子（空输入、单元素、全相同）。很多错误在这一步就能发现。
3. **说暴力解（1 分钟）**：先给一个一定对的解法和它的复杂度，说明"这是下界，我来优化"。这保证你不会零分。
4. **说优化思路（3～5 分钟）**：指出暴力解里重复做了什么，用什么结构或什么单调性消除它。得到认可再写。
5. **写代码（15 分钟）**：先写函数签名和边界处理，再写主体。变量名写全（`left`、`right` 而不是 `i`、`j`），边界用 `[left, right)` 这样的半开区间统一起来。
6. **自测（5 分钟）**：拿第 2 步的例子手动走一遍，重点看边界：空输入、只有一个元素、越界、死循环。
7. **分析复杂度、说改进（2 分钟）**：时间和空间各是多少、能不能降到 O(1) 空间、数据再大十倍怎么办。

常被问的追问：**"如果数据放不进内存呢？"**（外排序、分块处理、布隆过滤器）、**"如果是多线程呢？"**（分片 + 归并、锁的粒度）、**"如果要支持在线更新呢？"**（换成支持增删的结构，比如堆、平衡树、跳表）。推理岗的面试官很喜欢这类追问，因为它更接近工程。

## 练多少、怎么练

- **数量**：经典的 Hot 100 过两遍，加上本书六章的练习（约 80 道），对推理岗的算法轮基本够用。第二遍只看"一眼能不能说出思路"，说不出来的才重写。
- **限时**：中等题 20 分钟、困难题 35 分钟。超时就看题解，然后第二天重写一遍。
- **手写**：至少有一半的题脱离自动补全写，因为面试时通常是共享文档或白板。
- **复盘**：错在哪一类（读错题、边界、复杂度、写不出来）比错在哪一题重要。同一类错三次就专门练那一类。

!!! interview "怎么讲清楚"
    算法轮没有"怎么答"的技巧，只有流程：复述题目和边界 → 举一个小例子和一个边界例子 → 说暴力解和复杂度 → 指出重复计算、给出优化思路 → 得到认可再写 → 自测边界 → 分析复杂度并主动说改进方向。写的时候变量名写全、区间统一成半开、先处理边界。被追问"数据放不下内存""要并发""要在线更新"时，往外排序与分块、分片与锁粒度、堆与平衡树这几个方向答。复杂度要和数据范围对上：n ≤ 1e5 就该是 O(n log n)，n ≤ 20 才可以枚举子集。

## 练习

**1. 猜复杂度。** 一道题给定 `1 <= n <= 200000`，要求返回一个整数。你会先排除哪些解法？如果数据范围是 `1 <= n <= 18` 呢？

??? success "参考答案"
    n ≤ 2e5：排除 O(n²)（4e10 次操作，肯定超时），目标是 O(n) 或 O(n log n)——常见的是排序、前缀和、哈希表、双指针、堆、二分答案。

    n ≤ 18：这个范围几乎就是在提示"状态压缩"——枚举子集 O(2ⁿ) = 26 万，甚至 O(2ⁿ × n²) 也能过。看到 n ≤ 20 就该往位运算和状压 DP 想。

**2. 找出慢在哪。** 下面这段代码要统计一个列表里出现次数超过一次的元素，n = 1e5 时非常慢。慢在哪？怎么改成 O(n)？

    ```python
    def dups(a):
        return [x for i, x in enumerate(a) if x in a[i + 1:]]
    ```

??? success "参考答案"
    两处问题：`a[i+1:]` 每次都**拷贝**一份剩余列表（O(n) 的空间和时间），`x in 列表` 又是 O(n) 的查找，整体是 O(n²) 甚至更差。

    改法：用 `Counter` 一次遍历统计，再筛出计数大于 1 的（注意去重和保持顺序）：

    ```python
    from collections import Counter

    def dups(a):
        cnt = Counter(a)
        seen = set()
        return [x for x in a if cnt[x] > 1 and not (x in seen or seen.add(x))]
    ```

**3. 递归改迭代。** 把下面的链表反转改成迭代写法，并说明为什么在 Python 里更可取。

    ```python
    def reverse(head):
        if head is None or head.next is None:
            return head
        new_head = reverse(head.next)
        head.next.next = head
        head.next = None
        return new_head
    ```

??? success "参考答案"
    ```python
    def reverse(head):
        prev = None
        while head:
            head.next, prev, head = prev, head, head.next
        return prev
    ```

    递归版本的栈深度等于链表长度，Python 默认上限 1000，链表上万就会 `RecursionError`；而且每层调用都有开销。迭代版本是 O(1) 额外空间，也更容易在面试时一次写对。

**4. 设计解题流程。** 面试官给了一道你没见过的题，5 分钟后你还没有思路。接下来怎么做？

??? success "参考答案"
    （1）把暴力解写出来或说清楚，保证有一个正确的基线；（2）从数据范围反推目标复杂度，缩小解法空间；（3）举一个更小的例子，手动找规律；（4）问自己"暴力解重复做了什么"——重复计算往往对应 DP 或前缀和，重复查找对应哈希表，重复排序对应堆；（5）主动和面试官交流："我在考虑用哈希表消除这层查找，您觉得方向对吗"。沉默是最差的选择，思路的展示本身就是考察点。

## 小结

- [x] 复杂度看增长，也要知道常数：列表查找与集合差两万倍，`list.pop(0)` 与 `deque.popleft` 差上千倍。
- [x] 从数据范围反推复杂度：1e5 → O(n log n)，1e3 → O(n²)，20 → O(2ⁿ)，10 → O(n!)。
- [x] Python 的坑：可变默认参数、`[[0]*n]*m`、切片是拷贝、递归上限 1000、负数除法与取模。
- [x] 用好标准库：`deque`、`heapq`、`bisect`、`Counter`、`defaultdict`、`itertools`。
- [x] 面试流程：复述 → 举例 → 暴力解 → 优化思路 → 写代码 → 自测边界 → 复杂度与改进。
