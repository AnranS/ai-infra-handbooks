# 性能分析与优化

<p class="lead">"过早的优化是万恶之源"的下一句是"但我们也不应放过那关键的 3%"。这一章讲怎么找到那 3%：先测量，再定位热点，然后用正确的手段优化。大多数时候，最有效的优化是换一个更合适的数据结构或算法。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 测一小段代码的执行时间，为什么用 `timeit` 而不是 `time.time()` 相减？
    2. 怎么找出一个程序里最耗时的函数？
    3. 在循环里用 `+=` 拼接字符串有什么问题？
    4. 怎么找出是哪行代码分配了最多的内存？
    5. 一个函数被调用了一百万次、每次都很快，但总体很慢，有哪些优化思路？

??? success "自测参考答案（先自己答，再展开对照）"
    1. `timeit` 自动重复很多次取统计值、用高精度的 `perf_counter`、默认暂时关闭垃圾回收，排除了单次测量的噪声和计时器精度的问题。
    2. 用 `cProfile` 跑一遍（`python -m cProfile -s cumtime`），按累计时间排序看最耗时的函数；线上的进程用 py-spy 采样，不需要改代码也不用重启。
    3. 字符串是不可变的，每次 `+=` 都可能创建一个新字符串并复制已有内容，n 次拼接总共复制 O(n²) 个字符。应该把片段放进列表，最后 `"".join(parts)`。
    4. 用 `tracemalloc`：`start()` 之后拍快照，`snapshot.statistics("lineno")` 按代码行统计分配的内存。
    5. 减少调用的次数（换算法和数据结构，避免重复计算、加缓存）；把循环交给内置函数、标准库或 numpy 这类向量化的库；局部化频繁访问的属性和全局名；批量处理；最后才考虑编译扩展或多进程。

## 优化的顺序

1. **先让它正确。** 有测试兜底，再谈优化。
2. **测量，不要猜。** 程序员对热点位置的直觉经常是错的。
3. **先改算法和数据结构。** O(n²) 变 O(n) 的收益，远超任何微优化。
4. **利用现成的高效实现。** 内置函数、标准库、NumPy/Polars 这类向量化库，都是用 C 写的。
5. **最后才考虑微优化**，或者换工具（多进程、Cython、Rust 扩展、PyPy）。

## 测量：`timeit`

`timeit` 会多次运行代码取统计结果，自动选择合适的循环次数，并且关掉了垃圾回收的干扰，比手动计时可靠得多。

命令行里最方便：

```bash
python -m timeit -s "data = list(range(10_000))" "9_999 in data"
# 示例输出：5000 loops, best of 5: 45.2 usec per loop
python -m timeit -s "data = set(range(10_000))" "9_999 in data"
# 示例输出：10000000 loops, best of 5: 21.3 nsec per loop
```

代码里用 `timeit.timeit` 或 `timeit.repeat`：

```python
import timeit

setup = "data = list(range(10_000)); s = set(data)"
t_list = min(timeit.repeat("9_999 in data", setup=setup, number=1_000, repeat=5))
t_set = min(timeit.repeat("9_999 in s", setup=setup, number=1_000, repeat=5))
print(f"list: {t_list * 1e3:.3f} ms, set: {t_set * 1e3:.3f} ms per 1000 lookups")
assert t_set * 50 < t_list          # 集合查找快了几个数量级
```

取多次重复中的**最小值**：它最接近代码本身的速度，其他值偏大是因为受到了系统中其他进程的干扰。

对于一段较长的流程，用 `time.perf_counter()`（高精度、单调递增）计时，不要用 `time.time()`（会受系统时间调整影响）。

## 定位热点：`cProfile`

`cProfile` 记录每个函数被调用的次数和耗时。最简单的用法是命令行：

```bash
python -m cProfile -s cumtime my_script.py | head -20
```

也可以在代码里只分析某一段：

```python
import cProfile
import io
import pstats

def parse(lines):
    return [line.split(",") for line in lines]

def slow_dedupe(rows):
    unique = []
    for r in rows:
        if r not in unique:                 # 列表的 in：O(n)，整体 O(n²)
            unique.append(r)
    return unique

def pipeline():
    lines = [f"{i % 500},{i % 7}" for i in range(20_000)]
    return slow_dedupe(parse(lines))

profiler = cProfile.Profile()
with profiler:                               # Profile 对象可以当上下文管理器用
    result = pipeline()

out = io.StringIO()
pstats.Stats(profiler, stream=out).sort_stats("cumulative").print_stats(5)
report = out.getvalue()
print(report)
assert "slow_dedupe" in report
```

报告里的关键列：

| 列 | 含义 |
| --- | --- |
| `ncalls` | 调用次数 |
| `tottime` | 函数**自身**耗时（不含它调用的其他函数） |
| `cumtime` | 累计耗时（含它调用的所有函数） |

先按 `cumtime` 排序找到"哪条调用链慢"，再按 `tottime` 找到"具体慢在哪个函数"。

!!! tip "更好用的第三方分析器"
    - **py-spy**：采样分析器，可以直接附加到**正在运行的**进程上（`py-spy top --pid 1234`），不用改代码，还能生成火焰图。线上排查首选。
    - **Scalene**：同时分析 CPU、内存和 GPU，并区分时间花在 Python 代码还是 C 代码里。
    - **line_profiler**：逐行统计耗时，找到函数内部的慢语句。
    - **memray**：Bloomberg 出品的内存分析器，能追踪每一次分配。

    Python 3.15 将内置一个统计采样分析器 `profiling.sampling`，开销很低，可以用于生产环境。

## 内存分析：`tracemalloc`

```python
import tracemalloc

tracemalloc.start()

big_list = [str(i) * 10 for i in range(100_000)]      # 分配大量内存的一行
small = {i: i for i in range(100)}

snapshot = tracemalloc.take_snapshot()
top = snapshot.statistics("lineno")[0]                # 按代码行统计，取最大的一项
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(top)
print(f"current={current / 1e6:.1f} MB, peak={peak / 1e6:.1f} MB")
assert current > 5_000_000
```

`statistics("lineno")` 告诉你是哪一行分配了最多的内存。排查内存泄漏时，可以在不同时间点拍两个快照，用 `snapshot2.compare_to(snapshot1, "lineno")` 看增长最多的是哪里。

## 常见的优化手段

### 选对数据结构

这是收益最大的一类优化，[容器与数据结构](../core/containers.md#复杂度速查)一章的复杂度表要烂熟于心：

```python
import timeit

def common_slow(a, b):
    return [x for x in a if x in b]          # b 是列表：O(len(a) * len(b))

def common_fast(a, b):
    b_set = set(b)                           # 建一次集合：O(len(b))
    return [x for x in a if x in b_set]      # 每次查找 O(1)

a = list(range(0, 4_000, 2))
b = list(range(0, 4_000, 3))
assert common_slow(a, b) == common_fast(a, b)

t_slow = timeit.timeit(lambda: common_slow(a, b), number=3)
t_fast = timeit.timeit(lambda: common_fast(a, b), number=3)
print(f"speedup: {t_slow / t_fast:.0f}x")
assert t_fast * 20 < t_slow
```

### 字符串拼接用 `join`

字符串不可变，循环里的 `s += piece` 理论上每次都要创建新字符串，整体是 O(n²)。CPython 对这种情况有一定优化，但不可依赖。标准写法是把片段收集到列表，最后 `"".join(parts)`，或者直接写成生成器表达式：

```python
rows = [("amy", 95), ("bob", 90)]
csv_text = "\n".join(f"{name},{score}" for name, score in rows)
assert csv_text == "amy,95\nbob,90"
```

写入大量文本时，用 `io.StringIO` 或者直接逐行写文件。

### 用内置函数和标准库

内置函数和标准库大多是 C 实现的，比等价的 Python 循环快得多：

| 与其手写循环 | 不如用 |
| --- | --- |
| 累加 | `sum()`、`math.fsum()`、`math.prod()` |
| 找最大最小 | `max()`、`min()`、`heapq.nlargest()` |
| 判断存在 | `any()`、`all()` |
| 计数 | `collections.Counter`、`str.count()` |
| 排序后分组 | `itertools.groupby` |
| 查找子串 | `in`、`str.find()`，而不是逐字符比较 |

### 避免重复计算

- **把不变的计算移出循环**：正则表达式预先 `re.compile`；循环里用到的常量提前算好。
- **缓存纯函数的结果**：`functools.cache`、`cached_property`。
- **惰性求值**：只需要前几个结果时，用生成器配合 `itertools.islice`，或者 `next()` 取第一个满足条件的元素，不要先算出完整的列表。

```python
import re

LOG_LINE = re.compile(r"(?P<level>ERROR|WARN) (?P<msg>.*)")     # 只编译一次

def first_error(lines):
    return next((m["msg"] for line in lines if (m := LOG_LINE.match(line)) and m["level"] == "ERROR"), None)

assert first_error(["INFO ok", "WARN disk 80%", "ERROR disk full", "ERROR again"]) == "disk full"
```

### 减少内存占用

- 用生成器代替列表，数据流式处理，见[迭代器与生成器](../core/iterators.md)。
- 大量小对象用 `__slots__` 或 `@dataclass(slots=True)`。
- 大量同类型数值用 `array.array` 或 NumPy 数组，而不是 `list`。

### 数值计算：向量化

对数值数组做逐元素的 Python 循环，比 NumPy 的向量化运算慢一到两个数量级：

```py
import numpy as np

prices = np.random.rand(1_000_000)
# 慢：Python 循环
total = sum(p * 1.1 for p in prices)
# 快：一条向量化表达式，循环在 C 里完成
total = (prices * 1.1).sum()
```

表格数据处理用 pandas 或 Polars，同样要避免逐行的 Python 循环（`iterrows`、`apply` 里写复杂逻辑）。

## 还不够快怎么办

| 手段 | 适用场景 |
| --- | --- |
| 多进程 / 自由线程版 Python | CPU 密集、可以拆分成独立任务，见[线程、进程与 GIL](threads-processes.md) |
| PyPy | 纯 Python 的长时间运行程序，JIT 可以带来数倍提升；对 C 扩展支持较差 |
| Cython / mypyc | 给热点模块加类型后编译成 C 扩展 |
| Rust（PyO3 + maturin）或 C/C++ 扩展 | 性能关键的核心模块；ruff、pydantic-core、Polars 都是这样做的 |

CPython 自身也在变快：3.11 平均提速约 25%，之后的版本持续优化；3.13 起还有实验性的 JIT 编译器。**升级 Python 版本本身就是一种低成本的优化。**

## 练习

**1. 找出并修复热点。** 下面的函数统计每个单词出现在多少行里。先用 `cProfile` 或 `timeit` 确认它慢在哪里，再把它优化到至少快 10 倍，并验证结果不变。

```py
def word_line_counts(lines):
    words = []
    for line in lines:
        for w in line.split():
            if w not in words:
                words.append(w)
    result = {}
    for w in words:
        result[w] = sum(1 for line in lines if w in line.split())
    return result
```

??? success "参考答案"
    ```python
    import timeit
    from collections import Counter

    def word_line_counts(lines):
        words = []
        for line in lines:
            for w in line.split():
                if w not in words:
                    words.append(w)
        result = {}
        for w in words:
            result[w] = sum(1 for line in lines if w in line.split())
        return result

    def word_line_counts_fast(lines):
        counts = Counter()
        for line in lines:
            counts.update(set(line.split()))     # 每行去重后计数：一次遍历
        return dict(counts)

    lines = [" ".join(f"w{(i * j) % 97}" for j in range(8)) for i in range(400)]
    assert word_line_counts(lines) == word_line_counts_fast(lines)

    slow = timeit.timeit(lambda: word_line_counts(lines), number=1)
    fast = timeit.timeit(lambda: word_line_counts_fast(lines), number=1)
    assert slow > 10 * fast
    ```

    原版有两个问题：`words` 是列表，`in` 判断是 O(n)；更严重的是对每个单词都把所有行重新 `split` 一遍，复杂度是 O(单词数 × 行数 × 行长)。优化后只遍历一次所有行。两个字典的键顺序也一致，因为 `Counter` 同样按首次出现的顺序插入。

**2. 测量内存。** 分别创建 10 万个普通类实例和 `@dataclass(slots=True)` 实例（各有 3 个字段），用 `tracemalloc` 比较它们的内存占用。

??? success "参考答案"
    ```python
    import tracemalloc
    from dataclasses import dataclass

    class Plain:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    @dataclass(slots=True)
    class Slotted:
        x: int
        y: int
        z: int

    def measure(cls, n=100_000):
        tracemalloc.start()
        objs = [cls(i, i, i) for i in range(n)]
        size, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return size, objs

    plain_size, _ = measure(Plain)
    slotted_size, _ = measure(Slotted)
    print(f"plain {plain_size / 1e6:.1f} MB, slots {slotted_size / 1e6:.1f} MB")
    assert slotted_size < plain_size
    ```

    `__slots__` 去掉了每个实例的 `__dict__`。近几个版本的 CPython 已经对普通实例的属性存储做了不少优化，差距比以前小，但在对象数量巨大时仍然明显。

## 小结

- [x] 先测量再优化：小片段用 `timeit`，整体用 `cProfile`，线上用 py-spy，内存用 `tracemalloc`。
- [x] 最大的收益来自算法和数据结构：把 `list` 的 `in` 换成 `set`，把 O(n²) 变成 O(n)。
- [x] 多用内置函数和标准库；字符串拼接用 `join`；正则预编译。
- [x] 数值计算用向量化库；CPU 密集用多进程；最后才考虑编译扩展。
- [x] 升级 Python 版本本身就能带来免费的性能提升。
