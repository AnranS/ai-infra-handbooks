# 迭代器与生成器

<p class="lead"><code>for</code> 循环背后是一套简单的协议。掌握它之后，你可以用生成器写出内存占用恒定的数据管道，用 <code>itertools</code> 把十几行循环压成一行，还能读懂 asyncio 的前身。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 可迭代对象（iterable）和迭代器（iterator）有什么区别？
    2. 为什么对同一个生成器遍历两次，第二次什么都没有？
    3. `yield from` 做了什么？
    4. `sum([x * x for x in data])` 和 `sum(x * x for x in data)` 有什么区别？
    5. `itertools.groupby` 在什么情况下会给出"错误"的分组？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 可迭代对象能通过 `iter()` 产生一个迭代器（实现了 `__iter__`），可以被遍历多次；迭代器实现了 `__next__`，记录遍历到哪里，是一次性的。
    2. 生成器是迭代器，第一次遍历已经把它耗尽了，第二次调用 `next` 立刻就是 `StopIteration`。要么重新调用生成器函数，要么先把结果存进列表。
    3. 把另一个可迭代对象产生的值逐个 `yield` 出去（相当于 `for x in sub: yield x`），而且会把 `send`、`throw` 和子生成器的返回值正确地传递。
    4. 前者先建出一个完整的列表再求和，要占 O(n) 内存；后者是生成器表达式，边产生边消费，内存是常数。
    5. 它只把**相邻**的相同键分到一组；数据没按这个键排序时，同一个键会被分成好几组。要先按同样的键排序。

## 迭代协议

- **可迭代对象**：实现了 `__iter__`，调用 `iter(obj)` 能得到一个迭代器。列表、字符串、字典、文件都是。
- **迭代器**：实现了 `__next__`（每次返回下一个元素，没有了就抛 `StopIteration`），并且 `__iter__` 返回自己。

`for x in obj` 实际上做的是：

```python
it = iter([10, 20])
while True:
    try:
        x = next(it)
    except StopIteration:
        break
    print(x)
```

```pycon
>>> nums = [1, 2]
>>> it = iter(nums)
>>> next(it), next(it)
(1, 2)
>>> next(it)
Traceback (most recent call last):
  ...
StopIteration
>>> next(it, "done")          # 给默认值就不抛异常
'done'
```

**迭代器是一次性的。** 列表每次 `iter()` 都会给一个新迭代器，所以可以反复遍历；但迭代器本身遍历完就空了：

```pycon
>>> squares = (x * x for x in range(3))     # 生成器也是迭代器
>>> list(squares)
[0, 1, 4]
>>> list(squares)
[]
```

这是一个很常见的 bug 来源：把 `map`、`filter`、`zip` 或生成器的结果存到变量里，用了两次。需要多次遍历时，先 `list()` 一下。

## 自己实现迭代器

用类实现需要维护状态和两个方法：

```python
class Countdown:
    def __init__(self, start):
        self.start = start

    def __iter__(self):
        n = self.start
        while n > 0:
            yield n
            n -= 1

assert list(Countdown(3)) == [3, 2, 1]
assert list(Countdown(3)) == [3, 2, 1]     # 每次 iter() 都是新的生成器，可以反复遍历
```

这里取巧了：`__iter__` 本身写成生成器函数。这是实现**可迭代容器**的推荐方式，比手写 `__next__` 简单得多，而且天然支持多次遍历。

![图：生成器管道与生成器的状态](../assets/figures/generator-pipeline.svg){.aig-svg}

## 生成器函数

函数体里出现 `yield`，调用它就不会执行函数体，而是返回一个**生成器对象**。每次 `next()` 时执行到下一个 `yield` 暂停，并把值交出去；局部变量都保留着，下次从暂停处继续。

```pycon
>>> def fib():
...     a, b = 0, 1
...     while True:           # 无限序列也没问题：按需计算
...         yield a
...         a, b = b, a + b
...
>>> gen = fib()
>>> [next(gen) for _ in range(10)]
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

### 为什么要用生成器：惰性求值

列表一次性把所有元素放进内存；生成器一次只产生一个。

```pycon
>>> import sys
>>> sys.getsizeof([x for x in range(1_000_000)]) > 8_000_000
True
>>> sys.getsizeof(x for x in range(1_000_000)) < 500
True
```

所以把结果直接交给 `sum`、`max`、`any`、`"".join` 这类消费函数时，用生成器表达式（去掉方括号）就好：

```py
total = sum(x * x for x in data)
has_error = any("ERROR" in line for line in lines)   # 找到第一个就停止
```

## 用生成器搭建数据管道

生成器可以像 Unix 管道一样串起来，每一步只处理一行，内存占用和文件大小无关。下面用 `io.StringIO` 模拟一个日志文件：

```python
import io

LOG = io.StringIO("""\
2026-09-24 10:00:01 INFO  user=amy action=login
2026-09-24 10:00:05 ERROR user=bob action=pay msg=timeout
# comment line
2026-09-24 10:01:10 ERROR user=amy action=pay msg=declined
2026-09-24 10:02:00 INFO  user=cat action=logout
""")

def read_lines(f):
    for line in f:
        yield line.rstrip("\n")

def skip_comments(lines):
    return (ln for ln in lines if ln and not ln.startswith("#"))

def parse(lines):
    for ln in lines:
        date, time, level, *pairs = ln.split()
        yield {"level": level, **dict(p.split("=", 1) for p in pairs)}

def only(level, records):
    return (r for r in records if r["level"] == level)

errors = only("ERROR", parse(skip_comments(read_lines(LOG))))
assert [e["user"] for e in errors] == ["bob", "amy"]
```

每个函数只做一件事，可以单独测试、自由组合。整个管道在最后被消费之前，一行都不会读。

## `yield from`：委托给子生成器

`yield from iterable` 相当于 `for x in iterable: yield x`，但还会正确传递 `send()`、异常和子生成器的返回值。最常见的用途是递归生成器：

```pycon
>>> def flatten(items):
...     for x in items:
...         if isinstance(x, (list, tuple)):
...             yield from flatten(x)
...         else:
...             yield x
...
>>> list(flatten([1, [2, [3, (4, 5)]], 6]))
[1, 2, 3, 4, 5, 6]
```

## 生成器的其他能力

生成器还支持 `send(value)`（向暂停处传值）、`throw(exc)`（在暂停处抛异常）、`close()`。这是 Python 早期实现协程的方式。现在写异步代码用 `async`/`await`（见 [asyncio](../concurrency/asyncio.md)），日常很少直接用 `send`，了解即可。

一个仍然常用的细节：生成器被关闭或回收时，会在暂停处触发 `GeneratorExit`，所以 `try/finally` 里的清理代码一定会执行。`contextlib.contextmanager` 就是利用这一点，见[异常与上下文管理器](errors-context.md)。

## `itertools`：迭代器工具箱

`itertools` 里的函数都返回迭代器，惰性、省内存，而且是 C 实现的，很快。

### 无限迭代器

```pycon
>>> from itertools import count, cycle, repeat, islice
>>> list(islice(count(10, 5), 4))          # 10, 15, 20, ...
[10, 15, 20, 25]
>>> list(islice(cycle("AB"), 5))
['A', 'B', 'A', 'B', 'A']
>>> list(zip("abc", repeat(0)))
[('a', 0), ('b', 0), ('c', 0)]
```

`islice` 是对迭代器做切片的方法（迭代器不支持 `[a:b]`）。

### 连接、切分、过滤

```pycon
>>> from itertools import chain, batched, pairwise, takewhile, dropwhile, compress
>>> list(chain([1, 2], (3,), "ab"))
[1, 2, 3, 'a', 'b']
>>> list(chain.from_iterable([[1, 2], [3]]))       # 展平一层
[1, 2, 3]
>>> list(batched(range(7), 3))                     # 分批，3.12+
[(0, 1, 2), (3, 4, 5), (6,)]
>>> list(pairwise([1, 4, 9, 16]))                  # 相邻两两配对
[(1, 4), (4, 9), (9, 16)]
>>> [b - a for a, b in pairwise([1, 4, 9, 16])]    # 求差分
[3, 5, 7]
>>> list(takewhile(lambda x: x < 3, [1, 2, 5, 1]))
[1, 2]
>>> list(dropwhile(lambda x: x < 3, [1, 2, 5, 1]))
[5, 1]
>>> list(compress("abcd", [1, 0, 1, 0]))
['a', 'c']
```

`batched` <span class="since">3.12+</span> 在批量写数据库、分批调用 API 时非常方便；3.13 起还可以传 `strict=True`，要求最后一批必须是满的。

### `groupby`：对**相邻**的相同键分组

```pycon
>>> from itertools import groupby
>>> data = ["apple", "avocado", "banana", "apricot"]
>>> [(k, list(g)) for k, g in groupby(data, key=lambda w: w[0])]
[('a', ['apple', 'avocado']), ('b', ['banana']), ('a', ['apricot'])]
>>> data.sort(key=lambda w: w[0])
>>> [(k, list(g)) for k, g in groupby(data, key=lambda w: w[0])]
[('a', ['apple', 'avocado', 'apricot']), ('b', ['banana'])]
```

`groupby` 只合并**连续**的相同键，所以通常要先按同一个 key 排序。如果不想排序，用 `defaultdict(list)` 分组。另外，每个组 `g` 也是迭代器，必须在前进到下一个组之前用掉。

### 累积与组合

```pycon
>>> from itertools import accumulate, product, permutations, combinations
>>> list(accumulate([1, 2, 3, 4]))                 # 前缀和
[1, 3, 6, 10]
>>> list(accumulate([3, 1, 4, 1, 5], max))         # 前缀最大值
[3, 3, 4, 4, 5]
>>> list(product("ab", [0, 1]))                    # 笛卡尔积，代替嵌套循环
[('a', 0), ('a', 1), ('b', 0), ('b', 1)]
>>> list(combinations("abc", 2))
[('a', 'b'), ('a', 'c'), ('b', 'c')]
>>> len(list(permutations(range(4), 2)))
12
```

### `tee` 和 `zip_longest`

```pycon
>>> from itertools import tee, zip_longest
>>> a, b = tee(iter([1, 2, 3]))       # 把一个迭代器复制成两个
>>> list(a), list(b)
([1, 2, 3], [1, 2, 3])
>>> list(zip_longest("ab", [1], fillvalue="-"))
[('a', 1), ('b', '-')]
```

`tee` 之后就不要再使用原来的迭代器了。如果一个副本远远跑在另一个前面，`tee` 需要缓存中间的所有元素，这时不如直接 `list()`。

!!! interview "面试怎么答"
    迭代器题：可迭代对象能用 `iter()` 产生迭代器，迭代器是一次性的，所以同一个生成器遍历第二次是空的；生成器函数按需产出，适合把大文件处理写成内存恒定的管道；`yield from` 把迭代（以及 `send` 和返回值）委托给子生成器；交给 `sum`、`any`、`join` 消费时用生成器表达式不建中间列表。`groupby` 只合并相邻的相同键，分组前要先排序。推理服务里的流式输出（逐 token 返回）就是生成器 / 异步生成器的典型用法。

## 练习

**1. 滑动窗口。** 写生成器 `window(iterable, n)`，产出长度为 `n` 的滑动窗口元组。`window("abcde", 3)` 产出 `('a','b','c')`、`('b','c','d')`、`('c','d','e')`。要求对无限迭代器也能工作。

??? success "参考答案"
    ```python
    from collections import deque
    from itertools import count, islice

    def window(iterable, n):
        it = iter(iterable)
        buf = deque(islice(it, n), maxlen=n)
        if len(buf) == n:
            yield tuple(buf)
        for x in it:
            buf.append(x)          # maxlen 会自动挤掉最旧的元素
            yield tuple(buf)

    assert list(window("abcde", 3)) == [("a", "b", "c"), ("b", "c", "d"), ("c", "d", "e")]
    assert list(window("ab", 3)) == []
    assert next(window(count(), 2)) == (0, 1)
    ```

    `itertools` 官方文档末尾的配方（recipes）里有一个 `sliding_window`，思路相同。

**2. 保序去重。** 写 `unique(iterable, key=None)`，惰性地产出第一次出现的元素，`key` 用于决定"什么算重复"。

??? success "参考答案"
    ```python
    def unique(iterable, key=None):
        seen = set()
        for x in iterable:
            k = x if key is None else key(x)
            if k not in seen:
                seen.add(k)
                yield x

    assert list(unique([3, 1, 3, 2, 1])) == [3, 1, 2]
    assert list(unique(["a", "B", "A", "b"], key=str.lower)) == ["a", "B"]
    ```

    如果不需要惰性、元素也都可哈希，`list(dict.fromkeys(items))` 是一行的保序去重写法。

**3. 按块读取大文件。** 写 `read_chunks(f, size)`，每次从文件对象读 `size` 个字符，直到读完。提示：内置函数 `iter` 有一个很少人知道的两参数形式 `iter(callable, sentinel)`。

??? success "参考答案"
    ```python
    import io
    from functools import partial

    def read_chunks(f, size=8192):
        return iter(partial(f.read, size), "")      # 读到空字符串就停止

    f = io.StringIO("abcdefg")
    assert list(read_chunks(f, 3)) == ["abc", "def", "g"]
    ```

    `iter(callable, sentinel)` 会反复调用 `callable()`，直到返回值等于 `sentinel`。读二进制文件时哨兵换成 `b""`。

**4. 无限素数生成器。** 写 `primes()`，无限产出素数。然后用 `itertools` 一行求出前 10 个素数，以及小于 100 的素数个数。

??? success "参考答案"
    ```python
    from itertools import count, islice, takewhile

    def primes():
        found = []
        for n in count(2):
            if all(n % p for p in takewhile(lambda p: p * p <= n, found)):
                found.append(n)
                yield n

    assert list(islice(primes(), 10)) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    assert sum(1 for _ in takewhile(lambda p: p < 100, primes())) == 25
    ```

    只需要用已找到的、平方不超过 `n` 的素数去试除。`sum(1 for _ in ...)` 是统计迭代器长度的惯用法。

## 小结

- [x] 可迭代对象能产生迭代器；迭代器是一次性的。
- [x] 给容器实现 `__iter__` 时，写成生成器函数最简单。
- [x] 数据交给 `sum`/`any`/`join` 消费时用生成器表达式，省内存。
- [x] 用多个小生成器串成管道，处理大文件内存恒定。
- [x] 熟记 `islice`、`chain`、`batched`、`pairwise`、`groupby`（先排序）、`accumulate`、`product`。
