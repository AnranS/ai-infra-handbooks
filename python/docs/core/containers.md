# 容器与数据结构

<p class="lead">选对数据结构，代码会同时变短、变快、变清楚。这一章讲内置容器的进阶用法、它们的性能特征，以及 <code>collections</code>、<code>heapq</code>、<code>bisect</code> 这几个常被忽略的利器。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `x in some_list` 和 `x in some_set` 的时间复杂度分别是多少？
    2. 怎么按"先按分数降序、分数相同再按名字升序"排序一个字典列表？
    3. `first, *rest = [1]` 之后 `rest` 是什么？
    4. `defaultdict(list)` 和 `dict.setdefault` 分别适合什么场景？
    5. 要一个"只保留最近 100 条"的缓冲区，用什么最合适？

## 复杂度速查

写代码时脑子里要有这张表。最常见的性能问题就是在循环里对 `list` 做 `in` 判断或者 `pop(0)`。

| 操作 | `list` | `dict` / `set` | `collections.deque` |
| --- | --- | --- | --- |
| 按索引取值 `x[i]` | O(1) | — | O(n)（两端 O(1)） |
| 末尾追加 / 弹出 | O(1) | — | O(1) |
| 开头插入 / 弹出 | **O(n)** | — | O(1) |
| 成员判断 `x in c` | **O(n)** | O(1) | O(n) |
| 插入 / 删除任意位置 | O(n) | O(1) | O(n) |
| 排序 | O(n log n) | — | — |

!!! tip "经验法则"
    - 需要反复判断"在不在"：先转成 `set`。
    - 需要在两端进出（队列、滑动窗口）：用 `deque`。
    - 需要按键查找：用 `dict`，别在列表里线性搜索。

## 序列操作进阶

### 切片

切片 `seq[start:stop:step]` 总是返回**新对象**（浅拷贝），越界不会报错。

```pycon
>>> s = "abcdefgh"
>>> s[::2], s[::-1], s[-3:]
('aceg', 'hgfedcba', 'fgh')
>>> nums = list(range(10))
>>> nums[2:5] = ["x"]          # 切片赋值：长度可以不同
>>> nums
[0, 1, 'x', 5, 6, 7, 8, 9]
>>> del nums[::2]              # 删除偶数位置
>>> nums
[1, 5, 7, 9]
```

切片可以命名，让代码自解释。解析定长格式的数据时很有用：

```pycon
>>> record = "20260924ERROR  disk full"
>>> DATE, LEVEL, MSG = slice(0, 8), slice(8, 15), slice(15, None)
>>> record[DATE], record[LEVEL].strip(), record[MSG]
('20260924', 'ERROR', 'disk full')
```

### 解包

```pycon
>>> first, *middle, last = [1, 2, 3, 4, 5]
>>> first, middle, last
(1, [2, 3, 4], 5)
>>> first, *rest = [1]
>>> rest
[]
>>> a, b = 1, 2
>>> a, b = b, a                # 交换，不需要临时变量
>>> a, b
(2, 1)
>>> for name, (x, y) in [("p1", (0, 1)), ("p2", (3, 4))]:
...     print(name, x + y)
...
p1 1
p2 7
```

`*` 和 `**` 也能用在字面量里合并数据：

```pycon
>>> [*range(3), *"ab"]
[0, 1, 2, 'a', 'b']
>>> defaults = {"host": "localhost", "port": 80}
>>> {**defaults, "port": 8080}
{'host': 'localhost', 'port': 8080}
```

### 排序

`sorted()` 返回新列表，`list.sort()` 原地排序并返回 `None`。两者都是**稳定排序**：键相同的元素保持原来的相对顺序。

```pycon
>>> users = [
...     {"name": "bob", "score": 90},
...     {"name": "amy", "score": 95},
...     {"name": "cat", "score": 90},
... ]
>>> [u["name"] for u in sorted(users, key=lambda u: (-u["score"], u["name"]))]
['amy', 'bob', 'cat']
```

用元组当 key 可以实现多级排序；数值字段取负号实现降序。如果某个字段不能取负（比如字符串要降序），利用稳定性分两次排：先按次要键排，再按主要键排。

```pycon
>>> from operator import itemgetter
>>> rows = [("b", 2), ("a", 2), ("c", 1)]
>>> rows.sort(key=itemgetter(0), reverse=True)   # 次要键：名字降序
>>> rows.sort(key=itemgetter(1))                 # 主要键：数字升序
>>> rows
[('c', 1), ('b', 2), ('a', 2)]
```

`min`、`max` 也接受 `key`，还有 `default` 参数处理空序列：

```pycon
>>> max(users, key=lambda u: u["score"])["name"]
'amy'
>>> max([], default=None) is None
True
```

### `enumerate` 与 `zip`

```pycon
>>> for i, ch in enumerate("abc", start=1):
...     print(i, ch)
...
1 a
2 b
3 c
>>> names, scores = ["amy", "bob"], [95, 90]
>>> dict(zip(names, scores))
{'amy': 95, 'bob': 90}
>>> list(zip(*[(1, "a"), (2, "b")]))       # "解压"：转置
[(1, 2), ('a', 'b')]
```

`zip` 默认在最短的序列结束时停止，数据长度不一致时会**静默丢数据**。确定两边应该等长时，加上 `strict=True`：

```pycon
>>> list(zip([1, 2, 3], "ab", strict=True))
Traceback (most recent call last):
  ...
ValueError: zip() argument 2 is shorter than argument 1
```

## 推导式

推导式是 Python 最有特色的语法之一。列表、字典、集合各有一种，生成器表达式用圆括号。

```pycon
>>> words = ["apple", "Bob", "cat", "Apple"]
>>> [w.lower() for w in words if len(w) > 3]
['apple', 'apple']
>>> {w.lower() for w in words}  == {"apple", "bob", "cat"}
True
>>> {w: len(w) for w in words}
{'apple': 5, 'Bob': 3, 'cat': 3, 'Apple': 5}
>>> sum(len(w) for w in words)            # 生成器表达式：不建中间列表
16
```

多层循环的顺序和写 `for` 语句的顺序一致：

```pycon
>>> matrix = [[1, 2, 3], [4, 5, 6]]
>>> [x for row in matrix for x in row]              # 展平
[1, 2, 3, 4, 5, 6]
>>> [[row[i] for row in matrix] for i in range(3)]  # 转置
[[1, 4], [2, 5], [3, 6]]
```

!!! warning "推导式不是越长越好"
    超过两层循环或者条件很复杂时，改回普通 `for` 循环，或者拆出一个函数。推导式只用来**构建集合**，不要为了副作用（比如 `[print(x) for x in xs]`）而写。

## 字典技巧

从 3.7 起，字典**保持插入顺序**，这是语言规范的一部分。

```pycon
>>> stock = {"apple": 3}
>>> stock.get("pear", 0)                 # 不存在时给默认值
0
>>> stock.setdefault("pear", 0)          # 不存在时插入并返回
0
>>> stock | {"apple": 5, "kiwi": 1}      # 合并，右边覆盖左边
{'apple': 5, 'pear': 0, 'kiwi': 1}
>>> stock |= {"kiwi": 2}                 # 原地合并
>>> stock
{'apple': 3, 'pear': 0, 'kiwi': 2}
```

几个常见操作的地道写法：

```pycon
>>> scores = {"amy": 95, "bob": 90, "cat": 90}
>>> {v: k for k, v in scores.items()}              # 反转（值重复时后者覆盖）
{95: 'amy', 90: 'cat'}
>>> sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:2]   # 按值取前 2
[('amy', 95), ('bob', 90)]
>>> scores.keys() & {"amy", "dan"}                 # 键视图支持集合运算
{'amy'}
```

!!! warning "遍历时不能增删键"
    在 `for k in d:` 里给 `d` 添加或删除键会抛 `RuntimeError: dictionary changed size during iteration`。需要删除时，遍历一个副本：`for k in list(d):`，或者用推导式构建新字典。

## `collections`：专用容器

### `Counter`：计数

```pycon
>>> from collections import Counter
>>> c = Counter("mississippi")
>>> c.most_common(2)
[('i', 4), ('s', 4)]
>>> c["z"]                     # 不存在的键返回 0，不报错
0
>>> c.update("ssz")
>>> c["s"], c["z"]
(6, 1)
>>> Counter(a=3, b=1) - Counter(a=1, b=2)    # 减法只保留正数
Counter({'a': 2})
```

### `defaultdict`：自动初始化

分组、建索引、构造邻接表时特别顺手：

```pycon
>>> from collections import defaultdict
>>> words = ["apple", "avocado", "banana", "blueberry", "cherry"]
>>> by_letter = defaultdict(list)
>>> for w in words:
...     by_letter[w[0]].append(w)
...
>>> dict(by_letter)
{'a': ['apple', 'avocado'], 'b': ['banana', 'blueberry'], 'c': ['cherry']}
```

`defaultdict` 在**读取**不存在的键时也会插入默认值，有时这会导致意外。只想查询时用 `d.get(k)`。

### `deque`：双端队列

两端进出都是 O(1)。设置 `maxlen` 后就是一个自动丢弃旧数据的环形缓冲区：

```pycon
>>> from collections import deque
>>> recent = deque(maxlen=3)
>>> for line in ["a", "b", "c", "d", "e"]:
...     recent.append(line)
...
>>> recent
deque(['c', 'd', 'e'], maxlen=3)
>>> q = deque([1, 2, 3])
>>> q.appendleft(0); q.pop()
3
>>> q.rotate(1)
>>> q
deque([2, 0, 1])
```

### `ChainMap`：分层查找

按顺序在多个字典里查找，适合"命令行参数 > 环境变量 > 默认值"这种分层配置：

```pycon
>>> from collections import ChainMap
>>> defaults = {"debug": False, "port": 80}
>>> env = {"port": 8080}
>>> cli = {"debug": True}
>>> cfg = ChainMap(cli, env, defaults)
>>> cfg["debug"], cfg["port"]
(True, 8080)
```

### `OrderedDict`：需要"移动顺序"时

普通字典已经有序了，`OrderedDict` 现在主要用在需要 `move_to_end` 和 `popitem(last=False)` 的场景，比如 LRU 缓存（见练习）。

`namedtuple` 放在[数据建模](data-classes.md)一章讲。

## `heapq`：优先队列与 Top-K

`heapq` 在普通列表上维护**最小堆**：`heap[0]` 永远是最小值，入堆出堆都是 O(log n)。

```pycon
>>> import heapq
>>> tasks = []
>>> heapq.heappush(tasks, (2, "write docs"))
>>> heapq.heappush(tasks, (1, "fix bug"))
>>> heapq.heappush(tasks, (3, "refactor"))
>>> heapq.heappop(tasks)
(1, 'fix bug')
>>> heapq.nlargest(2, [5, 1, 8, 3, 9])
[9, 8]
>>> heapq.nsmallest(1, [{"n": "a", "p": 3}, {"n": "b", "p": 1}], key=lambda d: d["p"])
[{'n': 'b', 'p': 1}]
```

!!! tip "优先级相同时的比较问题"
    元组比较时，如果优先级相同，会接着比较第二个元素。第二个元素不可比较（比如字典）就会报错。常用做法是插入一个递增的序号：`(priority, next(counter), item)`。

从很多数据里取前 K 个时：K 很小用 `nlargest`/`nsmallest`；K 接近 n 时直接 `sorted(...)[:k]`；只要最大或最小一个就用 `max`/`min`。

`heapq.merge` 可以惰性合并多个**已排序**的序列：

```pycon
>>> list(heapq.merge([1, 4, 7], [2, 5], [3, 6]))
[1, 2, 3, 4, 5, 6, 7]
```

## `bisect`：在有序列表里二分查找

```pycon
>>> import bisect
>>> def grade(score, cutoffs=(60, 70, 80, 90), grades="FDCBA"):
...     return grades[bisect.bisect_right(cutoffs, score)]
...
>>> [grade(s) for s in (33, 60, 77, 90, 100)]
['F', 'D', 'C', 'A', 'A']
>>> xs = [1, 3, 5]
>>> bisect.insort(xs, 4)         # 插入并保持有序
>>> xs
[1, 3, 4, 5]
```

`bisect` 系列函数也支持 `key` 参数，可以在对象列表上按某个字段二分。

## 怎么选

| 需求 | 选择 |
| --- | --- |
| 有序、可重复、按位置访问 | `list` |
| 不可变的一组值，或做字典的键 | `tuple` |
| 去重、快速判断成员、集合运算 | `set` / `frozenset` |
| 按键查找 | `dict` |
| 计数 | `Counter` |
| 分组、建索引 | `defaultdict(list)` |
| 队列、滑动窗口、最近 N 条 | `deque` |
| 不断取最小/最大值 | `heapq` |
| 有序列表里查找和插入 | `bisect` |
| 大量同类型数值 | `array.array`，或者第三方的 NumPy |

## 练习

**1. 词频统计。** 给定一段英文文本，忽略大小写和标点，返回出现最多的 3 个单词及次数。

??? success "参考答案"
    ```python
    import re
    from collections import Counter

    def top_words(text, n=3):
        words = re.findall(r"[a-z']+", text.lower())
        return Counter(words).most_common(n)

    text = "The cat and the hat. The cat sat! And then? The end."
    assert top_words(text) == [("the", 4), ("cat", 2), ("and", 2)]
    ```

    `most_common` 在次数相同时按首次出现的顺序排列，所以 `cat` 排在 `and` 前面。

**2. 字母异位词分组。** 把 `["eat", "tea", "tan", "ate", "nat", "bat"]` 分成 `[["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]`。

??? success "参考答案"
    ```python
    from collections import defaultdict

    def group_anagrams(words):
        groups = defaultdict(list)
        for w in words:
            groups["".join(sorted(w))].append(w)
        return list(groups.values())

    assert group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]) == [
        ["eat", "tea", "ate"], ["tan", "nat"], ["bat"]
    ]
    ```

    排序后的字符串作为"指纹"，相同指纹的就是异位词。字典保持插入顺序，所以分组顺序也是确定的。

**3. 滑动窗口最大值。** 给定列表 `nums` 和窗口大小 `k`，返回每个窗口的最大值，要求 O(n)。例如 `[1, 3, -1, -3, 5, 3, 6, 7]`、`k=3` 返回 `[3, 3, 5, 5, 6, 7]`。

??? success "参考答案"
    用 `deque` 存**下标**，并保持对应的值单调递减。队头永远是当前窗口的最大值。

    ```python
    from collections import deque

    def max_sliding_window(nums, k):
        dq, out = deque(), []
        for i, x in enumerate(nums):
            while dq and nums[dq[-1]] <= x:   # 比 x 小的元素不可能再成为最大值
                dq.pop()
            dq.append(i)
            if dq[0] <= i - k:                # 队头已滑出窗口
                dq.popleft()
            if i >= k - 1:
                out.append(nums[dq[0]])
        return out

    assert max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
    ```

    每个下标最多进出队列各一次，所以总体是 O(n)。

**4. 用 `OrderedDict` 实现 LRU 缓存。** 支持 `get(key)`（不存在返回 `None`）和 `put(key, value)`，容量满时淘汰最久未使用的键。

??? success "参考答案"
    ```python
    from collections import OrderedDict

    class LRUCache:
        def __init__(self, capacity):
            self.capacity = capacity
            self._data = OrderedDict()

        def get(self, key):
            if key not in self._data:
                return None
            self._data.move_to_end(key)          # 标记为最近使用
            return self._data[key]

        def put(self, key, value):
            self._data[key] = value
            self._data.move_to_end(key)
            if len(self._data) > self.capacity:
                self._data.popitem(last=False)   # 弹出最久未使用的

    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")          # a 变成最近使用
    cache.put("c", 3)       # 淘汰 b
    assert cache.get("b") is None
    assert cache.get("a") == 1 and cache.get("c") == 3
    ```

    实际项目里给函数加缓存，直接用 `functools.lru_cache`，见[函数进阶](functions.md)。

## 小结

- [x] 记住复杂度：`list` 的 `in` 和头部操作是 O(n)，`set`/`dict` 查找是 O(1)。
- [x] 多级排序用元组 key；排序是稳定的。
- [x] `zip(strict=True)` 防止静默丢数据。
- [x] 计数用 `Counter`，分组用 `defaultdict`，队列用 `deque`，Top-K 用 `heapq`。
- [x] 推导式用来构建集合，复杂了就改回循环。
