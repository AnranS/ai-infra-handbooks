# 惯用法与常见坑

<p class="lead">这一页是全书的"速查表"：上半部分是地道的写法，下半部分是最常踩的坑。每一条都尽量用最短的例子说明，详细解释在对应的章节里。适合在写完代码后对照检查，或者面试前快速过一遍。</p>

## 地道的写法

### 遍历

```py
# 需要下标时用 enumerate
for i, item in enumerate(items):          # 而不是 for i in range(len(items))
    ...

# 同时遍历多个序列用 zip（长度应当相同时加 strict=True）
for name, score in zip(names, scores, strict=True):
    ...

# 遍历字典的键值对
for key, value in d.items():
    ...

# 反向、排序遍历
for x in reversed(items): ...
for x in sorted(items, key=len): ...
```

### 解包与交换

```pycon
>>> a, b = 1, 2
>>> a, b = b, a
>>> head, *tail = [1, 2, 3]
>>> (a, b), head, tail
((2, 1), 1, [2, 3])
```

### 判断

```pycon
>>> x = 5
>>> 0 < x < 10                            # 链式比较
True
>>> x in {1, 3, 5}                        # 多值判断用 in + 集合
True
>>> items = [0, "", None, 3]
>>> any(items), all(items)
(True, False)
```

- 判断空容器用 `if not items:`，判断 `None` 用 `if x is None:`。
- `isinstance(x, (int, float))` 而不是 `type(x) == int`。

### 构建集合

```pycon
>>> words = ["apple", "Bob", "cat"]
>>> [w.upper() for w in words if len(w) > 2]
['APPLE', 'BOB', 'CAT']
>>> {w: len(w) for w in words}
{'apple': 5, 'Bob': 3, 'cat': 3}
>>> ", ".join(words)                      # 拼接字符串用 join
'apple, Bob, cat'
>>> list(dict.fromkeys([3, 1, 3, 2]))     # 保序去重
[3, 1, 2]
```

### 字典

```py
value = d.get(key, default)               # 而不是 if key in d: ... else: ...
d.setdefault(key, []).append(x)           # 或者用 defaultdict(list)
merged = defaults | overrides             # 合并
counts = Counter(words)                   # 计数
```

### 海象运算符 `:=`

在表达式中赋值，避免重复计算或重复调用：

```pycon
>>> import re
>>> if (m := re.search(r"(\d+)", "order 42")):
...     print(m[1])
...
42
>>> data = [3, 8, 1, 9]
>>> [y for x in data if (y := x * x) > 10]
[64, 81]
```

读取数据块直到结束，也是它的经典用法：`while chunk := f.read(8192): ...`。

### `for ... else`

循环**没有被 `break` 打断**时执行 `else`，适合"找不到时怎么办"：

```pycon
>>> for n in [4, 6, 8]:
...     if n % 2:
...         print("found odd", n)
...         break
... else:
...     print("no odd number")
...
no odd number
```

### 资源管理

- 文件、锁、连接一律用 `with`。
- 路径用 `pathlib.Path`，打开文本文件写 `encoding="utf-8"`。

### 其他

- 用 `_` 表示不关心的值：`for _ in range(3)`、`_, ext = os.path.splitext(p)`。
- 大数字加下划线：`1_000_000`。
- 返回多个值时返回元组或 dataclass，不要返回一个语义不明的列表。
- 函数的开关参数设为仅关键字参数：`def fetch(url, *, retry=False)`。

## 常见的坑

### 可变默认参数

```pycon
>>> def add(x, bucket=[]):
...     bucket.append(x)
...     return bucket
...
>>> add(1), add(2)
([1, 2], [1, 2])
```

默认值只在定义时求值一次。用 `None` 做默认值，在函数里创建新列表。详见[函数进阶](../core/functions.md#默认参数只求值一次)。

### 闭包的延迟绑定

```pycon
>>> fs = [lambda: i for i in range(3)]
>>> [f() for f in fs]
[2, 2, 2]
```

闭包捕获的是变量不是值。用默认参数 `lambda i=i: i` 或 `functools.partial` 修复。详见[函数进阶](../core/functions.md#延迟绑定陷阱)。

### 用 `*` 复制嵌套列表

```pycon
>>> grid = [[0] * 2] * 2
>>> grid[0][0] = 1
>>> grid
[[1, 0], [1, 0]]
```

外层复制的是同一个内层列表的引用。用 `[[0] * 2 for _ in range(2)]`。

### 遍历时修改容器

```pycon
>>> nums = [1, 2, 2, 3]
>>> for n in nums:
...     if n == 2:
...         nums.remove(n)
...
>>> nums                                  # 第二个 2 被跳过了
[1, 2, 3]
>>> d = {"a": 1, "b": 2}
>>> for k in d:
...     if d[k] == 1:
...         del d[k]
...
Traceback (most recent call last):
  ...
RuntimeError: dictionary changed size during iteration
```

构建一个新的容器（`[n for n in nums if n != 2]`），或者遍历副本（`for k in list(d):`）。

### 用 `is` 比较值

`a is b` 比较的是"是不是同一个对象"。小整数和部分字符串会被缓存，所以有时 `is` 恰好返回 `True`，换个值就不行了。**比较值永远用 `==`**，`is` 只用于 `None`、`True`、`False` 和哨兵对象。

### 浮点数精度

```pycon
>>> 0.1 + 0.2 == 0.3
False
>>> import math
>>> math.isclose(0.1 + 0.2, 0.3)
True
>>> from decimal import Decimal
>>> Decimal("0.1") + Decimal("0.2") == Decimal("0.3")
True
>>> round(2.5), round(3.5), round(2.675, 2)       # 银行家舍入，以及二进制表示误差
(2, 4, 2.67)
```

比较浮点数用 `math.isclose`；金额用 `Decimal`（从字符串构造）或者整数"分"。`round` 采用"四舍六入五成双"，不是四舍五入。

### 整除和取模的符号

```pycon
>>> -7 // 2, -7 % 2
(-4, 1)
>>> int(-7 / 2)
-3
```

`//` 向负无穷取整，不是向零截断。和其他语言交互（比如对照 C/Java 的结果）时要注意。

### `bool` 是 `int` 的子类

```pycon
>>> True + True, isinstance(True, int)
(2, True)
>>> {1: "one", True: "true"}
{1: 'true'}
```

`True == 1` 且哈希相同，所以在字典里它们是同一个键。校验"是整数"时，如果要排除布尔值，需要额外判断 `not isinstance(x, bool)`。

### 就地修改的方法返回 `None`

```pycon
>>> items = [3, 1, 2]
>>> result = items.sort()
>>> print(result)
None
>>> sorted([3, 1, 2])                     # 需要返回值时用 sorted
[1, 2, 3]
```

`list.sort()`、`list.append()`、`list.reverse()`、`dict.update()`、`random.shuffle()` 都是原地修改并返回 `None`。

### 迭代器只能用一次

```pycon
>>> squares = map(lambda x: x * x, [1, 2, 3])
>>> sum(squares), sum(squares)
(14, 0)
```

`map`、`filter`、`zip`、生成器、文件对象都是一次性的。需要多次使用时先 `list()`。

### `except` 的变量在块结束后被删除

```py
try:
    1 / 0
except ZeroDivisionError as e:
    pass
print(e)          # NameError：e 在 except 块结束时被删除了
```

需要在块外使用异常对象时，先赋值给另一个变量。

### 捕获过宽的异常

```py
try:
    value = compute(data)
except Exception:          # 连拼写错误导致的 NameError 也被吞掉了
    value = None
```

只捕获你预期的、知道怎么处理的具体异常，并且让 `try` 块尽量小。详见[异常与上下文管理器](../core/errors-context.md)。

### 脚本和标准库模块同名

把自己的文件命名为 `random.py`、`json.py`、`email.py`、`test.py`，`import random` 就会导入你自己的文件，报出莫名其妙的 `AttributeError`。3.13 起错误信息会提示你可能是这个原因，但最好一开始就避开这些名字。

### 循环导入

`a.py` 导入 `b.py`，`b.py` 又导入 `a.py`，就可能出现 `ImportError: cannot import name ... (most likely due to a circular import)`。解决办法：

1. 把两边共同依赖的东西抽到第三个模块；
2. 只在函数内部导入（延迟到调用时）；
3. 如果只是类型标注需要，放在 `if TYPE_CHECKING:` 块里导入。

循环导入往往说明模块的职责划分有问题。

### 相对导入与运行方式

在包里的模块使用了相对导入（`from .utils import x`），直接 `python mypkg/cli.py` 运行会报 `ImportError: attempted relative import with no known parent package`。应该用模块方式运行：`python -m mypkg.cli`，或者通过 `[project.scripts]` 定义的命令运行。

### 在协程里调用阻塞函数

`async def` 里的 `time.sleep()`、`requests.get()` 会让整个事件循环停住。用 `await asyncio.sleep()`、异步的 HTTP 库，或者 `await asyncio.to_thread(func)`。详见 [asyncio](../concurrency/asyncio.md#在异步代码里调用阻塞代码)。

### 时间没有时区

`datetime.now()` 和 `datetime.utcnow()`（已弃用）返回的都是不带时区的时间，跨时区、跨服务器时会出错。用 `datetime.now(UTC)`，详见[常用标准库](../engineering/stdlib.md#datetime-与-zoneinfo处理时间)。

### 默认编码

`open("f.txt")` 不写 `encoding` 时，在 Windows 上通常不是 UTF-8。统一写 `encoding="utf-8"`。

## 代码审查清单

提交代码前，对照检查：

- [ ] 没有可变的默认参数，没有可变的类属性被当作实例数据
- [ ] 没有裸 `except:`，没有被吞掉的异常，`try` 块足够小
- [ ] 文件、锁、连接都在 `with` 里
- [ ] 比较 `None` 用 `is`，比较值用 `==`，浮点数用 `isclose`，金额用 `Decimal`
- [ ] 循环里没有对 `list` 做 `in` 判断或 `pop(0)`
- [ ] 时间带时区，文本文件指定编码
- [ ] 公开函数有类型标注和文档字符串
- [ ] 新逻辑有测试，ruff 和 mypy 通过
