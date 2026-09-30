# 函数进阶

<p class="lead">函数是 Python 组织代码的基本单位。这一章讲清楚参数的各种形式、作用域规则、闭包，以及 <code>functools</code> 和 <code>operator</code> 里那些让代码更简洁的工具。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `def f(a, /, b, *, c)` 里，`a`、`b`、`c` 分别只能怎么传？
    2. 下面代码为什么报 `UnboundLocalError`？
       ```py
       count = 0
       def inc():
           count += 1
       ```
    3. `[lambda: i for i in range(3)]` 里每个函数调用后返回什么？怎么修？
    4. `functools.partial` 和 `lambda` 包装有什么区别？
    5. `lru_cache` 对参数有什么要求？

??? success "自测参考答案（先自己答，再展开对照）"
    1. `a` 只能按位置传，`b` 按位置或关键字都行，`c` 只能按关键字传。
    2. 函数体里对 `count` 赋值（`+=` 也是赋值），它就成了局部变量；执行 `count += 1` 时要先读这个局部变量，而它还没有值。要修改全局变量得先声明 `global count`（嵌套函数里用 `nonlocal`）。
    3. 三个函数都返回 2：闭包捕获的是变量 `i` 本身，调用时循环早已结束。修复：用默认参数绑定当时的值 `lambda i=i: i`，或者用 `functools.partial`。
    4. `partial` 预先绑定的参数值在创建时就固定下来，它是一个可以查看 `func`、`args`、`keywords` 的对象，可以被 pickle；`lambda` 在调用时才去查找外面的变量，有闭包的延迟绑定问题。
    5. 所有参数都必须可哈希（它们要组成缓存的键）；还要注意缓存会一直持有这些参数和结果（内存），以及函数应该是纯函数（同样的输入总得到同样的输出）。

## 参数的五种形式

```python
def api(pos_only, /, normal, *args, kw_only, **kwargs):
    return pos_only, normal, args, kw_only, kwargs

print(api(1, 2, 3, 4, kw_only=5, extra=6))
# (1, 2, (3, 4), 5, {'extra': 6})
```

| 形式 | 写法 | 调用方怎么传 |
| --- | --- | --- |
| 仅位置参数 | `/` 之前 | 只能按位置 |
| 普通参数 | `/` 和 `*` 之间 | 按位置或按名字都行 |
| 可变位置参数 | `*args` | 多余的位置参数收集成元组 |
| 仅关键字参数 | `*` 或 `*args` 之后 | 只能按名字 |
| 可变关键字参数 | `**kwargs` | 多余的关键字参数收集成字典 |

### 什么时候用仅关键字参数

布尔开关和容易混淆的参数应该强制按名字传，调用处一眼能看懂：

```pycon
>>> def connect(host, port, *, timeout=10, ssl=True):
...     return f"{host}:{port} timeout={timeout} ssl={ssl}"
...
>>> connect("db", 5432, timeout=3)
'db:5432 timeout=3 ssl=True'
>>> connect("db", 5432, 3)
Traceback (most recent call last):
  ...
TypeError: connect() takes 2 positional arguments but 3 were given
```

### 什么时候用仅位置参数

1. 参数名没有意义，不希望调用方依赖它（将来可以随意改名），比如 `def distance(p, q, /)`。
2. 需要同时接受任意关键字参数，又不想和某个参数名冲突：

```pycon
>>> def render(template, /, **context):
...     return template.format(**context)
...
>>> render("{template} is a word", template="this")   # 不会和第一个参数冲突
'this is a word'
```

### 默认参数只求值一次

默认值在 `def` 执行时求值，之后所有调用共享同一个对象。可变的默认值（列表、字典）几乎一定是 bug，统一用 `None` 代替：

```python
def append_to(item, target=None):
    if target is None:
        target = []
    target.append(item)
    return target

assert append_to(1) == [1]
assert append_to(2) == [2]
```

同理，`def log(msg, when=datetime.now())` 里的时间是函数定义时的时间，不是调用时的。

### 调用时解包

```pycon
>>> def point(x, y, z=0):
...     return (x, y, z)
...
>>> args = [1, 2]
>>> kwargs = {"z": 3}
>>> point(*args, **kwargs)
(1, 2, 3)
```

## 作用域：LEGB

Python 按 **L**ocal → **E**nclosing → **G**lobal → **B**uiltin 的顺序查找名字：

```python
x = "global"

def outer():
    x = "enclosing"
    def inner():
        return x          # 找到 enclosing 里的 x
    return inner()

assert outer() == "enclosing"
assert len("abc") == 3    # len 在 builtin 作用域里
```

关键规则：**只要函数体内对某个名字赋值（包括 `+=`），这个名字在整个函数内就是局部变量**。这是编译时决定的，所以下面的代码会在读取时报错：

```pycon
>>> count = 0
>>> def inc():
...     count += 1
...
>>> inc()
Traceback (most recent call last):
  ...
UnboundLocalError: cannot access local variable 'count' where it is not associated with a value
```

要修改外层的变量，需要显式声明：

- 修改模块级变量：`global count`
- 修改外层函数的变量：`nonlocal count`

!!! tip "尽量少用 `global`"
    需要在多次调用间保存状态时，优先考虑闭包、类实例，或者把状态作为参数和返回值传递。`global` 会让函数的行为依赖隐藏的外部状态，难以测试。

## 闭包

内层函数引用了外层函数的变量，并且在外层函数返回后仍然能访问这些变量，这就是闭包。

```pycon
>>> def make_counter():
...     count = 0
...     def inc():
...         nonlocal count
...         count += 1
...         return count
...     return inc
...
>>> c1, c2 = make_counter(), make_counter()
>>> c1(), c1(), c2()
(1, 2, 1)
>>> c1.__closure__[0].cell_contents     # 闭包捕获的变量存在 cell 里
2
```

### 延迟绑定陷阱

闭包捕获的是**变量**，不是变量当时的值。循环结束后变量是最后一个值：

```pycon
>>> funcs = [lambda: i for i in range(3)]
>>> [f() for f in funcs]
[2, 2, 2]
```

两种修法：

```pycon
>>> funcs = [lambda i=i: i for i in range(3)]      # 默认参数在定义时求值
>>> [f() for f in funcs]
[0, 1, 2]
>>> from functools import partial
>>> def identity(x):
...     return x
...
>>> funcs = [partial(identity, i) for i in range(3)]
>>> [f() for f in funcs]
[0, 1, 2]
```

## lambda 与高阶函数

`lambda` 只适合写**一个表达式**的小函数，通常作为参数传给别的函数。需要名字、需要多行、需要文档时，用 `def`。

`map` 和 `filter` 在 Python 里大多可以被推导式替代，后者通常更易读：

```pycon
>>> nums = [1, 2, 3, 4]
>>> list(map(lambda x: x * x, filter(lambda x: x % 2 == 0, nums)))
[4, 16]
>>> [x * x for x in nums if x % 2 == 0]       # 更推荐
[4, 16]
```

当已经有现成函数时，`map` 很简洁：`list(map(int, "1 2 3".split()))`。

## `operator`：代替简单的 lambda

```pycon
>>> from operator import itemgetter, attrgetter, methodcaller
>>> rows = [("amy", 95), ("bob", 90)]
>>> sorted(rows, key=itemgetter(1))
[('bob', 90), ('amy', 95)]
>>> itemgetter(0, 1)({"0": "x", 0: "a", 1: "b"})
('a', 'b')
>>> list(map(methodcaller("upper"), ["a", "b"]))
['A', 'B']
```

`attrgetter("address.city")` 还支持点号访问嵌套属性。它们比等价的 lambda 更快，也更能表达意图。

## `functools`：函数工具箱

### `partial`：固定部分参数

```pycon
>>> from functools import partial
>>> int_from_hex = partial(int, base=16)
>>> int_from_hex("ff")
255
>>> int_from_hex.func, int_from_hex.keywords
(<class 'int'>, {'base': 16})
```

和 lambda 相比，`partial` 对象可以被 pickle（能传给子进程），并且能看到它包装了什么。

### `cache` 与 `lru_cache`：记忆化

```pycon
>>> from functools import cache, lru_cache
>>> @cache
... def fib(n):
...     return n if n < 2 else fib(n - 1) + fib(n - 2)
...
>>> fib(100)
354224848179261915075
>>> fib.cache_info()
CacheInfo(hits=98, misses=101, maxsize=None, currsize=101)
```

- `@cache` 没有容量上限；`@lru_cache(maxsize=128)` 超过容量时淘汰最久未使用的结果。
- 参数必须**可哈希**，所以不能传列表、字典。
- 只适合**纯函数**（相同输入永远相同输出，无副作用）。
- 用在方法上要小心：`self` 也是缓存的键，缓存会让实例一直无法被回收。方法级缓存可以考虑 `functools.cached_property`（见[面向对象](oop.md)）。

### `reduce`：折叠

```pycon
>>> from functools import reduce
>>> import operator
>>> reduce(operator.mul, [1, 2, 3, 4], 1)
24
```

能用 `sum`、`max`、`any`、`all`、`math.prod`、`"".join` 的地方就别用 `reduce`，它们更直白。

### `singledispatch`：按类型分派

根据**第一个参数的类型**选择实现，是替代长串 `isinstance` 判断的好办法：

```python
from datetime import date
from functools import singledispatch

@singledispatch
def to_json(obj):
    raise TypeError(f"cannot serialize {type(obj).__name__}")

@to_json.register
def _(obj: date):
    return obj.isoformat()

@to_json.register
def _(obj: set):
    return sorted(obj)

@to_json.register(list)
@to_json.register(tuple)
def _(obj):
    return [to_json(x) if not isinstance(x, (int, str)) else x for x in obj]

assert to_json(date(2026, 9, 24)) == "2026-09-24"
assert to_json({3, 1, 2}) == [1, 2, 3]
assert to_json((1, date(2026, 1, 1))) == [1, "2026-01-01"]
```

类的方法用 `functools.singledispatchmethod`。

### 其他

- `functools.wraps`：写装饰器必备，见下一章。
- `functools.total_ordering`：只写 `__eq__` 和一个比较方法，自动补全其余比较方法，见[面向对象](oop.md)。
- `functools.cached_property`：只计算一次的属性。

## 函数的内省

函数对象带有丰富的元数据，框架（比如 FastAPI、pytest）就是靠这些实现"根据参数自动注入"的：

```pycon
>>> import inspect
>>> def handler(request, user_id: int, *, verbose: bool = False) -> dict:
...     ...
...
>>> sig = inspect.signature(handler)
>>> [(p.name, p.kind.name) for p in sig.parameters.values()]
[('request', 'POSITIONAL_OR_KEYWORD'), ('user_id', 'POSITIONAL_OR_KEYWORD'), ('verbose', 'KEYWORD_ONLY')]
>>> sig.parameters["user_id"].annotation
<class 'int'>
>>> sig.bind("req", "42").arguments
{'request': 'req', 'user_id': '42'}
```

!!! interview "面试怎么答"
    函数题常考四个坑：参数种类（`/` 之前只能按位置、`*` 之后只能按关键字）；可变默认值只在定义时求值一次，要用 `None` 代替；在函数里给变量赋值会让它成为局部变量，读外层变量前先赋值就是 `UnboundLocalError`，修改外层要用 `nonlocal` / `global`；闭包捕获的是变量不是值，`[lambda: i for i in range(3)]` 全返回 2，用默认参数 `lambda i=i: i` 或 `functools.partial` 固定。再提 `lru_cache` 要求参数可哈希、`singledispatch` 按第一个参数的类型分派。

## 练习

**1. 函数组合。** 写 `compose(*funcs)`，返回一个新函数，从右往左依次调用：`compose(f, g, h)(x) == f(g(h(x)))`。不传函数时返回恒等函数。

??? success "参考答案"
    ```python
    from functools import reduce

    def compose(*funcs):
        def composed(x):
            return reduce(lambda acc, f: f(acc), reversed(funcs), x)
        return composed

    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert compose(inc, double)(5) == 11     # inc(double(5))
    assert compose(double, inc)(5) == 12     # double(inc(5))
    assert compose()(5) == 5
    ```

**2. 修复按钮回调。** 下面代码想为每个按钮绑定"打印自己的名字"的回调，结果全都打印 `C`。用两种方式修复。

```py
callbacks = {}
for name in ["A", "B", "C"]:
    callbacks[name] = lambda: print(name)
```

??? success "参考答案"
    ```python
    from functools import partial

    # 方式一：默认参数在定义时绑定当前值
    callbacks = {}
    for name in ["A", "B", "C"]:
        callbacks[name] = lambda name=name: name
    assert [f() for f in callbacks.values()] == ["A", "B", "C"]

    # 方式二：partial，更不容易被误传参数覆盖
    def echo(value):
        return value

    callbacks = {name: partial(echo, name) for name in ["A", "B", "C"]}
    assert [f() for f in callbacks.values()] == ["A", "B", "C"]
    ```

    方式一的缺点是调用方可以传参覆盖 `name`；方式二语义更明确。

**3. 网格路径计数。** 在 `m × n` 的网格里从左上角走到右下角，每次只能向右或向下，有多少条路径？写一个递归版本，用 `@cache` 让 `paths(18, 18)` 瞬间算完。

??? success "参考答案"
    ```python
    from functools import cache

    @cache
    def paths(m, n):
        if m == 1 or n == 1:
            return 1
        return paths(m - 1, n) + paths(m, n - 1)

    assert paths(3, 3) == 6
    assert paths(18, 18) == 2333606220
    ```

    没有缓存时，递归树是指数级的；有了缓存，每个 `(m, n)` 只算一次，复杂度是 O(m·n)。

**4. 设计一个好用的 API。** 写 `retry_call(func, *args, retries=3, delay=0.0, exceptions=(Exception,), **kwargs)`：调用 `func(*args, **kwargs)`，遇到指定异常就重试，最多 `retries` 次，最后一次仍失败就把异常抛出。思考：为什么 `retries` 等参数要设计成仅关键字参数？

??? success "参考答案"
    ```python
    import time

    def retry_call(func, *args, retries=3, delay=0.0, exceptions=(Exception,), **kwargs):
        for attempt in range(1, retries + 1):
            try:
                return func(*args, **kwargs)
            except exceptions:
                if attempt == retries:
                    raise
                time.sleep(delay)

    calls = []
    def flaky(x):
        calls.append(x)
        if len(calls) < 3:
            raise ConnectionError("try again")
        return x * 10

    assert retry_call(flaky, 4) == 40
    assert len(calls) == 3
    ```

    `*args` 之后的参数自动成为仅关键字参数。这样 `retry_call` 自己的配置不会和被调用函数的位置参数混在一起，调用方也不可能误把业务参数传给 `retries`。

## 小结

- [x] 用 `*` 强制关键字参数，让调用处可读；用 `/` 保护不想暴露的参数名。
- [x] 可变默认值用 `None` 代替。
- [x] 函数内对变量赋值会让它变成局部变量；修改外层变量用 `nonlocal`/`global`。
- [x] 闭包捕获的是变量不是值，循环里创建函数要特别小心。
- [x] `partial`、`cache`、`singledispatch`、`operator` 能让很多代码变得更短更清楚。
