# 装饰器

<p class="lead">装饰器是"接收一个函数、返回一个函数"的函数。理解了这句话和闭包，装饰器就没有任何神秘之处。日志、计时、重试、缓存、权限校验、路由注册，全都可以用它优雅地实现。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `@deco` 写在函数上面，等价于哪一行代码？
    2. 不用 `functools.wraps` 会出什么问题？
    3. 带参数的装饰器（比如 `@retry(times=3)`）为什么要写三层函数？
    4. 两个装饰器叠加时，哪个先执行？
    5. 怎么写一个既能 `@deco` 又能 `@deco(level=2)` 使用的装饰器？

??? success "自测参考答案（先自己答，再展开对照）"
    1. `f = deco(f)`，在函数定义的时候执行一次。
    2. 包装后的函数会丢掉原函数的名字、文档字符串、签名等元数据（`__name__` 变成 `wrapper`），调试、日志和依赖这些信息的工具（比如 pytest、`inspect`）都会受影响。
    3. `retry(times=3)` 先要被调用一次，返回真正的装饰器；装饰器再接收函数，返回包装函数——所以是"接收参数 → 接收函数 → 包装调用"三层。
    4. 装饰时从下往上：离函数近的先装饰；调用时从上往下：最外层的包装最先执行。
    5. 第一个参数默认为 `None`，其余参数只能按关键字传：被直接用作 `@deco` 时第一个参数就是函数，直接装饰并返回；否则（`@deco(level=2)`）返回一个接收函数的装饰器。

## 本质：语法糖

```py
@deco
def f():
    ...
```

完全等价于：

```py
def f():
    ...
f = deco(f)
```

装饰器在**函数定义时**执行一次，不是在每次调用时。它的返回值替换了原来的名字。

```pycon
>>> def announce(func):
...     print(f"decorating {func.__name__}")
...     return func
...
>>> @announce
... def hello():
...     return "hi"
...
decorating hello
>>> hello()
'hi'
```

## 第一个真正的装饰器：计时

大多数装饰器会返回一个**包装函数**，在调用原函数前后做点事：

```python
import functools
import time

def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            print(f"{func.__name__} took {elapsed * 1000:.1f} ms")
    return wrapper

@timed
def slow_add(a, b):
    """Add two numbers slowly."""
    time.sleep(0.01)
    return a + b

assert slow_add(1, 2) == 3          # 打印 slow_add took 10.x ms
assert slow_add.__name__ == "slow_add"
assert slow_add.__doc__ == "Add two numbers slowly."
```

要点：

- `*args, **kwargs` 让包装函数能接受任意参数，原样转发。
- 一定要把原函数的返回值 `return` 出去。
- 用 `try/finally`，即使原函数抛异常也能记录耗时。

### 为什么必须用 `functools.wraps`

不加 `wraps`，被装饰后的函数名字、文档、签名都变成了 `wrapper` 的：

```pycon
>>> def bad_deco(func):
...     def wrapper(*args, **kwargs):
...         return func(*args, **kwargs)
...     return wrapper
...
>>> @bad_deco
... def greet(name):
...     """Greet someone."""
...
>>> greet.__name__, greet.__doc__
('wrapper', None)
```

这会搞坏日志、调试、文档生成，以及依赖函数签名的框架（比如 pytest 的 fixture、FastAPI 的参数解析）。`wraps` 会复制 `__name__`、`__doc__`、`__module__`、`__qualname__` 等属性，并设置 `__wrapped__` 指向原函数，`inspect.signature` 会顺着它找到真实的签名。

```pycon
>>> import functools, inspect
>>> def good_deco(func):
...     @functools.wraps(func)
...     def wrapper(*args, **kwargs):
...         return func(*args, **kwargs)
...     return wrapper
...
>>> @good_deco
... def greet(name, *, loud=False):
...     """Greet someone."""
...
>>> greet.__name__, str(inspect.signature(greet))
('greet', '(name, *, loud=False)')
```

## 带参数的装饰器

`@retry(times=3)` 的执行顺序是：先调用 `retry(times=3)`，**它的返回值**才是真正的装饰器。所以要多包一层：

```python
import functools
import time

def retry(times=3, exceptions=(Exception,), delay=0.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == times:
                        raise
                    print(f"{func.__name__} failed ({e!r}), retry {attempt}/{times - 1}")
                    time.sleep(delay)
        return wrapper
    return decorator

attempts = []

@retry(times=3, exceptions=(ConnectionError,))
def fetch():
    attempts.append(1)
    if len(attempts) < 3:
        raise ConnectionError("network down")
    return "data"

assert fetch() == "data" and len(attempts) == 3
```

三层分别是：**配置层**（接收装饰器参数）→ **装饰层**（接收函数）→ **包装层**（接收调用参数）。

### 同时支持 `@deco` 和 `@deco(...)`

利用"第一个参数是不是函数"来区分两种用法，并把配置参数设为仅关键字参数：

```python
import functools

def log_calls(func=None, *, prefix="CALL"):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            print(f"{prefix} {f.__name__}{args}")
            return f(*args, **kwargs)
        return wrapper

    if func is None:          # 用法 @log_calls(prefix=...)
        return decorator
    return decorator(func)    # 用法 @log_calls

@log_calls
def add(a, b):
    return a + b

@log_calls(prefix="DEBUG")
def sub(a, b):
    return a - b

assert add(1, 2) == 3     # 打印 CALL add(1, 2)
assert sub(5, 3) == 2     # 打印 DEBUG sub(5, 3)
```

## 叠加顺序

```py
@a
@b
def f(): ...
```

等价于 `f = a(b(f))`：**离函数近的先装饰**，调用时**外层的包装先执行**。

```pycon
>>> def tag(name):
...     def decorator(func):
...         @functools.wraps(func)
...         def wrapper():
...             return f"<{name}>{func()}</{name}>"
...         return wrapper
...     return decorator
...
>>> @tag("b")
... @tag("i")
... def text():
...     return "hi"
...
>>> text()
'<b><i>hi</i></b>'
```

顺序有时很重要。比如 `@app.route` 和 `@login_required` 一起用时，`route` 必须在最外层，才能注册到带权限检查的版本。

## 用类实现装饰器

需要保存状态时，可以用实现了 `__call__` 的类：

```python
import functools

class CountCalls:
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.func(*args, **kwargs)

@CountCalls
def ping():
    return "pong"

ping(); ping()
assert ping.calls == 2
assert ping.__name__ == "ping"
```

!!! warning "类装饰器用在方法上"
    上面的 `CountCalls` 直接装饰**类里的方法**时，`self` 不会被自动传入，因为类实例不是描述符。要支持方法，需要实现 `__get__`（见[元编程](../types/metaprogramming.md)），或者干脆用函数闭包实现，再把状态挂在 `wrapper` 的属性上：`wrapper.calls = 0`。

## 装饰类

装饰器也可以接收一个类、返回一个类。标准库的 `@dataclass`、`@functools.total_ordering` 都是这样的。

一个常见用法是**注册表**：把被装饰的函数或类登记到一个字典里，后面按名字查找。Web 框架的路由、命令行子命令、插件系统都是这个模式：

```python
COMMANDS = {}

def command(name):
    def decorator(func):
        COMMANDS[name] = func
        return func            # 原样返回，不包装
    return decorator

@command("greet")
def greet(who):
    return f"hello {who}"

@command("shout")
def shout(who):
    return f"HELLO {who.upper()}!"

def dispatch(line):
    name, arg = line.split(maxsplit=1)
    return COMMANDS[name](arg)

assert dispatch("greet world") == "hello world"
assert dispatch("shout py") == "HELLO PY!"
```

## 给装饰器加类型标注

如果你的项目用类型检查，用 `ParamSpec` 让装饰器保留原函数的参数类型。3.12 起可以直接写在函数的类型参数里：

```python
import functools
from collections.abc import Callable

def logged[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        print(f"calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@logged
def area(w: float, h: float) -> float:
    return w * h

assert area(2, 3) == 6
```

类型检查器会知道 `area` 仍然是 `(w: float, h: float) -> float`。详见[类型标注](../types/typing.md)。

## 标准库里的装饰器

| 装饰器 | 作用 |
| --- | --- |
| `@property` `@classmethod` `@staticmethod` | 定义特殊方法和属性，见[面向对象](oop.md) |
| `@functools.cache` / `@lru_cache` | 记忆化 |
| `@functools.wraps` | 写装饰器用 |
| `@functools.singledispatch` | 按类型分派 |
| `@functools.total_ordering` | 补全比较方法 |
| `@dataclasses.dataclass` | 生成数据类 |
| `@contextlib.contextmanager` | 用生成器写上下文管理器 |
| `@typing.override` <span class="since">3.12+</span> | 标记覆盖父类方法 |
| `@warnings.deprecated` <span class="since">3.13+</span> | 标记弃用，调用时发出警告 |

## 练习

**1. `@debug` 装饰器。** 调用时打印参数和返回值，形如 `add(1, b=2) -> 3`。要求保留函数元信息。

??? success "参考答案"
    ```python
    import functools

    def debug(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            parts = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
            result = func(*args, **kwargs)
            print(f"{func.__name__}({', '.join(parts)}) -> {result!r}")
            return result
        return wrapper

    @debug
    def add(a, b):
        return a + b

    assert add(1, b=2) == 3        # 打印 add(1, b=2) -> 3
    assert add.__name__ == "add"
    ```

**2. `@once` 装饰器。** 被装饰的函数只执行第一次，之后的调用直接返回第一次的结果（常用于初始化）。

??? success "参考答案"
    ```python
    import functools

    def once(func):
        sentinel = object()
        result = sentinel

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal result
            if result is sentinel:
                result = func(*args, **kwargs)
            return result
        return wrapper

    calls = []

    @once
    def init():
        calls.append(1)
        return "ready"

    assert init() == "ready" and init() == "ready"
    assert len(calls) == 1
    ```

    用哨兵而不是 `None` 判断，是因为函数本身可能返回 `None`。多线程下还需要加锁，见[线程与进程](../concurrency/threads-processes.md)。

**3. 指数退避重试。** 改进正文的 `retry`：第 n 次重试前等待 `base_delay * 2**(n-1)` 秒，并且最长不超过 `max_delay`。把"睡眠"函数做成可注入的参数，方便测试时不真的等待。

??? success "参考答案"
    ```python
    import functools
    import time

    def retry(times=3, *, exceptions=(Exception,), base_delay=0.1, max_delay=2.0, sleep=time.sleep):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                for attempt in range(1, times + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions:
                        if attempt == times:
                            raise
                        sleep(min(base_delay * 2 ** (attempt - 1), max_delay))
            return wrapper
        return decorator

    waits = []

    @retry(times=5, base_delay=0.5, max_delay=1.5, sleep=waits.append)
    def always_fails():
        raise TimeoutError

    try:
        always_fails()
    except TimeoutError:
        pass
    assert waits == [0.5, 1.0, 1.5, 1.5]
    ```

    把 `sleep` 作为参数注入，测试时传一个只记录参数的函数，测试瞬间就能跑完，还能断言退避的时间序列。这是一种很实用的"依赖注入"思路。

## 小结

- [x] `@deco` 就是 `f = deco(f)`，在定义时执行一次。
- [x] 包装函数用 `*args, **kwargs` 转发参数，记得 `return` 结果，永远加 `@functools.wraps`。
- [x] 带参数的装饰器是"返回装饰器的函数"，三层结构。
- [x] 叠加时离函数近的先装饰，外层的包装先执行。
- [x] 注册表模式不需要包装，原样返回函数即可。
