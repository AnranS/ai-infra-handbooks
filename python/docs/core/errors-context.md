# 异常与上下文管理器

<p class="lead">写出能跑的代码不难，难的是在出错时也表现正确：错误信息有用、资源一定被释放、不该吞的异常不被吞掉。这一章讲异常处理的正确姿势，以及用 <code>with</code> 语句管理资源。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `try/except/else/finally` 四个分支各在什么时候执行？
    2. `raise NewError(...) from e` 和直接 `raise NewError(...)` 有什么区别？
    3. 为什么不要写裸的 `except:`？
    4. `__exit__` 返回 `True` 意味着什么？
    5. `contextlib.contextmanager` 装饰的生成器里，`yield` 前后的代码分别对应什么？

??? success "自测参考答案（先自己答，再展开对照）"
    1. `try` 先执行；出现匹配的异常时执行对应的 `except`；`try` 没有抛出异常时执行 `else`；不管怎样，最后都执行 `finally`（包括 `return`、`break` 的时候）。
    2. `from e` 把原异常记录为新异常的直接原因（`__cause__`），回溯里显示"上面的异常是下面这个异常的直接原因"；直接 `raise` 时原异常只作为隐式的上下文（`__context__`），读起来像是处理异常时又出了一个错。
    3. 它会捕获所有异常，包括 `KeyboardInterrupt`、`SystemExit`，程序无法用 Ctrl+C 中断，还会把真正的 bug 静默吞掉。至少写 `except Exception`，最好只捕获具体的类型。
    4. 异常已被处理，会被吞掉，`with` 语句之后的代码继续正常执行。
    5. `yield` 之前相当于 `__enter__`（获取资源），`yield` 出去的值就是 `as` 得到的值；`yield` 之后相当于 `__exit__`（释放资源），为了在异常时也能清理，要把它写在 `try/finally` 的 `finally` 里。

## 异常的层级

所有异常都继承自 `BaseException`。日常只应该捕获 `Exception` 的子类：

```text
BaseException
├── SystemExit            sys.exit() 触发
├── KeyboardInterrupt     Ctrl+C 触发
├── GeneratorExit
├── BaseExceptionGroup
└── Exception             ← 业务代码只和这一支打交道
    ├── ArithmeticError → ZeroDivisionError, OverflowError
    ├── LookupError     → KeyError, IndexError
    ├── OSError         → FileNotFoundError, PermissionError, TimeoutError, ConnectionError ...
    ├── ValueError      → UnicodeDecodeError ...
    ├── TypeError, AttributeError, NameError, RuntimeError ...
    └── ExceptionGroup
```

裸的 `except:` 和 `except BaseException:` 会连 Ctrl+C 和 `sys.exit()` 一起吞掉，让程序无法正常退出。**永远写出你要捕获的具体异常类型。**

## 完整的 `try` 语句

```python
def read_port(text):
    try:
        port = int(text)                 # 只把可能出错的那一行放进 try
    except ValueError:
        print(f"not a number: {text!r}")
        return None
    else:
        print("parsed ok")               # 没有异常时执行
        return port
    finally:
        print("done")                    # 无论如何都执行，用于清理

assert read_port("8080") == 8080         # parsed ok / done
assert read_port("abc") is None          # not a number / done
```

- `try` 块要尽量**小**：只包住可能抛出你要处理的异常的代码，否则可能误捕获别处的同类异常。
- `else` 放"只在成功时才做"的事，它抛出的异常不会被上面的 `except` 捕获。
- `finally` 里不要写 `return`、`break`、`continue`：它们会悄悄吞掉正在传播的异常。3.14 起这种写法会产生 `SyntaxWarning`。

一次捕获多种异常：

```py
except (KeyError, IndexError) as e:     # 需要 as 时必须加括号
    ...
except KeyError, IndexError:            # 3.14+ 不带 as 时可以省略括号
    ...
```

## 抛出异常

### 选对异常类型

| 情况 | 抛出 |
| --- | --- |
| 参数的类型不对 | `TypeError` |
| 类型对但值不合法 | `ValueError` |
| 键/索引不存在 | `KeyError` / `IndexError`（或它们的子类） |
| 对象当前状态不允许这个操作 | `RuntimeError` 或自定义异常 |
| 子类必须实现的方法 | `NotImplementedError` |

错误信息要**包含上下文**：`ValueError("port must be 1-65535, got 70000")` 远比 `ValueError("invalid")` 有用。

### 异常链：`raise ... from ...`

在 `except` 块里抛出新异常时，Python 会自动把原异常记在 `__context__` 上，打印成 "During handling of the above exception, another exception occurred"——看起来像是**处理过程中又出了 bug**。

如果是**有意地转换**异常类型，用 `from` 明确因果关系：

```pycon
>>> class ConfigError(Exception):
...     pass
...
>>> def load_port(cfg):
...     try:
...         return int(cfg["port"])
...     except (KeyError, ValueError) as e:
...         raise ConfigError(f"bad port in config: {cfg!r}") from e
...
>>> try:
...     load_port({"port": "http"})
... except ConfigError as e:
...     print(e, "| caused by:", repr(e.__cause__))
...
bad port in config: {'port': 'http'} | caused by: ValueError("invalid literal for int() with base 10: 'http'")
```

打印回溯时会显示 "The above exception was the direct cause of the following exception"。用 `raise NewError(...) from None` 可以隐藏原异常，只在原异常对调用方毫无意义时使用。

### 给异常补充信息：`add_note`

不改变异常类型，只追加上下文信息：

```pycon
>>> def parse_all(lines):
...     for lineno, line in enumerate(lines, 1):
...         try:
...             int(line)
...         except ValueError as e:
...             e.add_note(f"at line {lineno}")
...             raise
...
>>> try:
...     parse_all(["1", "2", "x"])
... except ValueError as e:
...     print(e.__notes__)
...
['at line 3']
```

注释会显示在回溯信息的末尾。裸的 `raise` 会重新抛出当前异常，并保留原来的回溯。

## 自定义异常

一个库或模块应该定义自己的异常层级：一个基类，加上若干具体的子类。调用方可以按需粗粒度或细粒度地捕获。

```python
class PaymentError(Exception):
    """所有支付相关错误的基类。"""

class CardDeclined(PaymentError):
    def __init__(self, card_last4, reason):
        super().__init__(f"card ****{card_last4} declined: {reason}")
        self.card_last4 = card_last4
        self.reason = reason

class InsufficientFunds(CardDeclined):
    pass

try:
    raise InsufficientFunds("4242", "balance too low")
except PaymentError as e:              # 调用方只关心"支付失败"
    assert isinstance(e, CardDeclined)
    assert e.reason == "balance too low"
    assert str(e) == "card ****4242 declined: balance too low"
```

记得调用 `super().__init__(message)`，否则 `str(e)` 和回溯信息里看不到消息。

## EAFP 与 LBYL

- **LBYL**（Look Before You Leap，先检查再做）：`if key in d: value = d[key]`
- **EAFP**（Easier to Ask Forgiveness than Permission，先做了再说）：`try: value = d[key] except KeyError: ...`

Python 社区更偏好 EAFP：

```py
# LBYL：检查和使用之间文件可能被删掉（竞态条件），还多了一次系统调用
if os.path.exists(path):
    with open(path) as f:
        ...

# EAFP：没有竞态，意图更清楚
try:
    with open(path) as f:
        ...
except FileNotFoundError:
    ...
```

但不要教条：如果"失败"是常见情况，或者检查的代价很低、很直观（`if items:`），LBYL 更合适。

## 同时处理多个异常：`ExceptionGroup`

并发任务可能**同时**失败好几个。3.11 起，`ExceptionGroup` 可以把多个异常打包在一起抛出，`except*` 按类型分别处理其中的一部分：

```pycon
>>> def run_all():
...     raise ExceptionGroup("batch failed", [
...         ValueError("bad input"),
...         TimeoutError("slow service"),
...         ValueError("another bad input"),
...     ])
...
>>> try:
...     run_all()
... except* ValueError as eg:
...     print("value errors:", [str(e) for e in eg.exceptions])
... except* TimeoutError as eg:
...     print("timeouts:", len(eg.exceptions))
...
value errors: ['bad input', 'another bad input']
timeouts: 1
```

和普通 `except` 不同，**多个 `except*` 分支可以都执行**。你最常在 `asyncio.TaskGroup` 里遇到它，见 [asyncio](../concurrency/asyncio.md)。

## 记录异常

```py
import logging
logger = logging.getLogger(__name__)

try:
    process()
except Exception:
    logger.exception("processing failed")   # 自动附带完整的回溯信息
    raise                                   # 记录之后通常还要继续抛出
```

**要么处理，要么往上抛，不要"记录一下然后假装没事"。** 吞掉异常会让问题在更远的地方以更奇怪的方式爆发。

## 上下文管理器

`with` 语句保证"进入时做准备，离开时做清理"，**无论是正常离开还是因为异常离开**。文件、锁、数据库事务、临时目录都应该用它。

```py
with open("data.txt", encoding="utf-8") as f:
    data = f.read()
# 离开 with 块时文件一定已经关闭，即使 read() 抛了异常
```

多个资源可以写在一个 `with` 里，3.10 起可以加括号换行：

```py
with (
    open("in.txt", encoding="utf-8") as src,
    open("out.txt", "w", encoding="utf-8") as dst,
):
    dst.write(src.read())
```

### 用类实现

实现 `__enter__` 和 `__exit__` 两个方法：

```python
import time

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self                          # as 后面的变量拿到的是这个返回值

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start
        return False                         # 不吞异常

with Timer() as t:
    sum(range(100_000))
assert t.elapsed > 0
```

`__exit__` 收到的三个参数描述了块内发生的异常（没有异常时都是 `None`）。**返回真值表示"异常已处理，不要再往外抛"**，所以除非你确实要吞掉异常，否则返回 `False` 或 `None`。

### 用生成器实现：`@contextmanager`

大多数时候，用 `contextlib.contextmanager` 写一个生成器更简单：`yield` 之前是 `__enter__`，`yield` 的值给 `as`，`yield` 之后是 `__exit__`。

```python
import os
from contextlib import contextmanager

@contextmanager
def temp_env(**overrides):
    old = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:                                 # 必须用 finally，否则块内出异常时不会恢复
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

with temp_env(APP_MODE="test"):
    assert os.environ["APP_MODE"] == "test"
assert "APP_MODE" not in os.environ
```

### `contextlib` 里的其他工具

```pycon
>>> from contextlib import suppress, redirect_stdout, ExitStack, nullcontext
>>> import io, os
>>> with suppress(FileNotFoundError):         # 明确地忽略某种异常
...     os.remove("/tmp/definitely-not-exists-12345")
...
>>> buf = io.StringIO()
>>> with redirect_stdout(buf):                # 临时把 print 输出重定向
...     print("captured")
...
>>> buf.getvalue()
'captured\n'
```

- `ExitStack`：数量在运行时才确定的一组资源，比如打开一个列表里的所有文件：

```python
import tempfile
from contextlib import ExitStack
from pathlib import Path

with tempfile.TemporaryDirectory() as d:
    paths = [Path(d) / f"part{i}.txt" for i in range(3)]
    with ExitStack() as stack:
        files = [stack.enter_context(p.open("w", encoding="utf-8")) for p in paths]
        for i, f in enumerate(files):
            f.write(f"chunk {i}\n")
    # 离开 ExitStack 时所有文件按相反顺序关闭
    assert all(f.closed for f in files)
    assert (Path(d) / "part2.txt").read_text(encoding="utf-8") == "chunk 2\n"
```

- `nullcontext()`：条件性地使用上下文管理器，`with (lock if threaded else nullcontext()):`。
- `contextlib.chdir(path)`：临时切换工作目录。
- `contextlib.closing(obj)`：给只有 `close()` 方法的对象加上 `with` 支持。

## 练习

**1. 数据库事务。** 写一个上下文管理器 `transaction(conn)`：进入时什么也不做，正常退出时 `conn.commit()`，出现异常时 `conn.rollback()` 并让异常继续抛出。用 `sqlite3` 内存数据库验证。

??? success "参考答案"
    ```python
    import sqlite3
    from contextlib import contextmanager

    @contextmanager
    def transaction(conn):
        try:
            yield conn
        except BaseException:
            conn.rollback()
            raise
        else:
            conn.commit()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (x INTEGER)")

    with transaction(conn):
        conn.execute("INSERT INTO t VALUES (1)")

    try:
        with transaction(conn):
            conn.execute("INSERT INTO t VALUES (2)")
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert conn.execute("SELECT x FROM t").fetchall() == [(1,)]
    ```

    这里捕获 `BaseException` 是合理的：即使是 Ctrl+C 也要回滚，而且马上用 `raise` 重新抛出，不会吞掉它。其实 `sqlite3.Connection` 自己就支持 `with conn:` 实现同样的语义。

**2. 配置加载器的异常设计。** 写 `load_config(text)`，解析 `key=value` 格式的多行文本，要求 `port` 必须存在且是 1-65535 的整数。定义 `ConfigError` 基类和 `MissingKey`、`InvalidValue` 两个子类；类型转换失败时用 `raise ... from` 保留原因；错误信息里包含行号或键名。

??? success "参考答案"
    ```python
    class ConfigError(Exception):
        pass

    class MissingKey(ConfigError):
        pass

    class InvalidValue(ConfigError):
        pass

    def load_config(text):
        cfg = {}
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition("=")
            if not sep:
                raise InvalidValue(f"line {lineno}: expected key=value, got {line!r}")
            cfg[key.strip()] = value.strip()

        if "port" not in cfg:
            raise MissingKey("'port' is required")
        try:
            port = int(cfg["port"])
        except ValueError as e:
            raise InvalidValue(f"port must be an integer, got {cfg['port']!r}") from e
        if not 1 <= port <= 65535:
            raise InvalidValue(f"port must be 1-65535, got {port}")
        cfg["port"] = port
        return cfg

    assert load_config("host = db\nport = 5432")["port"] == 5432
    for bad, exc in [("host=db", MissingKey), ("port=http", InvalidValue), ("oops", InvalidValue)]:
        try:
            load_config(bad)
        except exc as e:
            assert isinstance(e, ConfigError)
        else:
            raise AssertionError(f"{bad!r} should fail")
    ```

**3. 带超时提醒的计时器。** 扩展正文的 `Timer`：接收 `warn_after` 秒数，如果块执行时间超过它，就用 `warnings.warn` 发出警告；块内的异常要正常传播。

??? success "参考答案"
    ```python
    import time
    import warnings

    class Timer:
        def __init__(self, label="block", warn_after=None):
            self.label, self.warn_after = label, warn_after

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb):
            self.elapsed = time.perf_counter() - self.start
            if self.warn_after is not None and self.elapsed > self.warn_after:
                warnings.warn(f"{self.label} took {self.elapsed:.3f}s (> {self.warn_after}s)", stacklevel=2)
            return False

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with Timer("sleep", warn_after=0.01):
            time.sleep(0.02)
    assert "sleep took" in str(caught[0].message)
    ```

## 小结

- [x] 只捕获具体的异常类型；`try` 块尽量小；`finally` 里不要 `return`。
- [x] 转换异常用 `raise ... from e`；补充上下文用 `add_note`。
- [x] 库要有自己的异常层级；自定义异常记得调用 `super().__init__`。
- [x] 要么处理异常，要么继续抛出，不要吞掉。
- [x] 资源管理一律用 `with`；自己写上下文管理器首选 `@contextmanager`，清理代码放进 `finally`。
