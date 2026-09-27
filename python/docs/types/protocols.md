# 协议与抽象基类

<p class="lead">"如果它走起来像鸭子、叫起来像鸭子，那它就是鸭子。"Python 更关心对象能做什么，而不是它是什么类型。这一章讲怎么把这种鸭子类型规范化：用抽象基类（ABC）定义需要继承的接口，用 Protocol 定义不需要继承的接口。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 实现一个只读的字典类，最少需要实现几个方法？
    2. 抽象基类为什么不能实例化？什么时候报错？
    3. `Protocol` 和 ABC 的核心区别是什么？
    4. `isinstance(x, Iterable)` 是怎么判断的？`x` 需要继承 `Iterable` 吗？
    5. `@runtime_checkable` 的 `isinstance` 检查有什么局限？

## 鸭子类型

Python 函数通常不检查参数类型，只管调用需要的方法：

```python
import io

def count_lines(f):
    return sum(1 for _ in f)          # 只要能迭代出行就行

assert count_lines(io.StringIO("a\nb\n")) == 2
assert count_lines(["x", "y", "z"]) == 3        # 列表也行
```

这让代码非常灵活：测试时可以传 `StringIO` 代替真实文件，传列表代替数据库游标。代价是"需要什么"只存在于实现里，没有明确写出来。ABC 和 Protocol 就是用来把它写出来的。

## 抽象基类（ABC）

ABC 定义一组必须实现的方法。子类没实现全部抽象方法时，**实例化就会失败**，错误在最早的时机暴露：

```pycon
>>> from abc import ABC, abstractmethod
>>> class Storage(ABC):
...     @abstractmethod
...     def get(self, key: str) -> bytes: ...
...     @abstractmethod
...     def put(self, key: str, data: bytes) -> None: ...
...     def copy(self, src: str, dst: str) -> None:      # 普通方法：基于抽象方法实现
...         self.put(dst, self.get(src))
...
>>> class MemoryStorage(Storage):
...     def __init__(self):
...         self._data = {}
...     def get(self, key):
...         return self._data[key]
...
>>> MemoryStorage()
Traceback (most recent call last):
  ...
TypeError: Can't instantiate abstract class MemoryStorage without an implementation for abstract method 'put'
```

ABC 除了约束子类，还能提供基于抽象方法的**默认实现**（上面的 `copy`），子类只需要实现核心的几个方法，就能获得一整套功能。

## `collections.abc`：站在标准库的肩膀上

标准库为容器定义了一整套 ABC。继承它们，实现少数几个抽象方法，其余方法自动获得：

| ABC | 需要实现 | 自动获得 |
| --- | --- | --- |
| `Iterable` | `__iter__` | — |
| `Sized` | `__len__` | — |
| `Container` | `__contains__` | — |
| `Sequence` | `__getitem__`、`__len__` | `__contains__`、`__iter__`、`__reversed__`、`index`、`count` |
| `Mapping` | `__getitem__`、`__iter__`、`__len__` | `__contains__`、`keys`、`items`、`values`、`get`、`__eq__` |
| `MutableMapping` | 上面三个 + `__setitem__`、`__delitem__` | 再加 `pop`、`popitem`、`clear`、`update`、`setdefault` |
| `Set` | `__contains__`、`__iter__`、`__len__` | 比较运算、`&`、`|`、`-`、`^`、`isdisjoint` |

例如，一个不区分大小写的只读字典，只需要写三个方法：

```python
from collections.abc import Mapping

class CaseInsensitiveDict(Mapping):
    def __init__(self, data=()):
        self._data = {k.lower(): (k, v) for k, v in dict(data).items()}

    def __getitem__(self, key):
        return self._data[key.lower()][1]

    def __iter__(self):
        return (original for original, _ in self._data.values())

    def __len__(self):
        return len(self._data)

headers = CaseInsensitiveDict({"Content-Type": "text/html", "X-Id": "42"})
assert headers["content-type"] == "text/html"
assert "CONTENT-TYPE" in headers                   # __contains__ 自动获得
assert headers.get("missing", "n/a") == "n/a"      # get 自动获得
assert dict(headers.items()) == {"Content-Type": "text/html", "X-Id": "42"}
```

!!! tip "为什么不直接继承 `dict`"
    `dict` 的 C 实现里，`get`、`update`、`__init__` 等方法**不会**调用你覆盖的 `__getitem__`/`__setitem__`，结果行为不一致。想定制字典的行为时，继承 `collections.abc.MutableMapping` 或 `collections.UserDict`。

### 用 ABC 做 `isinstance` 检查

很多 `collections.abc` 的类实现了 `__subclasshook__`：**只要类有对应的方法，就被认为是它的子类**，不需要继承：

```pycon
>>> from collections.abc import Iterable, Sized, Hashable
>>> class Bag:
...     def __len__(self):
...         return 0
...
>>> isinstance(Bag(), Sized), isinstance(Bag(), Iterable)
(True, False)
>>> isinstance([], Hashable), isinstance((), Hashable)
(False, True)
```

所以检查"能不能迭代"时，写 `isinstance(x, Iterable)` 比 `isinstance(x, (list, tuple))` 更通用。（最可靠的方法其实是直接调用 `iter(x)` 看是否报 `TypeError`，因为只有 `__getitem__` 的对象也能迭代。）

## Protocol：静态的鸭子类型

ABC 需要**继承**。但很多时候你无法或不想修改那些类：它们来自第三方库，或者你只是想描述"有个 `read` 方法的东西"。`typing.Protocol` 定义的是**结构**：只要一个类具有协议里声明的方法和属性，它就满足协议，不需要任何继承关系。

```python
from typing import Protocol

class Renderer(Protocol):
    def render(self, text: str) -> str: ...

class HtmlRenderer:                     # 注意：没有继承 Renderer
    def render(self, text: str) -> str:
        return f"<p>{text}</p>"

class ShoutRenderer:
    def render(self, text: str) -> str:
        return text.upper() + "!"

def publish(r: Renderer, text: str) -> str:
    return r.render(text)

assert publish(HtmlRenderer(), "hi") == "<p>hi</p>"
assert publish(ShoutRenderer(), "hi") == "HI!"
```

类型检查器会检查传进去的对象是否真的有 `render(str) -> str` 方法。协议可以是泛型的，也可以包含属性：

```python
from dataclasses import dataclass
from typing import Protocol

class HasId(Protocol):
    @property
    def id(self) -> int: ...

@dataclass
class User:
    id: int
    name: str

@dataclass
class Order:
    id: int
    amount: float

def index_by_id[T: HasId](items: list[T]) -> dict[int, T]:
    return {item.id: item for item in items}

users = index_by_id([User(1, "amy"), User(2, "bob")])
assert users[2].name == "bob"          # 检查器知道 users[2] 是 User
assert index_by_id([Order(9, 1.5)])[9].amount == 1.5
```

### 运行时检查：`@runtime_checkable`

默认情况下 Protocol 只给类型检查器用。加上 `@runtime_checkable` 后，可以用 `isinstance` 检查，但**只检查方法名是否存在**，不检查签名和返回类型：

```pycon
>>> from typing import Protocol, runtime_checkable
>>> @runtime_checkable
... class Closeable(Protocol):
...     def close(self) -> None: ...
...
>>> import io
>>> isinstance(io.StringIO(), Closeable), isinstance("text", Closeable)
(True, False)
```

## ABC 还是 Protocol？

| | ABC | Protocol |
| --- | --- | --- |
| 实现方需要继承吗 | 需要（或者手动 `register`） | 不需要 |
| 能约束第三方的类吗 | 不能 | 能 |
| 能提供默认实现吗 | 能，这是它的强项 | 技术上可以，但一般不这么用 |
| 错误何时暴露 | 实例化时（运行时） | 类型检查时（静态） |
| 适合 | 框架提供的基类，希望子类复用大量逻辑 | 描述函数参数"需要什么能力"，解耦模块 |

实践中的一个好习惯：**在使用方定义 Protocol**。比如 `report.py` 需要一个"能按日期查询订单"的东西，就在 `report.py` 里定义 `OrderSource(Protocol)`；真正的数据库实现、测试用的内存实现都不需要知道这个协议的存在。这就是依赖倒置原则在 Python 里最轻量的写法。

## 练习

**1. 有界栈。** 继承 `collections.abc.Sequence` 实现 `BoundedStack(capacity)`：支持 `push(x)`（满了抛 `OverflowError`）、`pop()`；作为 `Sequence`，`s[0]` 是栈底。验证你**自动获得**了 `in`、`index`、`count`、`reversed`。

??? success "参考答案"
    ```python
    from collections.abc import Sequence

    class BoundedStack(Sequence):
        def __init__(self, capacity):
            self.capacity = capacity
            self._items = []

        def push(self, x):
            if len(self._items) >= self.capacity:
                raise OverflowError(f"stack is full (capacity={self.capacity})")
            self._items.append(x)

        def pop(self):
            return self._items.pop()

        def __getitem__(self, i):
            return self._items[i]

        def __len__(self):
            return len(self._items)

    s = BoundedStack(3)
    for x in "aba":
        s.push(x)
    assert "b" in s and s.index("b") == 1 and s.count("a") == 2
    assert list(reversed(s)) == ["a", "b", "a"]
    try:
        s.push("c")
    except OverflowError:
        pass
    else:
        raise AssertionError("should overflow")
    ```

**2. 用 Protocol 解耦。** 写一个 `send_report(notifier, lines)` 函数：把多行文本拼起来，通过 `notifier.send(subject, body)` 发送。定义 `Notifier` 协议，并实现 `EmailNotifier`（只打印）和测试用的 `FakeNotifier`（把消息存进列表），两者都不继承 `Notifier`。

??? success "参考答案"
    ```python
    from typing import Protocol

    class Notifier(Protocol):
        def send(self, subject: str, body: str) -> None: ...

    def send_report(notifier: Notifier, lines: list[str]) -> None:
        notifier.send(f"Report ({len(lines)} items)", "\n".join(lines))

    class EmailNotifier:
        def __init__(self, to: str) -> None:
            self.to = to

        def send(self, subject: str, body: str) -> None:
            print(f"To: {self.to}\nSubject: {subject}\n\n{body}")

    class FakeNotifier:
        def __init__(self) -> None:
            self.sent: list[tuple[str, str]] = []

        def send(self, subject: str, body: str) -> None:
            self.sent.append((subject, body))

    fake = FakeNotifier()
    send_report(fake, ["a ok", "b failed"])
    assert fake.sent == [("Report (2 items)", "a ok\nb failed")]
    ```

    `send_report` 只依赖协议，测试时传假实现，不需要任何 mock 库。

## 小结

- [x] 鸭子类型关注"能做什么"；ABC 和 Protocol 把"需要什么"写出来。
- [x] 继承 `collections.abc` 的 ABC，实现少数方法就能获得完整的容器接口。
- [x] 定制字典行为时继承 `MutableMapping` 或 `UserDict`，不要直接继承 `dict`。
- [x] Protocol 是结构化类型，不需要继承，适合描述参数需要的能力。
- [x] 在使用方定义 Protocol，是解耦模块、方便测试的轻量做法。
