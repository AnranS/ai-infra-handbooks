# 类型标注

<p class="lead">类型标注不会让 Python 变成静态语言，运行时也不检查。但配合类型检查器和编辑器，它能在运行前抓到一大类 bug，让代码自带文档，重构时更有底气。这一章讲实用的类型标注：从基础写法到泛型、Protocol 和各种特殊类型。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 参数应该标注成 `list[str]` 还是 `Iterable[str]`？返回值呢？
    2. `Any` 和 `object` 有什么区别？
    3. 3.12 的 `def first[T](xs: list[T]) -> T` 是什么意思？
    4. `Optional[int]` 等价于什么？
    5. 怎么给一个装饰器标注类型，让被装饰函数的参数类型不丢失？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 参数用尽量抽象、只要求需要的能力的类型：只遍历就用 `Iterable[str]`，调用方可以传列表、元组、生成器；返回值用具体的类型 `list[str]`，让调用方知道能拿它做什么。
    2. `Any` 关闭类型检查：对它做什么都不报错；`object` 是所有类型的基类，能接收任何值，但只能做所有对象都支持的操作。表示"任何值"时应该用 `object`。
    3. PEP 695 的泛型语法：声明一个类型参数 `T`，函数接收元素类型为 `T` 的列表、返回一个 `T`，把输入和输出的类型关联起来。
    4. `int | None`（`Union[int, None]`）。
    5. 用 `ParamSpec` 和 `TypeVar`：`def deco[**P, R](f: Callable[P, R]) -> Callable[P, R]`，包装函数写成 `def wrapper(*args: P.args, **kwargs: P.kwargs) -> R`，签名就保留下来了。

## 为什么要写类型

```py
def total_price(items, discount):
    ...
```

`items` 是什么？列表？字典？里面装的是什么？`discount` 是 `0.1` 还是 `10`？加上标注，答案一目了然：

```py
def total_price(items: list[LineItem], discount: float = 0.0) -> Decimal:
    ...
```

类型检查器（mypy、pyright）会在你运行代码**之前**发现：传错了参数类型、对可能为 `None` 的值调用了方法、拼错了属性名、函数的某个分支忘了返回值……编辑器的自动补全和跳转也会准确得多。

!!! note "渐进式类型"
    Python 的类型系统是**渐进式**的：可以只给一部分代码加标注，没有标注的部分被当作 `Any`，不做检查。老项目可以从新代码、核心模块、公共接口开始逐步加。

## 基础写法

```python
from collections.abc import Callable, Iterable, Mapping, Sequence

count: int = 0
names: list[str] = []
scores: dict[str, float] = {}
point: tuple[int, int] = (0, 0)
row: tuple[str, ...] = ("a", "b", "c")         # 任意长度、同类型的元组
maybe_port: int | None = None                  # 3.10+ 的联合类型写法

def greet(name: str, times: int = 1) -> str:
    return ", ".join([f"hi {name}"] * times)

def log(msg: str) -> None:                     # 没有返回值
    print(msg)

def apply(f: Callable[[int, int], int], a: int, b: int) -> int:
    return f(a, b)

assert apply(lambda x, y: x * y, 3, 4) == 12
```

几个要点：

- 3.9 起直接用内置类型 `list[int]`、`dict[str, int]`，不需要 `typing.List`。
- `X | None` 等价于老写法 `Optional[X]`，表示"可能是 `None`"，不是"可选参数"。
- 抽象类型（`Iterable`、`Sequence`、`Mapping`、`Callable`）从 `collections.abc` 导入。

### 参数要宽，返回值要窄

参数用**能满足需要的最抽象的类型**，这样调用方传列表、元组、生成器都行；返回值用**具体类型**，调用方能用的操作最多。

```python
from collections.abc import Iterable, Mapping

def normalize(words: Iterable[str]) -> list[str]:     # 接受任何可迭代对象
    return sorted({w.strip().lower() for w in words})

def invert(d: Mapping[str, int]) -> dict[int, str]:  # Mapping 表示"只读，不会修改它"
    return {v: k for k, v in d.items()}

assert normalize(("B ", "a", "b")) == ["a", "b"]
assert normalize(w for w in ["x"]) == ["x"]
assert invert({"a": 1}) == {1: "a"}
```

| 你的函数需要 | 参数标注 |
| --- | --- |
| 只遍历一次 | `Iterable[T]` |
| 遍历多次、取长度、按索引访问 | `Sequence[T]` |
| 只读取键值 | `Mapping[K, V]` |
| 要修改它 | `list[T]`、`dict[K, V]`（或 `MutableSequence`、`MutableMapping`） |

### `Any` 与 `object`

- `Any`：**关闭检查**。任何值都能赋给它，它也能赋给任何类型，能调用任何方法。是类型系统的"逃生舱"。
- `object`：**所有类型的父类**。任何值都能赋给它，但你只能对它做所有对象都支持的操作（比如 `str()`），想用别的方法必须先用 `isinstance` 收窄。

当你想表达"接受任何东西"时，优先用 `object`，它更安全。

## 类型收窄

类型检查器能理解 `isinstance`、`is None`、`in` 等判断，在分支里自动**收窄**类型：

```python
def describe(value: int | str | None) -> str:
    if value is None:
        return "nothing"                  # 这里 value 是 None
    if isinstance(value, int):
        return f"number {value + 1}"      # 这里 value 是 int
    return f"text {value.upper()}"        # 这里只剩 str

assert describe(41) == "number 42" and describe("a") == "text A"
```

自己写的判断函数也可以告诉检查器它能收窄类型，用 `TypeIs` <span class="since">3.13+</span>：

```python
from collections.abc import Sequence
from typing import TypeIs

def is_str_seq(xs: Sequence[object]) -> TypeIs[Sequence[str]]:
    return all(isinstance(x, str) for x in xs)

def join_all(xs: Sequence[object]) -> str:
    if is_str_seq(xs):
        return ",".join(xs)               # 检查器知道这里 xs 是 Sequence[str]
    return ""

assert join_all(["a", "b"]) == "a,b"
```

这里特意用了 `Sequence` 而不是 `list`。`TypeIs` 要求收窄后的类型是输入类型的子类型，而 `list[str]` **不是** `list[object]` 的子类型：如果是的话，你就能把一个 `list[str]` 当作 `list[object]` 传出去，再往里塞一个整数。这叫**不变性**（invariance），可变容器都是不变的；只读的 `Sequence` 是**协变**的，`Sequence[str]` 是 `Sequence[object]` 的子类型。

## 类型别名

```python
type Vector = list[float]                        # 3.12+ 的 type 语句
type JSON = dict[str, JSON] | list[JSON] | str | int | float | bool | None   # 可以递归

def scale(v: Vector, k: float) -> Vector:
    return [x * k for x in v]

assert scale([1.0, 2.0], 2) == [2.0, 4.0]
```

`type` 语句的右侧是**惰性求值**的，所以可以引用还没定义的名字，也可以递归。3.12 之前的写法是 `Vector: TypeAlias = list[float]`。

## 泛型

当函数对"某种类型"通用、并且输入和输出的类型有关联时，用**类型参数**表达这种关联：

```python
from collections.abc import Sequence

def first[T](xs: Sequence[T]) -> T:              # 3.12+ 语法
    return xs[0]

n = first([1, 2, 3])        # 检查器推断 n: int
s = first("abc")            # 检查器推断 s: str
assert (n, s) == (1, "a")
```

如果标注成 `def first(xs: Sequence[object]) -> object`，调用方拿到的就是 `object`，还得自己转换，信息丢失了。

类型参数可以加**约束**：

```python
from collections.abc import Hashable, Iterable

def dedupe[T: Hashable](xs: Iterable[T]) -> list[T]:        # 上界：T 必须可哈希
    return list(dict.fromkeys(xs))

def biggest[N: (int, float)](a: N, b: N) -> N:             # 限定为 int 或 float 之一
    return a if a > b else b

assert dedupe([3, 1, 3]) == [3, 1]
assert biggest(2, 5) == 5
```

泛型类：

```python
class Stack[T]:
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()

    def __len__(self) -> int:
        return len(self._items)

s = Stack[int]()
s.push(1)
assert s.pop() == 1 and len(s) == 0
```

3.12 之前的写法需要先定义 `T = TypeVar("T")`，类继承 `Generic[T]`。读老代码时会经常看到。

## 常用的特殊类型

```python
from typing import ClassVar, Final, Literal, Self, override

Mode = Literal["r", "w", "a"]                  # 只能是这几个字面值之一

def open_file(path: str, mode: Mode = "r") -> str:
    return f"{path}:{mode}"

MAX_RETRIES: Final = 3                         # 常量：检查器禁止重新赋值

class Node:
    instances: ClassVar[int] = 0               # 类变量，不是实例字段

    def __init__(self) -> None:
        self.children: list[Self] = []

    def add(self, child: Self) -> Self:        # Self：返回"当前类"，子类调用时是子类
        self.children.append(child)
        return self

class TreeNode(Node):
    @override                                  # 3.12+：声明这是覆盖，父类没有这个方法时报错
    def add(self, child: Self) -> Self:
        return super().add(child)

root = TreeNode().add(TreeNode())
assert isinstance(root, TreeNode) and open_file("a.txt", "w") == "a.txt:w"
```

### `@overload`：返回类型取决于参数

```python
from typing import Literal, overload

@overload
def fetch(raw: Literal[True]) -> bytes: ...
@overload
def fetch(raw: Literal[False] = ...) -> str: ...
def fetch(raw: bool = False) -> str | bytes:   # 真正的实现
    data = b"hello"
    return data if raw else data.decode()

assert fetch() == "hello" and fetch(raw=True) == b"hello"
```

`@overload` 的几个签名只给检查器看，最后那个不带 `@overload` 的才是真正运行的实现。

## 给回调和装饰器标注：`ParamSpec`

`Callable[..., R]` 会丢失参数信息。要让装饰器"透传"原函数的签名，用 `ParamSpec`（3.12+ 写作 `**P`）：

```python
import functools
import time
from collections.abc import Callable

def timed[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            print(f"{func.__name__}: {time.perf_counter() - start:.4f}s")
    return wrapper

@timed
def resize(path: str, *, width: int) -> tuple[int, int]:
    return (width, width // 2)

assert resize("a.png", width=100) == (100, 50)
# 检查器仍然知道 resize 的签名：resize("a.png", width="100") 会被报错
```

## 结构化类型：Protocol

`Protocol` 描述"**具有某些方法的任何对象**"，不要求继承。这是给鸭子类型加上静态检查的方式，详见[协议与抽象基类](protocols.md)。

```python
from typing import Protocol

class SupportsClose(Protocol):
    def close(self) -> None: ...

class Resource:                    # 没有继承 SupportsClose
    closed = False
    def close(self) -> None:
        self.closed = True

def close_all(items: list[SupportsClose]) -> None:
    for item in items:
        item.close()

r = Resource()
close_all([r])                     # 检查器认可：Resource 结构上满足协议
assert r.closed
```

## 标注的求值时机

3.14 起，函数和类的标注改为**延迟求值**（PEP 649）：定义时不计算，只有在真正读取 `__annotations__` 时才计算。直接的好处是可以引用还没定义的类，不再需要写成字符串：

```python
class Tree:
    def add(self, child: Tree) -> Tree:        # 3.14 之前要写成 "Tree"
        return child

assert Tree().add(Tree()) is not None
```

在 3.13 及更早的版本，要么把类型写成字符串 `"Tree"`，要么在文件开头写 `from __future__ import annotations`。如果你的代码需要兼容老版本，保留这些写法仍然没问题。

运行时读取标注用 `typing.get_type_hints(obj)` 或 3.14 新增的 `annotationlib.get_annotations(obj)`，不要直接读 `__annotations__`。

## 类型检查器

| 工具 | 特点 |
| --- | --- |
| [mypy](https://mypy-lang.org/) | 最早、最成熟的检查器，插件生态丰富（Django、SQLAlchemy 等） |
| [pyright](https://github.com/microsoft/pyright) | 微软出品，速度快，VS Code 的 Pylance 就是基于它；检查规则更严格 |
| [ty](https://github.com/astral-sh/ty) | Astral（uv、ruff 的作者）出品的新检查器，主打速度，还在快速演进中 |

在项目里启用 mypy 的最小配置（写在 `pyproject.toml`）：

```toml
[tool.mypy]
python_version = "3.12"
strict = true                  # 新项目直接开严格模式
warn_unreachable = true

[[tool.mypy.overrides]]
module = ["some_untyped_lib.*"]
ignore_missing_imports = true
```

然后运行 `uv run mypy src/`。老项目可以先不开 `strict`，逐步收紧。

!!! tip "什么时候可以不写标注"
    - 局部变量：检查器能推断，`count = 0` 不需要写 `count: int = 0`。
    - 很短的内部脚本。
    - 需要写标注的地方：**函数签名**（参数和返回值）、类的属性、模块级的公共变量。这些是模块之间的"契约"。

!!! interview "怎么讲清楚"
    讲类型标注：参数用抽象类型（`Iterable[str]`、`Mapping`），返回值用具体类型；可能为空写 `X | None`；"任何值"用 `object`（类型检查器会要求先收窄），`Any` 是关掉检查；输入输出类型有关联时用泛型（3.12 起写 `def first[T](xs: list[T]) -> T`）；装饰器用 `ParamSpec` 保留被装饰函数的签名；鸭子类型用 `Protocol`。标注默认不做运行时检查，要靠 mypy / pyright 在 CI 里执行；运行时校验外部输入用 pydantic 之类的库。

## 练习

**1. 给函数加标注。** 为下面的函数加上尽可能准确的类型标注，然后用 mypy 检查：

```py
def group_by(items, key):
    groups = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    return groups
```

??? success "参考答案"
    ```python
    from collections.abc import Callable, Hashable, Iterable

    def group_by[T, K: Hashable](items: Iterable[T], key: Callable[[T], K]) -> dict[K, list[T]]:
        groups: dict[K, list[T]] = {}
        for item in items:
            groups.setdefault(key(item), []).append(item)
        return groups

    result = group_by(["apple", "avocado", "banana"], key=lambda w: w[0])
    assert result == {"a": ["apple", "avocado"], "b": ["banana"]}
    ```

    两个类型参数表达了这个函数的完整契约：元素类型 `T` 原样出现在结果里，`key` 的返回类型 `K` 成为字典的键。调用方拿到的 `result` 会被推断为 `dict[str, list[str]]`。

**2. 类型安全的结果类型。** 很多语言用 `Result` 类型代替异常。用泛型 dataclass 实现 `Ok[T]` 和 `Err[E]`，定义 `type Result[T, E] = Ok[T] | Err[E]`，写一个 `parse_int(s: str) -> Result[int, str]`，并用 `match` 处理结果。

??? success "参考答案"
    ```python
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Ok[T]:
        value: T

    @dataclass(frozen=True)
    class Err[E]:
        error: E

    type Result[T, E] = Ok[T] | Err[E]

    def parse_int(s: str) -> Result[int, str]:
        try:
            return Ok(int(s))
        except ValueError:
            return Err(f"not an integer: {s!r}")

    def double(s: str) -> str:
        match parse_int(s):
            case Ok(value):
                return str(value * 2)
            case Err(error):
                return f"error: {error}"
        raise AssertionError("unreachable")

    assert double("21") == "42"
    assert double("x") == "error: not an integer: 'x'"
    ```

    在 Python 里，异常仍然是处理错误的主流方式。`Result` 风格适合"失败是正常结果之一"的场景（比如批量校验时收集所有错误），了解一下它能加深对泛型和模式匹配的理解。

## 小结

- [x] 函数签名、类属性、公共变量要写标注；局部变量交给推断。
- [x] 参数用抽象类型（`Iterable`、`Mapping`），返回值用具体类型。
- [x] 用 `X | None` 表示可能为空；用 `object` 而不是 `Any` 表示"任何值"。
- [x] 输入输出类型有关联时用泛型：3.12+ 写 `def f[T](...)`、`class C[T]`。
- [x] 装饰器用 `ParamSpec` 保留签名；鸭子类型用 `Protocol`。
- [x] 用 mypy 或 pyright 在 CI 里跑类型检查，新项目开 `strict`。
