# Type annotations

<p class="lead">Type annotations do not make Python a static language and nothing is checked at run time. But with a type checker and an editor they catch a whole class of bugs before the code runs, make the code document itself, and make refactoring far less nerve-racking. This chapter covers annotations that earn their keep, from the basics to generics, Protocol and the special types.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Should a parameter be annotated `list[str]` or `Iterable[str]`? And the return value?
    2. What is the difference between `Any` and `object`?
    3. What does 3.12's `def first[T](xs: list[T]) -> T` mean?
    4. What is `Optional[int]` equivalent to?
    5. How do you annotate a decorator so the decorated function's parameter types are not lost?

??? success "Answers (try it yourself first, then expand)"
    1. A parameter takes the most abstract type that requires only the capability needed: `Iterable[str]` for merely iterating, so the caller may pass a list, a tuple or a generator; the return value takes the concrete `list[str]`, so the caller knows what can be done with it.
    2. `Any` turns type checking off: anything done to it passes; `object` is the base of every type and accepts any value, but only the operations every object supports can be done to it. "Any value" should be `object`.
    3. PEP 695's generic syntax: it declares a type parameter `T`, and the function takes a list whose elements are `T` and returns a `T`, which ties the input's and the output's types together.
    4. `int | None` (`Union[int, None]`).
    5. Through `ParamSpec` and `TypeVar`: `def deco[**P, R](f: Callable[P, R]) -> Callable[P, R]`, with the wrapper written as `def wrapper(*args: P.args, **kwargs: P.kwargs) -> R`, which keeps the signature.

## Why write types {#为什么要写类型}

```py
def total_price(items, discount):
    ...
```

What is `items`? A list? A dict? What is in it? Is `discount` `0.1` or `10`? With annotations the answer is plain:

```py
def total_price(items: list[LineItem], discount: float = 0.0) -> Decimal:
    ...
```

A type checker (mypy, pyright) finds, **before** you run the code: an argument of the wrong type, a method called on something that may be `None`, a misspelt attribute, a branch of a function that forgot to return. The editor's completion and navigation become far more accurate too.

!!! note "Gradual typing"
    Python's type system is **gradual**: part of the code can be annotated and the unannotated part is treated as `Any` and not checked. An old project can start with the new code, the core modules and the public interfaces and work outwards.

## The basics {#基础写法}

```python
from collections.abc import Callable, Iterable, Mapping, Sequence

count: int = 0
names: list[str] = []
scores: dict[str, float] = {}
point: tuple[int, int] = (0, 0)
row: tuple[str, ...] = ("a", "b", "c")         # a tuple of any length with one element type
maybe_port: int | None = None                  # the 3.10+ spelling of a union

def greet(name: str, times: int = 1) -> str:
    return ", ".join([f"hi {name}"] * times)

def log(msg: str) -> None:                     # no return value
    print(msg)

def apply(f: Callable[[int, int], int], a: int, b: int) -> int:
    return f(a, b)

assert apply(lambda x, y: x * y, 3, 4) == 12
```

A few points:

- from 3.9, use the built-in types `list[int]` and `dict[str, int]` directly, with no need for `typing.List`.
- `X | None` is equivalent to the older `Optional[X]` and means "may be `None`", not "an optional parameter".
- The abstract types (`Iterable`, `Sequence`, `Mapping`, `Callable`) are imported from `collections.abc`.

### Wide parameters, narrow return values {#参数要宽返回值要窄}

A parameter takes **the most abstract type that meets the need**, so a caller may pass a list, a tuple or a generator; a return value takes **a concrete type**, which gives the caller the most operations.

```python
from collections.abc import Iterable, Mapping

def normalize(words: Iterable[str]) -> list[str]:     # takes any iterable
    return sorted({w.strip().lower() for w in words})

def invert(d: Mapping[str, int]) -> dict[int, str]:  # Mapping says "read-only, it will not be modified"
    return {v: k for k, v in d.items()}

assert normalize(("B ", "a", "b")) == ["a", "b"]
assert normalize(w for w in ["x"]) == ["x"]
assert invert({"a": 1}) == {1: "a"}
```

| What your function needs | The parameter's annotation |
| --- | --- |
| to iterate once | `Iterable[T]` |
| to iterate repeatedly, take the length, index it | `Sequence[T]` |
| to read keys and values only | `Mapping[K, V]` |
| to modify it | `list[T]`, `dict[K, V]` (or `MutableSequence`, `MutableMapping`) |

### `Any` and `object` {#any-与-object}

- `Any`: **checking off**. Any value can be assigned to it, it can be assigned to any type, and any method can be called on it. It is the type system's escape hatch.
- `object`: **the parent of every type**. Any value can be assigned to it, but only the operations every object supports can be done to it (`str()`, say), and anything else requires narrowing with `isinstance` first.

To say "anything is accepted", prefer `object`, which is safer.

## Narrowing {#类型收窄}

A type checker understands `isinstance`, `is None`, `in` and similar tests and **narrows** the type inside the branch automatically:

```python
def describe(value: int | str | None) -> str:
    if value is None:
        return "nothing"                  # value is None here
    if isinstance(value, int):
        return f"number {value + 1}"      # value is an int here
    return f"text {value.upper()}"        # only str is left here

assert describe(41) == "number 42" and describe("a") == "text A"
```

A test function of your own can tell the checker it narrows a type too, through `TypeIs` <span class="since">3.13+</span>:

```python
from collections.abc import Sequence
from typing import TypeIs

def is_str_seq(xs: Sequence[object]) -> TypeIs[Sequence[str]]:
    return all(isinstance(x, str) for x in xs)

def join_all(xs: Sequence[object]) -> str:
    if is_str_seq(xs):
        return ",".join(xs)               # the checker knows xs is a Sequence[str] here
    return ""

assert join_all(["a", "b"]) == "a,b"
```

`Sequence` rather than `list` is used deliberately here. `TypeIs` requires the narrowed type to be a subtype of the input's, and `list[str]` is **not** a subtype of `list[object]`: if it were, you could pass a `list[str]` out as a `list[object]` and then put an integer into it. That is **invariance**, which every mutable container has; the read-only `Sequence` is **covariant**, so `Sequence[str]` is a subtype of `Sequence[object]`.

## Type aliases {#类型别名}

```python
type Vector = list[float]                        # the 3.12+ type statement
type JSON = dict[str, JSON] | list[JSON] | str | int | float | bool | None   # it can recurse

def scale(v: Vector, k: float) -> Vector:
    return [x * k for x in v]

assert scale([1.0, 2.0], 2) == [2.0, 4.0]
```

The right side of a `type` statement is **evaluated lazily**, so it can refer to names not yet defined and can recurse. Before 3.12 the spelling was `Vector: TypeAlias = list[float]`.

## Generics {#泛型}

When a function is general over "some type" and the input's and output's types are related, express that relation with **type parameters**:

```python
from collections.abc import Sequence

def first[T](xs: Sequence[T]) -> T:              # 3.12+ syntax
    return xs[0]

n = first([1, 2, 3])        # the checker infers n: int
s = first("abc")            # the checker infers s: str
assert (n, s) == (1, "a")
```

Annotated as `def first(xs: Sequence[object]) -> object`, the caller gets an `object` and has to convert it, and the information is lost.

A type parameter can carry **constraints**:

```python
from collections.abc import Hashable, Iterable

def dedupe[T: Hashable](xs: Iterable[T]) -> list[T]:        # an upper bound: T has to be hashable
    return list(dict.fromkeys(xs))

def biggest[N: (int, float)](a: N, b: N) -> N:             # constrained to int or float
    return a if a > b else b

assert dedupe([3, 1, 3]) == [3, 1]
assert biggest(2, 5) == 5
```

A generic class:

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

Before 3.12 this needed a `T = TypeVar("T")` first and the class inheriting `Generic[T]`. Old code shows this often.

## The common special types {#常用的特殊类型}

```python
from typing import ClassVar, Final, Literal, Self, override

Mode = Literal["r", "w", "a"]                  # only one of these literal values

def open_file(path: str, mode: Mode = "r") -> str:
    return f"{path}:{mode}"

MAX_RETRIES: Final = 3                         # a constant: the checker forbids reassignment

class Node:
    instances: ClassVar[int] = 0               # a class variable, not an instance field

    def __init__(self) -> None:
        self.children: list[Self] = []

    def add(self, child: Self) -> Self:        # Self: returns "the current class", which for a subclass's call is the subclass
        self.children.append(child)
        return self

class TreeNode(Node):
    @override                                  # 3.12+: declares an override, raising when the parent has no such method
    def add(self, child: Self) -> Self:
        return super().add(child)

root = TreeNode().add(TreeNode())
assert isinstance(root, TreeNode) and open_file("a.txt", "w") == "a.txt:w"
```

### `@overload`: the return type depends on the arguments {#overload返回类型取决于参数}

```python
from typing import Literal, overload

@overload
def fetch(raw: Literal[True]) -> bytes: ...
@overload
def fetch(raw: Literal[False] = ...) -> str: ...
def fetch(raw: bool = False) -> str | bytes:   # the actual implementation
    data = b"hello"
    return data if raw else data.decode()

assert fetch() == "hello" and fetch(raw=True) == b"hello"
```

The `@overload` signatures are for the checker only, and the last one without `@overload` is the implementation that actually runs.

## Annotating callbacks and decorators: `ParamSpec` {#给回调和装饰器标注paramspec}

`Callable[..., R]` loses the parameter information. To have a decorator "pass through" the original's signature, use `ParamSpec` (written `**P` from 3.12):

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
# the checker still knows resize's signature: resize("a.png", width="100") is reported
```

## Structural typing: Protocol {#结构化类型protocol}

A `Protocol` describes "**any object with certain methods**" without requiring inheritance. That is how duck typing gets static checking; see [protocols and abstract base classes](protocols.md).

```python
from typing import Protocol

class SupportsClose(Protocol):
    def close(self) -> None: ...

class Resource:                    # does not inherit SupportsClose
    closed = False
    def close(self) -> None:
        self.closed = True

def close_all(items: list[SupportsClose]) -> None:
    for item in items:
        item.close()

r = Resource()
close_all([r])                     # the checker accepts it: Resource satisfies the protocol structurally
assert r.closed
```

## When annotations are evaluated {#标注的求值时机}

From 3.14, functions' and classes' annotations are **evaluated lazily** (PEP 649): nothing is computed at definition and it happens only when `__annotations__` is really read. The immediate benefit is being able to refer to a class not yet defined, with no need for a string:

```python
class Tree:
    def add(self, child: Tree) -> Tree:        # before 3.14 this had to be "Tree"
        return child

assert Tree().add(Tree()) is not None
```

On 3.13 and earlier, either write the type as the string `"Tree"` or put `from __future__ import annotations` at the top of the file. Code that has to work on older versions can keep these spellings without harm.

To read annotations at run time, use `typing.get_type_hints(obj)` or 3.14's new `annotationlib.get_annotations(obj)` rather than reading `__annotations__` directly.

## The type checkers {#类型检查器}

| Tool | Notes |
| --- | --- |
| [mypy](https://mypy-lang.org/) | the earliest and most mature checker, with a rich plugin ecosystem (Django, SQLAlchemy and others) |
| [pyright](https://github.com/microsoft/pyright) | Microsoft's, fast, and the basis of VS Code's Pylance; its rules are stricter |
| [ty](https://github.com/astral-sh/ty) | a new checker from Astral (who wrote uv and ruff), aimed at speed and still evolving quickly |

The minimal configuration to enable mypy in a project (in `pyproject.toml`):

```toml
[tool.mypy]
python_version = "3.12"
strict = true                  # a new project turns strict mode straight on
warn_unreachable = true

[[tool.mypy.overrides]]
module = ["some_untyped_lib.*"]
ignore_missing_imports = true
```

Then run `uv run mypy src/`. An old project can leave `strict` off at first and tighten gradually.

!!! tip "Where annotations can be left out"
    - Local variables: the checker infers them, and `count = 0` needs no `count: int = 0`.
    - Very short internal scripts.
    - Where they are needed: **function signatures** (parameters and return values), class attributes, and module-level public variables. These are the contracts between modules.

!!! interview "How to explain it"
    On annotations: parameters take abstract types (`Iterable[str]`, `Mapping`) and return values concrete ones; "possibly absent" is `X | None`; "any value" is `object` (which the checker makes you narrow first) while `Any` turns checking off; related input and output types take generics (`def first[T](xs: list[T]) -> T` from 3.12); a decorator uses `ParamSpec` to keep the decorated function's signature; duck typing uses `Protocol`. Annotations are not checked at run time by default and depend on mypy / pyright running in CI; validating external input at run time takes a library like pydantic.

## Exercises {#练习}

**1. Annotate a function.** Give the function below the most accurate annotations you can, then check it with mypy:

```py
def group_by(items, key):
    groups = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    return groups
```

??? success "Answer"
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

    The two type parameters express the function's whole contract: the element type `T` appears unchanged in the result, and `key`'s return type `K` becomes the dict's key. The caller's `result` is inferred as `dict[str, list[str]]`.

**2. A type-safe result type.** Many languages use a `Result` type in place of exceptions. Implement `Ok[T]` and `Err[E]` as generic dataclasses, define `type Result[T, E] = Ok[T] | Err[E]`, write a `parse_int(s: str) -> Result[int, str]`, and handle the result with `match`.

??? success "Answer"
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

    In Python, exceptions are still the mainstream way to handle errors. The `Result` style suits cases where "failure is one of the normal outcomes" (collecting every error while validating in bulk), and knowing it deepens the understanding of generics and pattern matching.

## Summary {#小结}

- [x] Annotate function signatures, class attributes and public variables; leave local variables to inference.
- [x] Parameters take abstract types (`Iterable`, `Mapping`) and return values concrete ones.
- [x] Use `X | None` for "possibly absent"; use `object` rather than `Any` for "any value".
- [x] Related input and output types take generics: `def f[T](...)` and `class C[T]` from 3.12.
- [x] A decorator uses `ParamSpec` to keep the signature; duck typing uses `Protocol`.
- [x] Run mypy or pyright in CI, with `strict` on for a new project.
