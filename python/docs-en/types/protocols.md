# Protocols and abstract base classes

<p class="lead">"If it walks like a duck and quacks like a duck, it is a duck." Python cares more about what an object can do than about its type. This chapter covers making that duck typing explicit: abstract base classes (ABCs) define an interface that has to be inherited, and Protocols define one that does not.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How few methods does a read-only dict class have to implement?
    2. Why can an abstract base class not be instantiated? When is the error raised?
    3. What is the core difference between a `Protocol` and an ABC?
    4. How does `isinstance(x, Iterable)` decide? Does `x` have to inherit `Iterable`?
    5. What are the limits of `@runtime_checkable`'s `isinstance` check?

??? success "Answers (try it yourself first, then expand)"
    1. Inheriting `collections.abc.Mapping`, only three: `__getitem__`, `__len__` and `__iter__`, with `get`, `keys`, `items`, `__contains__` and the rest provided by the base class.
    2. It has abstract methods (`@abstractmethod`) that are not implemented, so it is incomplete; the `TypeError` comes at instantiation and not when the class is defined.
    3. An ABC is nominal typing: it has to be inherited (or `register`ed) for `isinstance` to hold; a `Protocol` is structural typing: having the required methods and attributes is enough, with no inheritance, and it is mainly for the static type checker.
    4. `Iterable` defines a `__subclasshook__` that only checks whether the object's type has an `__iter__` method, with no inheritance needed.
    5. It only checks that the methods and attributes exist, not their signatures, parameter types or return types; and it is slower than an ordinary `isinstance`.

## Duck typing {#鸭子类型}

![Figure: two kinds of interface - an ABC through inheritance, a Protocol through shape](../assets/figures/abc-vs-protocol.svg){.aig-svg}

A Python function usually does not check its arguments' types and simply calls the methods it needs:

```python
import io

def count_lines(f):
    return sum(1 for _ in f)          # anything that iterates lines will do

assert count_lines(io.StringIO("a\nb\n")) == 2
assert count_lines(["x", "y", "z"]) == 3        # a list works too
```

That makes the code very flexible: a test can pass a `StringIO` in place of a real file, or a list in place of a database cursor. The cost is that "what is needed" exists only in the implementation and is never written down. ABCs and Protocols are how it gets written down.

## Abstract base classes (ABCs) {#抽象基类abc}

An ABC defines a set of methods that have to be implemented. When a subclass has not implemented all of them, **instantiation fails**, so the error surfaces at the earliest opportunity:

```pycon
>>> from abc import ABC, abstractmethod
>>> class Storage(ABC):
...     @abstractmethod
...     def get(self, key: str) -> bytes: ...
...     @abstractmethod
...     def put(self, key: str, data: bytes) -> None: ...
...     def copy(self, src: str, dst: str) -> None:      # an ordinary method: built on the abstract ones
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

Besides constraining subclasses, an ABC can provide **default implementations** built on the abstract methods (the `copy` above), so that a subclass implementing a few core methods gets a whole set of behaviour.

## `collections.abc`: standing on the standard library's shoulders {#collectionsabc站在标准库的肩膀上}

The standard library defines a whole set of ABCs for containers. Inherit them, implement a few abstract methods, and the rest come free:

| ABC | To implement | Obtained automatically |
| --- | --- | --- |
| `Iterable` | `__iter__` | — |
| `Sized` | `__len__` | — |
| `Container` | `__contains__` | — |
| `Sequence` | `__getitem__`, `__len__` | `__contains__`, `__iter__`, `__reversed__`, `index`, `count` |
| `Mapping` | `__getitem__`, `__iter__`, `__len__` | `__contains__`, `keys`, `items`, `values`, `get`, `__eq__` |
| `MutableMapping` | those three + `__setitem__`, `__delitem__` | plus `pop`, `popitem`, `clear`, `update`, `setdefault` |
| `Set` | `__contains__`, `__iter__`, `__len__` | the comparisons, `&`, `|`, `-`, `^`, `isdisjoint` |

A case-insensitive read-only dict, for instance, takes three methods:

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
assert "CONTENT-TYPE" in headers                   # __contains__ comes free
assert headers.get("missing", "n/a") == "n/a"      # get comes free
assert dict(headers.items()) == {"Content-Type": "text/html", "X-Id": "42"}
```

!!! tip "Why not inherit `dict` directly"
    In `dict`'s C implementation, `get`, `update`, `__init__` and the rest **do not** call the `__getitem__`/`__setitem__` you overrode, which leaves the behaviour inconsistent. To customize a dict's behaviour, inherit `collections.abc.MutableMapping` or `collections.UserDict`.

### `isinstance` with an ABC {#用-abc-做-isinstance-检查}

Many classes in `collections.abc` implement a `__subclasshook__`: **a class with the right methods counts as a subclass**, with no inheritance:

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

So to test "can this be iterated", `isinstance(x, Iterable)` is more general than `isinstance(x, (list, tuple))`. (The most reliable test is in fact calling `iter(x)` and seeing whether it raises `TypeError`, because an object with only `__getitem__` can be iterated too.)

## Protocol: static duck typing {#protocol静态的鸭子类型}

An ABC requires **inheritance**. But often you cannot or do not want to modify those classes: they come from a third-party library, or you only want to describe "something with a `read` method". `typing.Protocol` defines **a structure**: a class with the methods and attributes the protocol declares satisfies it, with no inheritance relationship at all.

```python
from typing import Protocol

class Renderer(Protocol):
    def render(self, text: str) -> str: ...

class HtmlRenderer:                     # note: it does not inherit Renderer
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

The type checker checks that the object passed in really has a `render(str) -> str`. A protocol can be generic, and can include attributes:

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
assert users[2].name == "bob"          # the checker knows users[2] is a User
assert index_by_id([Order(9, 1.5)])[9].amount == 1.5
```

### Checking at run time: `@runtime_checkable` {#运行时检查runtime_checkable}

By default a Protocol is for the type checker only. With `@runtime_checkable` it can be used with `isinstance`, but that **only checks whether the method names exist** and not the signatures or return types:

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

## An ABC or a Protocol? {#abc-还是-protocol}

| | ABC | Protocol |
| --- | --- | --- |
| Does the implementer have to inherit | yes (or `register` by hand) | no |
| Can it constrain a third party's class | no | yes |
| Can it provide default implementations | yes, which is its strength | technically, but it is not usually used that way |
| When the error surfaces | at instantiation (run time) | at type checking (static) |
| Suits | a base class a framework provides, where subclasses reuse a lot of logic | describing "what capability" a function's parameter needs, decoupling modules |

One good habit in practice: **define the Protocol where it is used**. If `report.py` needs "something that can look orders up by date", define `OrderSource(Protocol)` in `report.py`; the real database implementation and the in-memory one for tests need not know the protocol exists. That is the lightest form of the dependency inversion principle in Python.

!!! interview "How to explain it"
    On protocols: an ABC is nominal typing, has to be inherited, and raises at instantiation when an abstract method is unimplemented; a `Protocol` is structural typing (static duck typing), satisfied as soon as the method signatures line up, with no inheritance, which suits describing "what capability I need" where it is used and makes decoupling and test doubles easy. An ABC from `collections.abc` gives a complete interface from a few abstract methods (a read-only mapping implements `__getitem__`, `__len__` and `__iter__`); `isinstance(x, Iterable)` checks for the method through `__subclasshook__` and requires no inheritance; `@runtime_checkable` checks only that the method names exist, not the signatures. To customize a dict's behaviour, inherit `MutableMapping` or `UserDict` rather than `dict`.

## Exercises {#练习}

**1. A bounded stack.** Implement `BoundedStack(capacity)` by inheriting `collections.abc.Sequence`: it supports `push(x)` (raising `OverflowError` when full) and `pop()`; as a `Sequence`, `s[0]` is the bottom of the stack. Verify that you get `in`, `index`, `count` and `reversed` **for free**.

??? success "Answer"
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

**2. Decoupling with a Protocol.** Write a `send_report(notifier, lines)` that joins the lines and sends them through `notifier.send(subject, body)`. Define a `Notifier` protocol and implement an `EmailNotifier` (which only prints) and a `FakeNotifier` for tests (which stores the messages in a list), neither inheriting `Notifier`.

??? success "Answer"
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

    `send_report` depends only on the protocol, so a test passes the fake implementation and needs no mocking library at all.

## Summary {#小结}

- [x] Duck typing is about what an object can do; ABCs and Protocols write "what is needed" down.
- [x] Inheriting an ABC from `collections.abc` gives a complete container interface from a few methods.
- [x] To customize a dict's behaviour, inherit `MutableMapping` or `UserDict` rather than `dict`.
- [x] A Protocol is structural typing, needs no inheritance, and suits describing the capability a parameter needs.
- [x] Defining the Protocol where it is used is the light way to decouple modules and make testing easy.
