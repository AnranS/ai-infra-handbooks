# Modelling data: dataclass, NamedTuple and Enum

<p class="lead">Many classes are just "a group of named fields". Writing <code>__init__</code>, <code>__repr__</code> and <code>__eq__</code> for them by hand is both verbose and easy to get wrong. This chapter covers expressing data with dataclasses, NamedTuple, TypedDict and Enum, and what each one suits.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What happens with `tags: list = []` in a dataclass? How should it be written?
    2. What does `frozen=True` bring?
    3. What is the main difference between `NamedTuple` and a dataclass?
    4. When should `TypedDict` be used rather than a dataclass?
    5. The enum member `Status.PAID` has the value `"paid"`. What is `Status.PAID == "paid"`? How do you make it true?

??? success "Answers (try it yourself first, then expand)"
    1. It raises outright (a dataclass does not allow a mutable default), because every instance would share that one list. It has to be `tags: list = field(default_factory=list)`.
    2. The instances cannot be modified (assigning to a field raises `FrozenInstanceError`) and a `__hash__` is generated, so they can go in a set and be dict keys; "modifying" means `dataclasses.replace` producing a new object.
    3. A `NamedTuple` is a tuple: immutable, indexable, unpackable and equal to a plain tuple of the same values; a dataclass is an ordinary class, mutable by default and not indexable, which suits a data object with behaviour or with subclasses.
    4. When the data is a dict to begin with (JSON, configuration, `**kwargs`) and has to stay a dict as it is passed around, `TypedDict` adds static type annotations only, and it is still a plain dict at run time.
    5. `False`: an ordinary `Enum` member does not equal its value. Making it true takes `StrEnum` (or inheriting from `str, Enum`), where the member is a string itself; or comparing `Status.PAID.value == "paid"`.

## Why dataclasses exist {#为什么需要-dataclass}

How much boilerplate a "data class" takes by hand:

```py
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __repr__(self):
        return f"Point(x={self.x!r}, y={self.y!r})"
    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return (self.x, self.y) == (other.x, other.y)
```

With a dataclass, only the fields are declared:

```pycon
>>> from dataclasses import dataclass
>>> @dataclass
... class Point:
...     x: float
...     y: float = 0.0
...
>>> p = Point(1.5)
>>> p
Point(x=1.5, y=0.0)
>>> p == Point(1.5, 0.0)
True
```

A field has to carry a type annotation (that is how a dataclass recognizes it), but the type is **not** checked at run time: `Point("a", "b")` is created happily. For run-time validation, do it in `__post_init__` or use a library like pydantic.

## The common dataclass options {#dataclass-常用选项}

### A mutable default needs `field(default_factory=...)` {#可变默认值用-fielddefault_factory}

```pycon
>>> from dataclasses import field
>>> @dataclass
... class Bad:
...     tags: list = []
...
Traceback (most recent call last):
  ...
ValueError: mutable default <class 'list'> for field tags is not allowed: use default_factory
>>> @dataclass
... class Article:
...     title: str
...     tags: list[str] = field(default_factory=list)
...
>>> a, b = Article("A"), Article("B")
>>> a.tags.append("py")
>>> b.tags
[]
```

A dataclass rejects a mutable default outright, which saves you from the trap of [a function's default argument](functions.md#默认参数只求值一次).

### `frozen=True`: immutable and hashable {#frozentrue不可变可哈希}

```pycon
>>> @dataclass(frozen=True)
... class Money:
...     amount: int
...     currency: str = "CNY"
...
>>> m = Money(100)
>>> m.amount = 200
Traceback (most recent call last):
  ...
dataclasses.FrozenInstanceError: cannot assign to field 'amount'
>>> {Money(100), Money(100), Money(5, "USD")} == {Money(100), Money(5, "USD")}
True
```

An immutable object is thread-safe, can be a dict key and cannot be changed by somebody else by accident. **Value objects** (an amount, a coordinate, a configuration snapshot) should all be frozen. To "modify" one, create a new object:

```pycon
>>> from dataclasses import replace
>>> replace(m, amount=300)
Money(amount=300, currency='CNY')
>>> import copy
>>> copy.replace(m, currency="USD")          # the generic spelling from 3.13+
Money(amount=100, currency='USD')
```

### The other options {#其他选项}

| Option | What it does |
| --- | --- |
| `order=True` | generates `<`, `<=` and the rest, comparing in field order |
| `slots=True` | generates `__slots__`, saving memory and forbidding stray attributes |
| `kw_only=True` | every field has to be passed by keyword, which reads better with many fields |
| `field(repr=False)` | left out of the `repr` (a password, say) |
| `field(compare=False)` | left out of equality |
| `field(kw_only=True)` | one field has to be passed by keyword |

### `__post_init__`: validation and derived fields {#\_\_post\_init\_\_校验和派生字段}

```python
from dataclasses import dataclass, field

@dataclass(slots=True)
class Order:
    item: str
    unit_price: float
    quantity: int = 1
    total: float = field(init=False)          # does not appear among __init__'s parameters

    def __post_init__(self):
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        self.total = round(self.unit_price * self.quantity, 2)

o = Order("book", 39.9, 3)
assert o.total == 119.7
assert "total=119.7" in repr(o)
```

### Converting to a dict and a tuple {#转换为字典和元组}

```pycon
>>> from dataclasses import asdict, astuple, fields
>>> asdict(Article("A", ["x"]))
{'title': 'A', 'tags': ['x']}
>>> astuple(Point(1, 2))
(1, 2)
>>> [f.name for f in fields(Money)]
['amount', 'currency']
```

`asdict` handles nested dataclasses, lists and dicts recursively, which is handy before serializing to JSON.

## `NamedTuple`: a tuple with names {#namedtuple带名字的元组}

`typing.NamedTuple` creates **a subclass of tuple**: immutable, hashable, indexable, unpackable, and as small in memory as a plain tuple.

```pycon
>>> from typing import NamedTuple
>>> class Row(NamedTuple):
...     name: str
...     score: int = 0
...
>>> r = Row("amy", 95)
>>> r.name, r[1]
('amy', 95)
>>> name, score = r
>>> r._replace(score=99)
Row(name='amy', score=99)
>>> r._asdict()
{'name': 'amy', 'score': 95}
>>> r == ("amy", 95)                          # it is a tuple, so it equals a plain tuple
True
```

That last line is both an advantage and a drawback: it can replace a function that returned a tuple seamlessly, but "equal to any tuple of the same values" is sometimes not the semantics you want. Old code's `collections.namedtuple("Row", "name score")` is the older spelling of the same thing.

## `TypedDict`: describing a dict's structure {#typeddict给字典描述结构}

When the data **is a dict to begin with** (JSON, configuration, an HTTP body) and you would rather not turn it into an object, `TypedDict` adds type information to it. At run time it is an ordinary `dict` and it means something only to the type checker:

```python
from typing import NotRequired, TypedDict

class UserPayload(TypedDict):
    id: int
    name: str
    email: NotRequired[str]          # this key may be absent

def display(u: UserPayload) -> str:
    return f"{u['name']} <{u.get('email', 'n/a')}>"

payload: UserPayload = {"id": 1, "name": "amy"}
assert type(payload) is dict
assert display(payload) == "amy <n/a>"
```

## Enum: a finite set of named constants {#enum有限的一组命名常量}

Representing a state by a string or a number (`"pending"`, `2`) is easy to misspell and says nothing about which values are legal. An Enum turns them into a type:

```pycon
>>> from enum import Enum, StrEnum, IntFlag, auto
>>> class Status(Enum):
...     PENDING = "pending"
...     PAID = "paid"
...     SHIPPED = "shipped"
...
>>> Status.PAID, Status.PAID.value, Status.PAID.name
(<Status.PAID: 'paid'>, 'paid', 'PAID')
>>> Status("shipped")                         # looked up by value
<Status.SHIPPED: 'shipped'>
>>> Status["PENDING"]                         # looked up by name
<Status.PENDING: 'pending'>
>>> [s.name for s in Status]                  # iterable
['PENDING', 'PAID', 'SHIPPED']
>>> Status("lost")
Traceback (most recent call last):
  ...
ValueError: 'lost' is not a valid Status
```

Enum members are singletons, so `is` can be used to compare them.

### `StrEnum` and `IntEnum` {#strenum-与-intenum}

An ordinary `Enum` member **does not equal** its value: `Status.PAID == "paid"` is `False`. Where it has to interoperate with strings (going straight into JSON, compared against a database column), use `StrEnum`:

```pycon
>>> class Color(StrEnum):
...     RED = auto()          # auto() in a StrEnum gives the lowercased name
...     GREEN = auto()
...
>>> Color.RED == "red", f"color={Color.GREEN}"
(True, 'color=green')
```

`IntEnum` is the same for interoperating with integer protocol codes.

### `Flag`: combinable switches {#flag可组合的开关}

```pycon
>>> class Perm(IntFlag):
...     READ = auto()
...     WRITE = auto()
...     EXEC = auto()
...
>>> rw = Perm.READ | Perm.WRITE
>>> Perm.WRITE in rw, Perm.EXEC in rw
(True, False)
>>> int(rw)
3
```

### Giving an enum behaviour {#给枚举加行为}

An enum is a class too and can define methods and properties. Putting the "state transition rules" in the enum keeps `if`s out of the business code:

```python
from enum import Enum

class OrderState(Enum):
    CREATED = "created"
    PAID = "paid"
    SHIPPED = "shipped"
    CANCELLED = "cancelled"

    def can_go_to(self, target):
        return target in _TRANSITIONS[self]

_TRANSITIONS = {
    OrderState.CREATED: {OrderState.PAID, OrderState.CANCELLED},
    OrderState.PAID: {OrderState.SHIPPED, OrderState.CANCELLED},
    OrderState.SHIPPED: set(),
    OrderState.CANCELLED: set(),
}

assert OrderState.CREATED.can_go_to(OrderState.PAID)
assert not OrderState.SHIPPED.can_go_to(OrderState.CANCELLED)
```

## How to choose {#怎么选}

| Need | Recommendation |
| --- | --- |
| an ordinary data object, possibly with methods | `@dataclass` |
| a value object that has to be hashable and immutable | `@dataclass(frozen=True)` |
| many small objects where memory matters | `@dataclass(slots=True)` or `NamedTuple` |
| replacing an old interface that returned a tuple, with unpacking | `NamedTuple` |
| data that is a dict already (JSON), where only type hints are wanted | `TypedDict` |
| external input needing run-time validation and conversion | pydantic, msgspec or attrs from outside the standard library |
| a fixed set of values | `Enum` / `StrEnum` / `IntFlag` |

!!! interview "How to explain it"
    On modelling data: a class that is mostly data takes `@dataclass`, which generates `__init__`, `__repr__` and `__eq__`; a mutable default has to be `field(default_factory=list)` (a dataclass rejects `tags: list = []` outright); `frozen=True` gives an immutable, hashable value object, and modifying one means a new object from `dataclasses.replace`. A `NamedTuple` is a tuple itself (unpackable, immutable, compared by position) and a `TypedDict` only describes a dict's structure while staying a dict at run time, which suits external data like JSON; a fixed set of values goes to an Enum, and `StrEnum` where it has to interoperate with strings. The requests and sampling parameters in an inference framework are mostly dataclasses or msgspec structs.

## Exercises {#练习}

**1. Modelling a shopping cart.** Model it with dataclasses: a `LineItem` (the product's name, unit price and quantity, frozen); a `Cart` (a group of `LineItem`s, mutable). `Cart` offers `add(item)`, a `total` property (to two decimal places) and a `from_dict(data)` class method building it from `{"items": [{"name": ..., "price": ..., "qty": ...}]}`.

??? success "Answer"
    ```python
    from dataclasses import dataclass, field

    @dataclass(frozen=True, slots=True)
    class LineItem:
        name: str
        price: float
        qty: int = 1

        def __post_init__(self):
            if self.price < 0 or self.qty <= 0:
                raise ValueError(f"invalid line item: {self}")

        @property
        def subtotal(self):
            return self.price * self.qty

    @dataclass
    class Cart:
        items: list[LineItem] = field(default_factory=list)

        def add(self, item: LineItem) -> None:
            self.items.append(item)

        @property
        def total(self) -> float:
            return round(sum(i.subtotal for i in self.items), 2)

        @classmethod
        def from_dict(cls, data: dict) -> "Cart":
            return cls([LineItem(d["name"], d["price"], d.get("qty", 1)) for d in data["items"]])

    cart = Cart.from_dict({"items": [{"name": "pen", "price": 2.5, "qty": 4}, {"name": "book", "price": 39.9}]})
    cart.add(LineItem("bag", 0.1, 3))
    assert cart.total == 50.2
    ```

    `round` does not solve every floating-point problem. Real money arithmetic should use integer "cents" or `decimal.Decimal`, see [idioms and common traps](../practice/pitfalls.md).

**2. HTTP status codes.** Define `HttpStatus(IntEnum)` with 200, 201, 400, 404 and 500, plus `is_success` and `is_client_error` properties. Verify that `HttpStatus(404).is_client_error` is true and that `HttpStatus.OK == 200`.

??? success "Answer"
    ```python
    from enum import IntEnum

    class HttpStatus(IntEnum):
        OK = 200
        CREATED = 201
        BAD_REQUEST = 400
        NOT_FOUND = 404
        SERVER_ERROR = 500

        @property
        def is_success(self):
            return 200 <= self < 300

        @property
        def is_client_error(self):
            return 400 <= self < 500

    assert HttpStatus(404).is_client_error and not HttpStatus(404).is_success
    assert HttpStatus.OK == 200 and HttpStatus.CREATED.is_success
    ```

    The standard library already has `http.HTTPStatus`, which is an `IntEnum` with `is_success` and similar properties (3.12+).

## Summary {#小结}

- [x] A class that is mostly data takes `@dataclass`, and a mutable default takes `field(default_factory=...)`.
- [x] A value object takes `frozen=True`, and "modifying" means a new object from `replace`.
- [x] A `NamedTuple` is a tuple and a `TypedDict` is a dict; each has its place.
- [x] A fixed set of values goes to an Enum, and to `StrEnum` or `IntEnum` where it has to interoperate with strings or integers.
- [x] Where external input needs run-time validation, consider a library like pydantic.
