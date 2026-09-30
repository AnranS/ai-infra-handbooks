# 数据建模：dataclass、NamedTuple 与 Enum

<p class="lead">很多类只是"一组有名字的字段"。为它们手写 <code>__init__</code>、<code>__repr__</code>、<code>__eq__</code> 既啰嗦又容易出错。这一章讲怎么用 dataclass、NamedTuple、TypedDict、Enum 来表达数据，以及它们各自适合什么场景。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. dataclass 里写 `tags: list = []` 会怎样？应该怎么写？
    2. `frozen=True` 带来了什么？
    3. `NamedTuple` 和 dataclass 的主要区别是什么？
    4. 什么时候该用 `TypedDict` 而不是 dataclass？
    5. 枚举成员 `Status.PAID` 的值是 `"paid"`，`Status.PAID == "paid"` 的结果是什么？怎样让它成立？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 直接报错（dataclass 不允许可变的默认值），因为所有实例会共享同一个列表。应该写 `tags: list = field(default_factory=list)`。
    2. 实例不可修改（给字段赋值会抛出 `FrozenInstanceError`），并且自动生成 `__hash__`，可以放进集合、作为字典的键；"修改"用 `dataclasses.replace` 生成新对象。
    3. `NamedTuple` 是元组：不可变、可以按下标访问和解包、和普通元组比较相等；dataclass 是普通的类，默认可变，不能按下标访问，更适合有行为、需要继承的数据对象。
    4. 数据本来就是字典（JSON、配置、`**kwargs`）、需要保持字典的形态传来传去时，用 `TypedDict` 只做静态的类型标注，运行时仍然是普通的字典。
    5. `False`：普通的 `Enum` 成员不等于它的值。让它成立要用 `StrEnum`（或继承 `str, Enum`），成员本身就是字符串；或者比较 `Status.PAID.value == "paid"`。

## 为什么需要 dataclass

手写一个"数据类"要写多少样板代码：

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

用 dataclass，只需要声明字段：

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

字段必须带类型标注（dataclass 靠标注识别字段），但类型**不会**在运行时检查：`Point("a", "b")` 也能创建成功。需要运行时校验时，在 `__post_init__` 里做，或者使用 pydantic 之类的第三方库。

## dataclass 常用选项

### 可变默认值用 `field(default_factory=...)`

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

dataclass 直接拒绝了可变默认值，替你挡掉了[函数默认参数](functions.md#默认参数只求值一次)的那个坑。

### `frozen=True`：不可变、可哈希

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

不可变对象线程安全、可以当字典的键、不用担心被别人意外修改。**值对象**（金额、坐标、配置快照）都应该是 frozen 的。需要"修改"时，创建一个新对象：

```pycon
>>> from dataclasses import replace
>>> replace(m, amount=300)
Money(amount=300, currency='CNY')
>>> import copy
>>> copy.replace(m, currency="USD")          # 3.13+ 的通用写法
Money(amount=100, currency='USD')
```

### 其他选项

| 选项 | 作用 |
| --- | --- |
| `order=True` | 按字段顺序生成 `<`、`<=` 等比较方法 |
| `slots=True` | 生成 `__slots__`，更省内存，禁止随意添加属性 |
| `kw_only=True` | 所有字段都只能按关键字传参，字段多的时候更可读 |
| `field(repr=False)` | 不在 `repr` 里显示（比如密码） |
| `field(compare=False)` | 不参与相等比较 |
| `field(kw_only=True)` | 单个字段只能按关键字传 |

### `__post_init__`：校验和派生字段

```python
from dataclasses import dataclass, field

@dataclass(slots=True)
class Order:
    item: str
    unit_price: float
    quantity: int = 1
    total: float = field(init=False)          # 不出现在 __init__ 参数里

    def __post_init__(self):
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        self.total = round(self.unit_price * self.quantity, 2)

o = Order("book", 39.9, 3)
assert o.total == 119.7
assert "total=119.7" in repr(o)
```

### 转换为字典和元组

```pycon
>>> from dataclasses import asdict, astuple, fields
>>> asdict(Article("A", ["x"]))
{'title': 'A', 'tags': ['x']}
>>> astuple(Point(1, 2))
(1, 2)
>>> [f.name for f in fields(Money)]
['amount', 'currency']
```

`asdict` 会递归处理嵌套的 dataclass、列表和字典，序列化成 JSON 前很方便。

## `NamedTuple`：带名字的元组

`typing.NamedTuple` 创建的是**元组的子类**：不可变、可哈希、可以按索引访问、可以解包，内存占用和普通元组一样小。

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
>>> r == ("amy", 95)                          # 它就是元组，会和普通元组相等
True
```

最后一行既是优点也是缺点：它可以无缝替换原来返回元组的函数，但"和任何同值元组相等"有时并不是你想要的语义。老代码里的 `collections.namedtuple("Row", "name score")` 是同一个东西的旧写法。

## `TypedDict`：给字典描述结构

当数据**本来就是字典**（JSON、配置、HTTP 请求体），而你又不想转换成对象时，用 `TypedDict` 给它加上类型信息。它在运行时就是普通的 `dict`，只对类型检查器有意义：

```python
from typing import NotRequired, TypedDict

class UserPayload(TypedDict):
    id: int
    name: str
    email: NotRequired[str]          # 这个键可以不存在

def display(u: UserPayload) -> str:
    return f"{u['name']} <{u.get('email', 'n/a')}>"

payload: UserPayload = {"id": 1, "name": "amy"}
assert type(payload) is dict
assert display(payload) == "amy <n/a>"
```

## Enum：有限的一组命名常量

用字符串或数字表示状态（`"pending"`、`2`）很容易拼错，也看不出一共有哪些合法值。Enum 把它们变成类型：

```pycon
>>> from enum import Enum, StrEnum, IntFlag, auto
>>> class Status(Enum):
...     PENDING = "pending"
...     PAID = "paid"
...     SHIPPED = "shipped"
...
>>> Status.PAID, Status.PAID.value, Status.PAID.name
(<Status.PAID: 'paid'>, 'paid', 'PAID')
>>> Status("shipped")                         # 按值查找
<Status.SHIPPED: 'shipped'>
>>> Status["PENDING"]                         # 按名字查找
<Status.PENDING: 'pending'>
>>> [s.name for s in Status]                  # 可以遍历
['PENDING', 'PAID', 'SHIPPED']
>>> Status("lost")
Traceback (most recent call last):
  ...
ValueError: 'lost' is not a valid Status
```

枚举成员是单例，比较时可以用 `is`。

### `StrEnum` 与 `IntEnum`

普通 `Enum` 成员**不等于**它的值：`Status.PAID == "paid"` 是 `False`。需要和字符串互通（比如直接写进 JSON、和数据库字段比较）时，用 `StrEnum`：

```pycon
>>> class Color(StrEnum):
...     RED = auto()          # StrEnum 里 auto() 生成小写的名字
...     GREEN = auto()
...
>>> Color.RED == "red", f"color={Color.GREEN}"
(True, 'color=green')
```

`IntEnum` 同理，适合和整数协议码互通。

### `Flag`：可组合的开关

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

### 给枚举加行为

枚举也是类，可以定义方法和属性。把"状态转移规则"写在枚举里，业务代码就不用到处写 `if`：

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

## 怎么选

| 需求 | 推荐 |
| --- | --- |
| 一般的数据对象，可能有方法 | `@dataclass` |
| 值对象，需要可哈希、不可变 | `@dataclass(frozen=True)` |
| 大量小对象，关心内存 | `@dataclass(slots=True)` 或 `NamedTuple` |
| 替换返回元组的旧接口、需要解包 | `NamedTuple` |
| 数据本来就是 dict（JSON），只想要类型提示 | `TypedDict` |
| 外部输入需要运行时校验和类型转换 | 第三方的 pydantic、msgspec、attrs |
| 一组固定的取值 | `Enum` / `StrEnum` / `IntFlag` |

!!! interview "面试怎么答"
    数据建模题：数据为主的类用 `@dataclass`，自动生成 `__init__`、`__repr__`、`__eq__`；可变默认值必须用 `field(default_factory=list)`（dataclass 直接拒绝 `tags: list = []`）；`frozen=True` 得到不可变、可哈希的值对象，修改用 `dataclasses.replace` 生成新对象。`NamedTuple` 本身是元组（能解包、不可变、按位置比较），`TypedDict` 只是给字典描述结构、运行时仍是 dict，适合 JSON 这类外部数据；固定的取值集合用 Enum，要和字符串互通用 `StrEnum`。推理框架里的请求、采样参数大多是 dataclass 或 msgspec 结构体。

## 练习

**1. 购物车建模。** 用 dataclass 建模：`LineItem`（商品名、单价、数量，frozen）；`Cart`（一组 `LineItem`，可变）。`Cart` 提供 `add(item)`、`total` 属性（保留两位小数），以及 `from_dict(data)` 类方法，从 `{"items": [{"name": ..., "price": ..., "qty": ...}]}` 构建。

??? success "参考答案"
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

    `round` 解决不了所有浮点问题。真实的金额计算应该用整数"分"或者 `decimal.Decimal`，见[惯用法与常见坑](../practice/pitfalls.md)。

**2. HTTP 状态码。** 定义 `HttpStatus(IntEnum)`，包含 200、201、400、404、500，并添加 `is_success` 和 `is_client_error` 两个属性。验证 `HttpStatus(404).is_client_error` 为真，并且 `HttpStatus.OK == 200`。

??? success "参考答案"
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

    标准库里其实已经有 `http.HTTPStatus`，它就是一个 `IntEnum`，并且带 `is_success` 等属性（3.12+）。

## 小结

- [x] 数据为主的类用 `@dataclass`，可变默认值用 `field(default_factory=...)`。
- [x] 值对象用 `frozen=True`，"修改"用 `replace` 生成新对象。
- [x] `NamedTuple` 是元组；`TypedDict` 是字典；它们各有适合的场景。
- [x] 固定的取值集合用 Enum；需要和字符串、整数互通时用 `StrEnum`、`IntEnum`。
- [x] 需要运行时校验外部输入时，考虑 pydantic 等库。
