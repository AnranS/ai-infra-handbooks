# 面向对象

<p class="lead">Python 的面向对象很"轻"：没有访问控制关键字，没有接口声明，更多靠约定和协议。这一章讲如何写出好用的类：属性与方法的各种形式、继承与 <code>super</code>、以及让你的对象像内置类型一样自然的特殊方法。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 类属性和实例属性同名时，读取的是哪个？给实例赋值时改的是哪个？
    2. `classmethod` 最典型的用途是什么？
    3. 多重继承时 `super()` 调用的一定是父类吗？
    4. `__repr__` 和 `__str__` 的区别？只能实现一个时实现哪个？
    5. `a + b` 在 `a.__add__(b)` 返回 `NotImplemented` 时会发生什么？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 读取时先找实例属性，找不到再找类属性；给实例赋值总是在实例上创建（或修改）属性，不会改变类属性。
    2. 备选构造函数，比如 `from_json`、`from_config`：用 `cls(...)` 创建对象，子类调用时创建的就是子类的实例。
    3. 不一定：`super()` 调用的是 MRO（方法解析顺序）里的下一个类，在多重继承中可能是兄弟类。所以协作式的多重继承要求每一层都调用 `super()`。
    4. `__repr__` 面向开发者，尽量无歧义（最好能看出怎么重新创建对象）；`__str__` 面向用户、可读。只实现一个时实现 `__repr__`：没有 `__str__` 时会用它。
    5. Python 会尝试 `b.__radd__(a)`（反向运算）；如果它也返回 `NotImplemented`，就抛出 `TypeError`。

## 类属性与实例属性

```pycon
>>> class Dog:
...     species = "canis"            # 类属性：所有实例共享
...     def __init__(self, name):
...         self.name = name         # 实例属性：每个实例一份
...
>>> a, b = Dog("rex"), Dog("max")
>>> a.species, b.species
('canis', 'canis')
>>> a.species = "wolf"               # 在实例上创建了同名属性，遮住了类属性
>>> a.species, b.species, Dog.species
('wolf', 'canis', 'canis')
>>> vars(a)
{'name': 'rex', 'species': 'wolf'}
```

读取属性时先找实例的 `__dict__`，找不到再去类（以及父类）里找；赋值时总是写到实例上。

!!! warning "可变的类属性会被所有实例共享"
    ```py
    class Team:
        members = []                 # 所有 Team 实例共用这一个列表！
        def add(self, m):
            self.members.append(m)   # 这是修改，不是赋值，改到的是类属性
    ```
    每个实例自己的数据，一定要在 `__init__` 里创建。

## 三种方法

```python
class Temperature:
    def __init__(self, celsius):
        self.celsius = celsius

    def to_fahrenheit(self):                 # 实例方法：操作某个实例
        return self.celsius * 9 / 5 + 32

    @classmethod
    def from_fahrenheit(cls, f):             # 类方法：备选构造函数
        return cls((f - 32) * 5 / 9)

    @staticmethod
    def is_valid(celsius):                   # 静态方法：和类相关的普通函数
        return celsius >= -273.15

t = Temperature.from_fahrenheit(212)
assert t.celsius == 100 and t.to_fahrenheit() == 212
assert Temperature.is_valid(-300) is False
```

`classmethod` 的第一个参数是**类本身**，所以子类调用时得到的是子类的实例。标准库里的 `dict.fromkeys`、`datetime.fromisoformat`、`int.from_bytes` 都是这样的备选构造函数。

`staticmethod` 用得较少。如果函数和类的关系不大，直接写成模块级函数更好。

## `property`：把方法伪装成属性

先用普通属性，等真的需要校验或计算时再换成 `property`，调用方的代码一行都不用改。这是 Python 不需要写 getter/setter 的原因。

```python
class Account:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance           # 这里也会走 setter 的校验

    @property
    def balance(self):
        return self._balance

    @balance.setter
    def balance(self, value):
        if value < 0:
            raise ValueError("balance cannot be negative")
        self._balance = value

    @property
    def is_rich(self):                   # 只读的计算属性
        return self._balance > 1_000_000

acct = Account("amy", 100)
acct.balance += 50
assert acct.balance == 150 and not acct.is_rich
try:
    acct.balance = -1
except ValueError as e:
    assert str(e) == "balance cannot be negative"
```

### `cached_property`：只算一次

计算代价高、结果不会变的属性，用 `functools.cached_property`。第一次访问时计算，结果存进实例的 `__dict__`，之后直接读取：

```python
from functools import cached_property

class Report:
    def __init__(self, rows):
        self.rows = rows

    @cached_property
    def summary(self):
        print("computing...")
        return {"count": len(self.rows), "total": sum(self.rows)}

r = Report([3, 4, 5])
assert r.summary["total"] == 12      # 打印 computing...
assert r.summary["count"] == 3       # 不再计算
del r.summary                        # 删除缓存，下次访问重新计算
```

## 继承与 `super()`

![图：菱形继承的方法解析顺序——C3 线性化](../assets/figures/mro-diamond.svg){.aig-svg}

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return "..."

    def introduce(self):
        return f"I am {self.name}, I say {self.speak()}"

class Cat(Animal):
    def __init__(self, name, indoor=True):
        super().__init__(name)           # 先让父类完成它的初始化
        self.indoor = indoor

    def speak(self):
        return "meow"

assert Cat("tom").introduce() == "I am tom, I say meow"
assert isinstance(Cat("tom"), Animal) and issubclass(Cat, Animal)
```

### MRO 与协作式 `super()`

多重继承时，Python 用 C3 算法计算出一个**方法解析顺序（MRO）**。`super()` 的准确含义是"**MRO 中的下一个类**"，不一定是父类。

```pycon
>>> class Base:
...     def hello(self):
...         return ["Base"]
...
>>> class A(Base):
...     def hello(self):
...         return ["A"] + super().hello()
...
>>> class B(Base):
...     def hello(self):
...         return ["B"] + super().hello()
...
>>> class C(A, B):
...     def hello(self):
...         return ["C"] + super().hello()
...
>>> [k.__name__ for k in C.__mro__]
['C', 'A', 'B', 'Base', 'object']
>>> C().hello()
['C', 'A', 'B', 'Base']
```

注意 `A` 里的 `super()` 调用到了 `B`，而 `A` 根本不知道 `B` 的存在。这就是"协作式多重继承"：每个类都调用 `super()`，整条链上的每个方法都会恰好执行一次。

### Mixin

Mixin 是只提供一小块功能、不单独使用的类，通过多重继承"混入"。约定上名字以 `Mixin` 结尾，放在基类列表的左边：

```python
import json

class JsonMixin:
    def to_json(self):
        return json.dumps(vars(self), sort_keys=True)

class ReprMixin:
    def __repr__(self):
        fields = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({fields})"

class User(JsonMixin, ReprMixin):
    def __init__(self, name, age):
        self.name, self.age = name, age

u = User("amy", 30)
assert u.to_json() == '{"age": 30, "name": "amy"}'
assert repr(u) == "User(name='amy', age=30)"
```

!!! tip "组合优于继承"
    继承表达"是一个"（is-a），组合表达"有一个"（has-a）。当你只是想复用某个类的功能时，把它作为属性持有（组合）通常比继承它更灵活：可以在运行时替换，也不会继承一堆不需要的方法。深的继承层次是维护的噩梦，Python 里两三层已经很多了。

## 特殊方法：让对象融入语言

特殊方法（dunder methods，双下划线方法）是 Python 数据模型的核心。实现它们，你的对象就能配合内置函数和运算符工作。

### `__repr__` 与 `__str__`

- `__repr__`：给开发者看的，要求**无歧义**，理想情况下 `eval(repr(obj)) == obj`。REPL、调试器、容器打印时用它。
- `__str__`：给用户看的，要求**可读**。`print()` 和 `str()` 用它，没有定义时回退到 `__repr__`。

**只实现一个的话，实现 `__repr__`。**

### 一个完整的例子：二维向量

```python
import math
from functools import total_ordering

@total_ordering
class Vector:
    __slots__ = ("x", "y")

    def __init__(self, x=0.0, y=0.0):
        self.x, self.y = x, y

    def __repr__(self):
        return f"Vector({self.x!r}, {self.y!r})"

    def __eq__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        return (self.x, self.y) == (other.x, other.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __lt__(self, other):                 # 按长度比较，其余比较由 total_ordering 补全
        if not isinstance(other, Vector):
            return NotImplemented
        return abs(self) < abs(other)

    def __abs__(self):
        return math.hypot(self.x, self.y)

    def __bool__(self):
        return bool(self.x or self.y)

    def __add__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(self.x + other.x, self.y + other.y)

    def __mul__(self, k):
        if not isinstance(k, (int, float)):
            return NotImplemented
        return Vector(self.x * k, self.y * k)

    __rmul__ = __mul__                       # 支持 3 * v

    def __iter__(self):                      # 支持解包 x, y = v
        yield self.x
        yield self.y
```

```pycon
>>> v = Vector(3, 4)
>>> v + Vector(1, 1), v * 2, 2 * v
(Vector(4, 5), Vector(6, 8), Vector(6, 8))
>>> abs(v), bool(Vector()), Vector(1, 1) <= v
(5.0, False, True)
>>> x, y = v
>>> x, y
(3, 4)
>>> v + 1
Traceback (most recent call last):
  ...
TypeError: unsupported operand type(s) for +: 'Vector' and 'int'
```

关于 `NotImplemented`：二元运算 `a + b` 先调用 `a.__add__(b)`；如果返回 `NotImplemented`，Python 会再尝试 `b.__radd__(a)`；都不行才抛 `TypeError`。所以遇到不认识的类型时，**返回 `NotImplemented` 而不是抛异常**，给对方一个机会。

### 常用特殊方法速查

| 想支持的操作 | 需要实现 |
| --- | --- |
| `repr(x)` / `str(x)` / `format(x, spec)` | `__repr__` / `__str__` / `__format__` |
| `x == y`、`x < y` 等 | `__eq__`、`__lt__` 等（配合 `total_ordering`） |
| 放进 `set`、当 `dict` 的键 | `__hash__`（与 `__eq__` 一致） |
| `if x:` | `__bool__` 或 `__len__` |
| `len(x)`、`x[i]`、`x[i] = v`、`del x[i]` | `__len__`、`__getitem__`、`__setitem__`、`__delitem__` |
| `for a in x` | `__iter__` |
| `a in x` | `__contains__`（否则回退到遍历） |
| `x + y`、`y + x`、`x += y` | `__add__`、`__radd__`、`__iadd__` |
| `x(...)` | `__call__` |
| `with x:` | `__enter__`、`__exit__` |

### 序列协议：只要 `__len__` 和 `__getitem__`

```pycon
>>> class Deck:
...     ranks = [str(n) for n in range(2, 11)] + list("JQKA")
...     suits = "♠♥♦♣"
...     def __init__(self):
...         self._cards = [r + s for s in self.suits for r in self.ranks]
...     def __len__(self):
...         return len(self._cards)
...     def __getitem__(self, i):
...         return self._cards[i]
...
>>> deck = Deck()
>>> len(deck), deck[0], deck[-1]
(52, '2♠', 'A♣')
>>> deck[:3]                         # 切片自动可用
['2♠', '3♠', '4♠']
>>> "Q♥" in deck                     # 没有 __contains__ 时自动遍历
True
>>> next(iter(deck))                 # 没有 __iter__ 时自动用 __getitem__ 迭代
'2♠'
```

只实现了两个方法，就获得了 `len`、索引、切片、迭代、`in`、`reversed`、`random.choice` 的支持。这就是"鸭子类型"和数据模型的力量。

## `__slots__`：固定属性、节省内存

默认情况下每个实例都有一个 `__dict__` 存属性。声明 `__slots__` 后，实例改用固定的槽位存储：

- 内存占用明显减少（创建百万级小对象时有意义）；
- 不能再添加 `__slots__` 之外的属性，拼错属性名会直接报错。

```pycon
>>> class P:
...     __slots__ = ("x", "y")
...     def __init__(self, x, y):
...         self.x, self.y = x, y
...
>>> p = P(1, 2)
>>> p.z = 3
Traceback (most recent call last):
  ...
AttributeError: 'P' object has no attribute 'z' and no __dict__ for setting new attributes
```

dataclass 可以用 `@dataclass(slots=True)` 自动生成 `__slots__`。

## 命名约定代替访问控制

| 写法 | 含义 |
| --- | --- |
| `name` | 公开 API |
| `_name` | 内部使用，"请不要从外部访问"。`from m import *` 不会导入 |
| `__name` | 触发**名字改写**：变成 `_类名__name`，用于避免子类意外覆盖，不是为了"私有" |
| `__name__` | 语言保留的特殊名字，不要自己发明 |

!!! interview "怎么讲清楚"
    讲面向对象：类属性被所有实例共享，同名时实例属性遮住类属性，可变的类属性是常见 bug；`classmethod` 做备选构造函数（用 `cls` 对子类友好），`staticmethod` 只是放在类里的普通函数；`property` 让你先用普通属性、以后再加校验而不改接口；`super()` 是"MRO 里的下一个"，不一定是父类，多重继承时每层都要调用；二元运算遇到不认识的操作数返回 `NotImplemented`，让 Python 去试对方的反向方法；`__repr__` 给开发者看、至少实现它；`__slots__` 省内存、禁止动态加属性。

## 练习

**1. 有校验的区间类。** 实现 `Range(lo, hi)` 表示闭区间：构造时如果 `lo > hi` 抛 `ValueError`；支持 `x in r`、`len(r)`（区间内整数个数）、`r1 & r2`（交集，不相交时返回 `None`）、友好的 `repr`、相等比较。

??? success "参考答案"
    ```python
    class Range:
        __slots__ = ("lo", "hi")

        def __init__(self, lo, hi):
            if lo > hi:
                raise ValueError(f"lo ({lo}) must be <= hi ({hi})")
            self.lo, self.hi = lo, hi

        def __repr__(self):
            return f"Range({self.lo}, {self.hi})"

        def __eq__(self, other):
            if not isinstance(other, Range):
                return NotImplemented
            return (self.lo, self.hi) == (other.lo, other.hi)

        def __hash__(self):
            return hash((self.lo, self.hi))

        def __contains__(self, x):
            return self.lo <= x <= self.hi

        def __len__(self):
            import math
            return max(0, math.floor(self.hi) - math.ceil(self.lo) + 1)

        def __and__(self, other):
            if not isinstance(other, Range):
                return NotImplemented
            lo, hi = max(self.lo, other.lo), min(self.hi, other.hi)
            return Range(lo, hi) if lo <= hi else None

    r = Range(1, 10)
    assert 5 in r and 11 not in r and 2.5 in r
    assert len(r) == 10 and len(Range(0.5, 2.5)) == 2
    assert (r & Range(8, 20)) == Range(8, 10)
    assert (r & Range(11, 12)) is None
    assert repr(r) == "Range(1, 10)"
    ```

**2. 备选构造函数与继承。** 给正文的 `Temperature` 写一个子类 `LabTemperature`，额外带一个 `sensor` 属性（默认 `"unknown"`）。验证 `LabTemperature.from_fahrenheit(212)` 返回的是 `LabTemperature` 实例。如果父类的 `from_fahrenheit` 写成了 `return Temperature(...)`，会怎样？

??? success "参考答案"
    ```python
    class Temperature:
        def __init__(self, celsius):
            self.celsius = celsius

        @classmethod
        def from_fahrenheit(cls, f):
            return cls((f - 32) * 5 / 9)

    class LabTemperature(Temperature):
        def __init__(self, celsius, sensor="unknown"):
            super().__init__(celsius)
            self.sensor = sensor

    t = LabTemperature.from_fahrenheit(212)
    assert type(t) is LabTemperature and t.sensor == "unknown" and t.celsius == 100
    ```

    如果写死 `Temperature(...)`，子类调用时拿到的是父类实例，丢了 `sensor` 属性。这正是用 `cls` 而不是类名的原因。注意子类的 `__init__` 新增参数要有默认值，否则 `cls(...)` 只传一个参数时会失败。

**3. 协作式初始化。** 写三个类：`Base` 接收 `name`，`Leveled(Base)` 接收 `level`（默认 1），`Colored(Base)` 接收 `color`（默认 `"black"`），再定义 `class Full(Leveled, Colored)`。要求 `Full("x", level=2, color="red")` 让每个类都拿到自己的参数：每个 `__init__` 只取自己关心的关键字参数，其余的通过 `super().__init__(**kwargs)` 往下传。

??? success "参考答案"
    ```python
    class Base:
        def __init__(self, name, **kwargs):
            super().__init__(**kwargs)       # 最终传给 object.__init__，此时 kwargs 应为空
            self.name = name

    class Leveled(Base):
        def __init__(self, *, level=1, **kwargs):
            super().__init__(**kwargs)
            self.level = level

    class Colored(Base):
        def __init__(self, *, color="black", **kwargs):
            super().__init__(**kwargs)
            self.color = color

    class Full(Leveled, Colored):
        def __init__(self, name, **kwargs):
            super().__init__(name=name, **kwargs)

    f = Full("x", level=2, color="red")
    assert (f.name, f.level, f.color) == ("x", 2, "red")
    assert [c.__name__ for c in Full.__mro__] == ["Full", "Leveled", "Colored", "Base", "object"]
    ```

    调用链沿着 MRO 走：`Full → Leveled → Colored → Base → object`，每一层取走自己的参数。如果有人传了拼错的参数，最终会在 `object.__init__` 处报错，而不是被悄悄忽略。

## 小结

- [x] 实例数据在 `__init__` 里创建；可变的类属性会被所有实例共享。
- [x] `classmethod` 做备选构造函数，用 `cls` 保证子类友好。
- [x] 先用普通属性，需要时再换成 `property`，接口不变。
- [x] `super()` 是"MRO 中的下一个"，多重继承时每层都调用它。
- [x] 实现特殊方法让对象融入语言；不认识的操作数返回 `NotImplemented`；至少实现 `__repr__`。
