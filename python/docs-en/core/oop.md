# Object orientation

<p class="lead">Python's object orientation is "light": no access control keywords, no interface declarations, and far more convention and protocol. This chapter covers how to write a class that is good to use: the forms attributes and methods take, inheritance and <code>super</code>, and the special methods that make your objects feel as natural as the built-in types.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. When a class attribute and an instance attribute share a name, which does a read see? Which does assigning to the instance change?
    2. What is `classmethod`'s most typical use?
    3. Under multiple inheritance, does `super()` always call the parent class?
    4. What is the difference between `__repr__` and `__str__`? If only one can be implemented, which?
    5. What happens with `a + b` when `a.__add__(b)` returns `NotImplemented`?

??? success "Answers (try it yourself first, then expand)"
    1. A read looks at the instance attribute first and falls back to the class attribute; assigning to the instance always creates (or changes) an attribute on the instance and never touches the class attribute.
    2. An alternative constructor, `from_json` or `from_config` say: it creates the object through `cls(...)`, so a subclass's call creates an instance of the subclass.
    3. Not always: `super()` calls the next class in the MRO (the method resolution order), which under multiple inheritance may be a sibling. So cooperative multiple inheritance requires every level to call `super()`.
    4. `__repr__` is for developers and should be unambiguous (ideally showing how to recreate the object); `__str__` is for users and should be readable. Implement `__repr__` if only one: it is used when there is no `__str__`.
    5. Python tries `b.__radd__(a)` (the reflected operation); if that returns `NotImplemented` too, it raises `TypeError`.

## Class attributes and instance attributes {#类属性与实例属性}

```pycon
>>> class Dog:
...     species = "canis"            # a class attribute: shared by every instance
...     def __init__(self, name):
...         self.name = name         # an instance attribute: one per instance
...
>>> a, b = Dog("rex"), Dog("max")
>>> a.species, b.species
('canis', 'canis')
>>> a.species = "wolf"               # creates an attribute of the same name on the instance, hiding the class attribute
>>> a.species, b.species, Dog.species
('wolf', 'canis', 'canis')
>>> vars(a)
{'name': 'rex', 'species': 'wolf'}
```

Reading an attribute looks in the instance's `__dict__` first and then in the class (and its parents); assigning always writes to the instance.

!!! warning "A mutable class attribute is shared by every instance"
    ```py
    class Team:
        members = []                 # every Team instance shares this one list!
        def add(self, m):
            self.members.append(m)   # this modifies rather than assigns, so it reaches the class attribute
    ```
    Data belonging to each instance has to be created in `__init__`.

## The three kinds of method {#三种方法}

```python
class Temperature:
    def __init__(self, celsius):
        self.celsius = celsius

    def to_fahrenheit(self):                 # an instance method: works on one instance
        return self.celsius * 9 / 5 + 32

    @classmethod
    def from_fahrenheit(cls, f):             # a class method: an alternative constructor
        return cls((f - 32) * 5 / 9)

    @staticmethod
    def is_valid(celsius):                   # a static method: an ordinary function related to the class
        return celsius >= -273.15

t = Temperature.from_fahrenheit(212)
assert t.celsius == 100 and t.to_fahrenheit() == 212
assert Temperature.is_valid(-300) is False
```

A `classmethod`'s first parameter is **the class itself**, so a subclass's call gives an instance of the subclass. The standard library's `dict.fromkeys`, `datetime.fromisoformat` and `int.from_bytes` are all alternative constructors of this kind.

`staticmethod` is used less. When a function has little to do with the class, a module-level function is better.

## `property`: a method disguised as an attribute {#property把方法伪装成属性}

Use a plain attribute first and switch to a `property` when validation or computation is really needed, without the callers changing a line. That is why Python needs no getters and setters.

```python
class Account:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance           # this goes through the setter's validation too

    @property
    def balance(self):
        return self._balance

    @balance.setter
    def balance(self, value):
        if value < 0:
            raise ValueError("balance cannot be negative")
        self._balance = value

    @property
    def is_rich(self):                   # a read-only computed attribute
        return self._balance > 1_000_000

acct = Account("amy", 100)
acct.balance += 50
assert acct.balance == 150 and not acct.is_rich
try:
    acct.balance = -1
except ValueError as e:
    assert str(e) == "balance cannot be negative"
```

### `cached_property`: computed once {#cached_property只算一次}

For an attribute that is expensive and never changes, use `functools.cached_property`. It computes on the first access, stores the result in the instance's `__dict__`, and reads it directly afterwards:

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
assert r.summary["total"] == 12      # prints computing...
assert r.summary["count"] == 3       # computed no more
del r.summary                        # drop the cache, so the next access recomputes
```

## Inheritance and `super()` {#继承与-super}

![Figure: the method resolution order of diamond inheritance - C3 linearization](../assets/figures/mro-diamond.svg){.aig-svg}

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
        super().__init__(name)           # let the parent finish its initialization first
        self.indoor = indoor

    def speak(self):
        return "meow"

assert Cat("tom").introduce() == "I am tom, I say meow"
assert isinstance(Cat("tom"), Animal) and issubclass(Cat, Animal)
```

### The MRO and cooperative `super()` {#mro-与协作式-super}

Under multiple inheritance, Python computes a **method resolution order (MRO)** with the C3 algorithm. `super()` means exactly "**the next class in the MRO**", which is not necessarily the parent.

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

Note that `super()` in `A` reached `B`, and `A` knows nothing of `B`'s existence. That is "cooperative multiple inheritance": every class calls `super()` and every method along the chain runs exactly once.

### Mixins {#mixin}

A mixin is a class that provides one small piece of behaviour and is never used on its own, "mixed in" through multiple inheritance. By convention its name ends in `Mixin` and it goes on the left of the base list:

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

!!! tip "Composition over inheritance"
    Inheritance expresses "is a" and composition "has a". When all you want is to reuse a class's behaviour, holding it as an attribute (composition) is usually more flexible than inheriting it: it can be replaced at run time and brings no pile of unwanted methods. A deep inheritance hierarchy is a maintenance nightmare, and two or three levels is already a lot in Python.

## Special methods: fitting an object into the language {#特殊方法让对象融入语言}

The special methods (dunder methods) are the heart of Python's data model. Implement them and your object works with the built-in functions and the operators.

### `__repr__` and `__str__` {#\_\_repr\_\_-与-\_\_str\_\_}

- `__repr__`: for developers, required to be **unambiguous**, ideally so that `eval(repr(obj)) == obj`. The REPL, the debugger and printing a container use it.
- `__str__`: for users, required to be **readable**. `print()` and `str()` use it and fall back to `__repr__` when it is not defined.

**If only one is implemented, implement `__repr__`.**

### A complete example: a two-dimensional vector {#一个完整的例子二维向量}

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

    def __lt__(self, other):                 # compared by length, with total_ordering filling in the rest
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

    __rmul__ = __mul__                       # supports 3 * v

    def __iter__(self):                      # supports unpacking, x, y = v
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

About `NotImplemented`: the binary operation `a + b` calls `a.__add__(b)` first; if that returns `NotImplemented`, Python tries `b.__radd__(a)`; and only when neither works does it raise `TypeError`. So on meeting a type it does not know, **return `NotImplemented` rather than raising**, to give the other side a chance.

### The common special methods at a glance {#常用特殊方法速查}

| To support | Implement |
| --- | --- |
| `repr(x)` / `str(x)` / `format(x, spec)` | `__repr__` / `__str__` / `__format__` |
| `x == y`, `x < y` and so on | `__eq__`, `__lt__` and so on (with `total_ordering`) |
| going in a `set`, being a `dict` key | `__hash__` (agreeing with `__eq__`) |
| `if x:` | `__bool__` or `__len__` |
| `len(x)`, `x[i]`, `x[i] = v`, `del x[i]` | `__len__`, `__getitem__`, `__setitem__`, `__delitem__` |
| `for a in x` | `__iter__` |
| `a in x` | `__contains__` (falling back to iteration otherwise) |
| `x + y`, `y + x`, `x += y` | `__add__`, `__radd__`, `__iadd__` |
| `x(...)` | `__call__` |
| `with x:` | `__enter__`, `__exit__` |

### The sequence protocol: `__len__` and `__getitem__` are enough {#序列协议只要-\_\_len\_\_-和-\_\_getitem\_\_}

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
>>> deck[:3]                         # slicing works automatically
['2♠', '3♠', '4♠']
>>> "Q♥" in deck                     # without __contains__ it iterates automatically
True
>>> next(iter(deck))                 # without __iter__ it iterates through __getitem__ automatically
'2♠'
```

Two methods bought support for `len`, indexing, slicing, iteration, `in`, `reversed` and `random.choice`. That is the power of duck typing and the data model.

## `__slots__`: fixed attributes, less memory {#\_\_slots\_\_固定属性节省内存}

By default every instance has a `__dict__` holding its attributes. Declaring `__slots__` switches the instance to fixed slots:

- noticeably less memory (which matters when creating millions of small objects);
- no attribute outside `__slots__` can be added, so a misspelt attribute name raises outright.

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

A dataclass can generate `__slots__` automatically with `@dataclass(slots=True)`.

## Naming conventions in place of access control {#命名约定代替访问控制}

| Spelling | Meaning |
| --- | --- |
| `name` | public API |
| `_name` | internal, "please do not touch from outside". `from m import *` does not import it |
| `__name` | triggers **name mangling** into `_ClassName__name`, which avoids a subclass overriding it accidentally and is not about "private" |
| `__name__` | a special name reserved by the language; do not invent your own |

!!! interview "Answering in an interview"
    On object orientation: a class attribute is shared by every instance and an instance attribute of the same name hides it, with a mutable class attribute a common bug; `classmethod` makes an alternative constructor (using `cls`, which is friendly to subclasses) while `staticmethod` is just an ordinary function kept in the class; `property` lets you start with a plain attribute and add validation later without changing the interface; `super()` is "the next in the MRO" rather than necessarily the parent, and multiple inheritance needs every level to call it; a binary operation meeting an operand it does not know returns `NotImplemented` so Python can try the other side's reflected method; `__repr__` is for developers and is the one to implement at minimum; `__slots__` saves memory and forbids adding attributes dynamically.

## Exercises {#练习}

**1. A validated interval class.** Implement `Range(lo, hi)` for a closed interval: raise `ValueError` at construction when `lo > hi`; support `x in r`, `len(r)` (the number of integers in it), `r1 & r2` (the intersection, returning `None` when they do not overlap), a friendly `repr` and equality.

??? success "Answer"
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

**2. An alternative constructor and inheritance.** Write a subclass `LabTemperature` of the `Temperature` above with an extra `sensor` attribute (defaulting to `"unknown"`). Verify that `LabTemperature.from_fahrenheit(212)` returns a `LabTemperature`. What would happen if the parent's `from_fahrenheit` were written as `return Temperature(...)`?

??? success "Answer"
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

    Hard-coding `Temperature(...)` would give a subclass's call a parent instance, losing the `sensor` attribute. That is exactly why `cls` is used rather than the class's name. Note that the subclass's `__init__` has to give its new parameter a default, or `cls(...)` with one argument fails.

**3. Cooperative initialization.** Write three classes: `Base` taking `name`, `Leveled(Base)` taking `level` (defaulting to 1) and `Colored(Base)` taking `color` (defaulting to `"black"`), then define `class Full(Leveled, Colored)`. `Full("x", level=2, color="red")` has to give every class its own argument: each `__init__` takes only the keyword arguments it cares about and passes the rest down through `super().__init__(**kwargs)`.

??? success "Answer"
    ```python
    class Base:
        def __init__(self, name, **kwargs):
            super().__init__(**kwargs)       # finally reaches object.__init__, where kwargs should be empty
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

    The chain of calls follows the MRO: `Full → Leveled → Colored → Base → object`, with each level taking its own argument away. If somebody passes a misspelt argument, it raises at `object.__init__` in the end rather than being quietly ignored.

## Summary {#小结}

- [x] Instance data is created in `__init__`; a mutable class attribute is shared by every instance.
- [x] `classmethod` makes an alternative constructor, and `cls` keeps it friendly to subclasses.
- [x] Start with a plain attribute and switch to a `property` when needed, leaving the interface unchanged.
- [x] `super()` is "the next in the MRO", and multiple inheritance needs every level to call it.
- [x] Implement the special methods to fit an object into the language; return `NotImplemented` for an operand you do not know; implement `__repr__` at minimum.
