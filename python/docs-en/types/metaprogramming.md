# Metaprogramming

<p class="lead">An ORM's field declarations, a web framework's route registration, a dataclass generating methods: behind this "magic" are a few rather plain mechanisms, the attribute lookup rules, descriptors, <code>__init_subclass__</code> and metaclasses. Understanding them lets you read a framework's source, and tells you when not to use them.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What is the difference between `__getattr__` and `__getattribute__`?
    2. Why does `obj.method` give a method with `self` already bound?
    3. What is a descriptor's `__set_name__` for?
    4. Without a metaclass, how do you have a base class register all its subclasses automatically?
    5. When is a metaclass genuinely needed?

??? success "Answers (try it yourself first, then expand)"
    1. `__getattribute__` is called on every attribute access; `__getattr__` is called only when the normal lookup fails, which suits proxies and dynamic attributes.
    2. A function is a non-data descriptor: accessed through an instance, its `__get__` returns a bound method object with the instance bound as the first argument.
    3. It is called when the class is created and tells the descriptor which attribute name it was assigned to, so the descriptor can store and fetch data in the instance's dict and produce error messages without the name being repeated at construction.
    4. Define `__init_subclass__` in the base class: it is called once for every subclass defined, and it can record the subclass in a registry.
    5. Very rarely: when the class's own creation or behaviour has to change (the arithmetic on class objects, controlling the class's namespace) and neither a class decorator nor `__init_subclass__` can do it. It is the last resort.

## The full order of attribute lookup {#属性查找的完整顺序}

![Figure: the lookup order of obj.attr - data descriptors, the instance dict, non-data descriptors, __getattr__](../assets/figures/attribute-lookup.svg){.aig-svg}

Executing `obj.attr`, Python looks roughly in this order (implemented by `object.__getattribute__`):

1. Look for `attr` in `type(obj)`'s MRO. If what it finds is a **data descriptor** (defining `__set__` or `__delete__`), call its `__get__` and return.
2. Look in `obj.__dict__` and return what is found.
3. If step 1 found a **non-data descriptor** (only `__get__`, as a function has), call its `__get__`; if it was an ordinary class attribute, return it directly.
4. Nothing found: if the class defines `__getattr__`, call it; otherwise raise `AttributeError`.

This order explains a great deal: a `property` is a data descriptor, so a value of the same name in the instance dict does not hide it; a function is a non-data descriptor, so a method can be overridden by an attribute of the same name on the instance.

## `__getattr__`: the fallback for attribute access {#\_\_getattr\_\_兜底的属性访问}

`__getattr__` is called **only when the normal lookup fails**, which suits proxies, lazy loading and dynamic attributes:

```python
class Proxy:
    def __init__(self, target):
        self._target = target
        self.calls = []

    def __getattr__(self, name):            # only an attribute we cannot find ourselves reaches this
        attr = getattr(self._target, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                self.calls.append(name)
                return attr(*args, **kwargs)
            return wrapper
        return attr

p = Proxy([3, 1, 2])
p.append(0)
p.sort()
assert p._target == [0, 1, 2, 3]
assert p.calls == ["append", "sort"]
```

`__getattribute__` intercepts **every** attribute access including `self._target`, makes infinite recursion very easy to write, and is almost never needed.

Correspondingly, `__setattr__` intercepts every attribute assignment. The implementation has to call `super().__setattr__(name, value)` to do the real write, or it recurses forever:

```python
class ReadOnlyAfterInit:
    def __init__(self, **fields):
        for k, v in fields.items():
            super().__setattr__(k, v)
        super().__setattr__("_frozen", True)

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError(f"cannot set {name!r}: object is read-only")
        super().__setattr__(name, value)

cfg = ReadOnlyAfterInit(host="db", port=5432)
try:
    cfg.port = 1
except AttributeError as e:
    assert "read-only" in str(e)
```

## Descriptors {#描述符}

A descriptor is an object implementing any of `__get__`, `__set__` and `__delete__`, and **as a class attribute** it takes over access to that attribute. `property`, `classmethod`, `staticmethod`, `functools.cached_property` and ordinary methods are all descriptors.

### Writing one: a validated field {#自己写一个带校验的字段}

```python
class Positive:
    def __set_name__(self, owner, name):          # called automatically when the class is created, telling the descriptor its name
        self.name = name
        self.storage = f"_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:                          # accessed through the class (Product.price), it returns the descriptor itself
            return self
        return getattr(obj, self.storage)

    def __set__(self, obj, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{self.name} must be a positive number, got {value!r}")
        setattr(obj, self.storage, value)

class Product:
    price = Positive()
    weight = Positive()

    def __init__(self, name, price, weight):
        self.name = name
        self.price = price                       # triggers Positive.__set__
        self.weight = weight

p = Product("pen", 2.5, 10)
assert p.price == 2.5
try:
    p.weight = -1
except ValueError as e:
    assert str(e) == "weight must be a positive number, got -1"
```

The same validation through `property` means writing a getter and a setter for every field; a descriptor is written once and reused everywhere. Django's model fields, SQLAlchemy's columns and part of pydantic are all built on this mechanism.

### How a method is bound {#方法是怎么绑定的}

A function object has a `__get__`. Accessed through an instance, `__get__` returns a **method object** with the instance "bound" as the first argument:

```pycon
>>> class Greeter:
...     def hello(self, name):
...         return f"hello {name}"
...
>>> g = Greeter()
>>> Greeter.__dict__["hello"]                  # the class dict holds an ordinary function
<function Greeter.hello at 0x...>
>>> g.hello                                    # accessed through the instance, triggering __get__
<bound method Greeter.hello of <__main__.Greeter object at 0x...>>
>>> Greeter.__dict__["hello"].__get__(g)("py")  # doing the same thing by hand
'hello py'
```

That is where `self` comes from, and the reason [the decorators chapter](../core/decorators.md#用类实现装饰器) said "a class decorator on a method loses `self`": a class instance has no `__get__` by default.

## `__init_subclass__`: a hook when a subclass is created {#\_\_init\_subclass\_\_子类创建时的钩子}

Whenever a subclass is defined, the parent's `__init_subclass__` is called. Using it for **automatic registration** is far simpler than a metaclass:

```python
class Command:
    registry: dict[str, type["Command"]] = {}

    def __init_subclass__(cls, *, name: str, **kwargs):
        super().__init_subclass__(**kwargs)
        Command.registry[name] = cls

    def run(self, arg: str) -> str:
        raise NotImplementedError

class Greet(Command, name="greet"):
    def run(self, arg):
        return f"hello {arg}"

class Reverse(Command, name="rev"):
    def run(self, arg):
        return arg[::-1]

def execute(line: str) -> str:
    name, _, arg = line.partition(" ")
    return Command.registry[name]().run(arg)

assert execute("greet world") == "hello world"
assert execute("rev abc") == "cba"
assert set(Command.registry) == {"greet", "rev"}
```

The keyword argument `name="greet"` in the class definition goes to `__init_subclass__`. A plugin system, registering serialization formats and dispatching message types can all follow this pattern.

## Class decorators {#类装饰器}

To modify a class after it is created (adding methods, wrapping attributes, registering it), a class decorator is usually the most direct choice. `@dataclass` is one:

```python
import inspect

def auto_repr(cls):
    fields = list(inspect.signature(cls.__init__).parameters)[1:]     # dropping self

    def __repr__(self):
        args = ", ".join(f"{f}={getattr(self, f)!r}" for f in fields)
        return f"{cls.__name__}({args})"

    cls.__repr__ = __repr__
    return cls

@auto_repr
class Server:
    def __init__(self, host, port=80):
        self.host, self.port = host, port

assert repr(Server("db", 5432)) == "Server(host='db', port=5432)"
```

## Metaclasses {#元类}

A class is an object too, and the "class" that creates classes is the metaclass, `type` by default. A `class` statement is essentially a call to `type(name, bases, namespace)`:

```pycon
>>> Point = type("Point", (), {"x": 0, "describe": lambda self: f"x={self.x}"})
>>> Point().describe()
'x=0'
>>> type(Point), type(type)
(<class 'type'>, <class 'type'>)
```

A custom metaclass (inheriting `type`) controls the class's creation:

```python
class UpperAttrs(type):
    def __new__(mcls, name, bases, namespace):
        upper = {
            (k.upper() if not k.startswith("__") else k): v
            for k, v in namespace.items()
        }
        return super().__new__(mcls, name, bases, upper)

class Settings(metaclass=UpperAttrs):
    debug = True
    timeout = 30

assert Settings.DEBUG is True and not hasattr(Settings, "debug")
```

!!! warning "You almost certainly do not need a metaclass"
    "Metaclasses are deeper magic than 99% of users should ever worry about. If you wonder whether you need them, you don't." — Tim Peters

    A metaclass spreads to every subclass, and two base classes with different metaclasses cannot be inherited together. Consider these in order:

    1. an ordinary function or a decorator
    2. a class decorator
    3. `__init_subclass__`, `__set_name__` (a descriptor)
    4. a metaclass (only when the class's creation itself has to be controlled, as an ORM rewriting the namespace before the class is created)

## Introspection tools {#内省工具}

```pycon
>>> import inspect
>>> class Demo:
...     x = 1
...     def method(self): ...
...     @property
...     def prop(self): return 2
...
>>> [name for name, _ in inspect.getmembers(Demo, inspect.isfunction)]
['method']
>>> getattr(Demo(), "prop"), hasattr(Demo, "nope")
(2, False)
>>> vars(Demo)["x"]
1
```

| Function | Use |
| --- | --- |
| `getattr(obj, name, default)` / `setattr` / `delattr` / `hasattr` | access an attribute dynamically by a string |
| `vars(obj)` | returns `obj.__dict__` |
| `dir(obj)` | lists every reachable attribute name |
| `inspect.signature(func)` | a function's signature |
| `inspect.getmembers(obj, predicate)` | the members matching a condition |
| `inspect.getsource(obj)` | the source code |

!!! interview "Answering in an interview"
    On metaprogramming: the attribute lookup order is data descriptors → the instance dict → non-data descriptors and class attributes → `__getattr__`; `__getattribute__` is called on every access while `__getattr__` only falls back when nothing is found (which suits a proxy, as an inference framework's platform class forwarding the device API to `torch.cuda` / `torch.npu`); a method binds `self` because a function is a non-data descriptor whose `__get__` returns a bound method; `property` and ORM fields are descriptors, and `__set_name__` tells one its own name. Registering subclasses automatically uses `__init_subclass__`, modifying a class uses a class decorator, and a metaclass is the last resort.

## Exercises {#练习}

**1. A typed field descriptor.** Write a descriptor `Typed(type_)` raising `TypeError` on assignment of the wrong type. Then write `NonEmptyStr`, inheriting `Typed(str)` and additionally requiring a non-empty value. Define a `User` class with them.

??? success "Answer"
    ```python
    class Typed:
        def __init__(self, type_):
            self.type_ = type_

        def __set_name__(self, owner, name):
            self.name = name

        def __get__(self, obj, objtype=None):
            if obj is None:
                return self
            return obj.__dict__[self.name]

        def __set__(self, obj, value):
            self.validate(value)
            obj.__dict__[self.name] = value         # write straight into the instance dict, the same name being fine (a data descriptor wins)

        def validate(self, value):
            if not isinstance(value, self.type_):
                raise TypeError(f"{self.name} must be {self.type_.__name__}, got {type(value).__name__}")

    class NonEmptyStr(Typed):
        def __init__(self):
            super().__init__(str)

        def validate(self, value):
            super().validate(value)
            if not value.strip():
                raise ValueError(f"{self.name} must not be empty")

    class User:
        name = NonEmptyStr()
        age = Typed(int)

        def __init__(self, name, age):
            self.name, self.age = name, age

    u = User("amy", 30)
    assert (u.name, u.age) == ("amy", 30)
    for bad in [("", 1), ("bob", "30")]:
        try:
            User(*bad)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(bad)
    ```

    Because `Typed` defines `__set__` it is a data descriptor, which takes priority over the instance dict, so the value can be stored under the same key in the instance dict with no underscore prefix.

**2. A lazily loaded attribute.** Without `functools.cached_property`, write a non-data descriptor `lazy`: the first access calls the decorated method and writes the result into the instance's dict; later accesses hit the instance dict directly and never reach the descriptor.

??? success "Answer"
    ```python
    class lazy:
        def __init__(self, func):
            self.func = func
            self.__doc__ = func.__doc__

        def __set_name__(self, owner, name):
            self.name = name

        def __get__(self, obj, objtype=None):
            if obj is None:
                return self
            value = self.func(obj)
            obj.__dict__[self.name] = value         # a non-data descriptor: next time the instance dict wins
            return value

    class Dataset:
        loads = 0

        @lazy
        def rows(self):
            Dataset.loads += 1
            return [1, 2, 3]

    d = Dataset()
    assert d.rows == [1, 2, 3] and d.rows == [1, 2, 3]
    assert Dataset.loads == 1
    ```

    The point is implementing only `__get__`: by the lookup order, the instance dict takes priority over a non-data descriptor, so once the cache is written the descriptor is never called again. That is exactly how `cached_property` works.

## Summary {#小结}

- [x] The attribute lookup order: data descriptors → the instance dict → non-data descriptors and class attributes → `__getattr__`.
- [x] `__getattr__` is called only when nothing is found and suits proxies and dynamic attributes; `__setattr__` has to call `super()`.
- [x] Descriptors are the mechanism behind `property`, method binding and ORM fields; `__set_name__` tells one its own name.
- [x] Registering subclasses automatically uses `__init_subclass__`; modifying a class uses a class decorator.
- [x] A metaclass is the last resort.
