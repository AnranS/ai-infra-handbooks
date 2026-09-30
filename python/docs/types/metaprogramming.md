# 元编程

<p class="lead">ORM 的字段声明、Web 框架的路由注册、dataclass 自动生成方法，这些"魔法"背后是几个相当朴素的机制：属性查找规则、描述符、<code>__init_subclass__</code> 和元类。理解它们，你就能读懂框架源码，也知道什么时候不该用它们。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `__getattr__` 和 `__getattribute__` 有什么区别？
    2. 为什么 `obj.method` 拿到的是一个绑定了 `self` 的方法？
    3. 描述符的 `__set_name__` 有什么用？
    4. 不用元类，怎么让一个基类自动登记它的所有子类？
    5. 什么时候真的需要元类？

??? success "自测参考答案（先自己答，再展开对照）"
    1. `__getattribute__` 在每次访问属性时都会被调用；`__getattr__` 只在正常的查找找不到属性时才调用，适合做代理、动态属性。
    2. 函数是一个非数据描述符：通过实例访问时，它的 `__get__` 返回一个把实例绑定为第一个参数的绑定方法对象。
    3. 在类创建时被调用，告诉描述符它被赋给了哪个属性名，描述符就可以据此在实例字典里存取数据、生成错误信息，而不用在构造时重复写一遍名字。
    4. 在基类里定义 `__init_subclass__`：每定义一个子类，它都会被调用一次，可以在里面把子类登记到注册表。
    5. 极少：需要改变类本身的创建过程或行为（比如类对象的运算、控制类的命名空间），而类装饰器和 `__init_subclass__` 都做不到的时候。它是最后的手段。

## 属性查找的完整顺序

执行 `obj.attr` 时，Python 大致按这个顺序查找（由 `object.__getattribute__` 实现）：

1. 在 `type(obj)` 的 MRO 中找 `attr`。如果找到的是**数据描述符**（定义了 `__set__` 或 `__delete__`），调用它的 `__get__` 并返回。
2. 在 `obj.__dict__` 里找，找到就返回。
3. 如果第 1 步找到的是**非数据描述符**（只有 `__get__`，比如函数），调用它的 `__get__`；如果是普通类属性，直接返回。
4. 都没有：如果类定义了 `__getattr__`，调用它；否则抛 `AttributeError`。

这个顺序解释了很多现象：`property` 是数据描述符，所以实例字典里同名的值不会遮住它；函数是非数据描述符，所以可以在实例上用同名属性覆盖方法。

## `__getattr__`：兜底的属性访问

`__getattr__` **只在正常查找失败时**才被调用，适合做代理、延迟加载、动态属性：

```python
class Proxy:
    def __init__(self, target):
        self._target = target
        self.calls = []

    def __getattr__(self, name):            # 只有自己找不到的属性才会走到这里
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

`__getattribute__` 则会拦截**所有**属性访问，包括 `self._target`，很容易写出无限递归，几乎用不到。

相应地，`__setattr__` 拦截所有属性赋值。实现时要调用 `super().__setattr__(name, value)` 真正写入，否则会无限递归：

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

## 描述符

描述符是实现了 `__get__`、`__set__`、`__delete__` 中任意一个的对象，**作为类属性**存在时，会接管对这个属性的访问。`property`、`classmethod`、`staticmethod`、`functools.cached_property`，还有普通的方法，全都是描述符。

### 自己写一个：带校验的字段

```python
class Positive:
    def __set_name__(self, owner, name):          # 类创建时自动调用，告诉描述符它叫什么
        self.name = name
        self.storage = f"_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:                          # 通过类访问时（Product.price），返回描述符自身
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
        self.price = price                       # 触发 Positive.__set__
        self.weight = weight

p = Product("pen", 2.5, 10)
assert p.price == 2.5
try:
    p.weight = -1
except ValueError as e:
    assert str(e) == "weight must be a positive number, got -1"
```

同样的校验逻辑用 `property` 写，每个字段都要写一遍 getter 和 setter；描述符写一次，到处复用。Django 的模型字段、SQLAlchemy 的列、pydantic 的一部分功能，都建立在这个机制上。

### 方法是怎么绑定的

函数对象有 `__get__` 方法。通过实例访问时，`__get__` 返回一个把实例"绑定"为第一个参数的**方法对象**：

```pycon
>>> class Greeter:
...     def hello(self, name):
...         return f"hello {name}"
...
>>> g = Greeter()
>>> Greeter.__dict__["hello"]                  # 类字典里存的是普通函数
<function Greeter.hello at 0x...>
>>> g.hello                                    # 通过实例访问，触发 __get__
<bound method Greeter.hello of <__main__.Greeter object at 0x...>>
>>> Greeter.__dict__["hello"].__get__(g)("py")  # 手动做同样的事
'hello py'
```

这就是 `self` 的由来，也是[装饰器一章](../core/decorators.md#用类实现装饰器)里提到"类装饰器用在方法上会丢失 `self`"的原因：类实例默认没有 `__get__`。

## `__init_subclass__`：子类创建时的钩子

每当定义一个子类，父类的 `__init_subclass__` 就会被调用。用它实现**自动注册**，比元类简单得多：

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

类定义里的关键字参数 `name="greet"` 会传给 `__init_subclass__`。插件系统、序列化格式注册、消息类型分发都可以用这个模式。

## 类装饰器

需要在类创建后修改它（添加方法、包装属性、登记），类装饰器通常是最直接的选择。`@dataclass` 就是一个类装饰器：

```python
import inspect

def auto_repr(cls):
    fields = list(inspect.signature(cls.__init__).parameters)[1:]     # 去掉 self

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

## 元类

类也是对象，创建类的"类"就是元类，默认是 `type`。`class` 语句本质上是调用 `type(name, bases, namespace)`：

```pycon
>>> Point = type("Point", (), {"x": 0, "describe": lambda self: f"x={self.x}"})
>>> Point().describe()
'x=0'
>>> type(Point), type(type)
(<class 'type'>, <class 'type'>)
```

自定义元类（继承 `type`）可以控制类的创建过程：

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

!!! warning "你大概率不需要元类"
    "元类是 99% 的用户都不必操心的深奥魔法。如果你在犹豫是否需要它，那你就不需要。"——Tim Peters

    元类会传染给所有子类，两个带不同元类的基类还无法一起继承。按优先顺序考虑：

    1. 普通函数、装饰器
    2. 类装饰器
    3. `__init_subclass__`、`__set_name__`（描述符）
    4. 元类（只在需要控制类的创建本身时，比如 ORM 需要在类创建前改写命名空间）

## 内省工具

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

| 函数 | 用途 |
| --- | --- |
| `getattr(obj, name, default)` / `setattr` / `delattr` / `hasattr` | 用字符串动态访问属性 |
| `vars(obj)` | 返回 `obj.__dict__` |
| `dir(obj)` | 列出所有可访问的属性名 |
| `inspect.signature(func)` | 函数签名 |
| `inspect.getmembers(obj, predicate)` | 按条件列出成员 |
| `inspect.getsource(obj)` | 源代码 |

!!! interview "面试怎么答"
    元编程题：属性查找顺序是数据描述符 → 实例字典 → 非数据描述符和类属性 → `__getattr__`；`__getattribute__` 每次访问都调用，`__getattr__` 只在找不到时兜底（适合代理，比如推理框架的平台类把设备 API 转发给 `torch.cuda` / `torch.npu`）；方法之所以能绑定 `self`，是因为函数是非数据描述符，`__get__` 返回绑定方法；`property`、ORM 字段都是描述符，`__set_name__` 让它知道自己的名字。自动登记子类用 `__init_subclass__`，修改类用类装饰器，元类是最后的手段。

## 练习

**1. 带类型的字段描述符。** 写描述符 `Typed(type_)`，赋值时如果类型不对抛 `TypeError`。再写 `NonEmptyStr`，继承 `Typed(str)` 并额外要求非空。用它们定义一个 `User` 类。

??? success "参考答案"
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
            obj.__dict__[self.name] = value         # 直接写实例字典，名字相同也没关系（数据描述符优先）

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

    因为 `Typed` 定义了 `__set__`，它是数据描述符，优先级高于实例字典，所以可以把值直接存在实例字典的同名键里，不用加下划线前缀。

**2. 懒加载属性。** 不用 `functools.cached_property`，自己写一个非数据描述符 `lazy`：第一次访问时调用被装饰的方法，把结果写进实例字典；之后的访问直接命中实例字典，不再经过描述符。

??? success "参考答案"
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
            obj.__dict__[self.name] = value         # 非数据描述符：下次实例字典优先
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

    关键在于只实现 `__get__`：根据属性查找顺序，实例字典的优先级高于非数据描述符，所以缓存写进去之后，描述符就再也不会被调用了。这正是 `cached_property` 的实现原理。

## 小结

- [x] 属性查找顺序：数据描述符 → 实例字典 → 非数据描述符/类属性 → `__getattr__`。
- [x] `__getattr__` 只在找不到时调用，适合代理和动态属性；`__setattr__` 里要调用 `super()`。
- [x] 描述符是 `property`、方法绑定、ORM 字段背后的机制；`__set_name__` 让它知道自己的名字。
- [x] 自动注册子类用 `__init_subclass__`；修改类用类装饰器。
- [x] 元类是最后的手段。
