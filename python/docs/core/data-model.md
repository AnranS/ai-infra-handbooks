# 对象模型：名字、引用与可变性

<p class="lead">Python 里绝大多数"诡异 bug"都来自对象模型没搞清楚：改了一个变量，另一个也跟着变；默认参数越用越长；拷贝了还是互相影响。这一章把地基打牢。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `a = [1, 2]; b = a; a += [3]` 之后 `b` 是什么？如果把 `+=` 换成 `a = a + [3]` 呢？
    2. `is` 和 `==` 有什么区别？什么时候必须用 `is`？
    3. 为什么 `t = (1, [2]); t[1] += [3]` 会报错，但 `t` 还是变了？
    4. 浅拷贝和深拷贝的区别？`list(x)`、`x[:]`、`x.copy()` 是哪一种？
    5. 一个类只定义了 `__eq__`，它的实例还能放进 `set` 吗？为什么？

??? success "自测参考答案（先自己答，再展开对照）"
    1. `a += [3]` 对列表是原地扩展，`a`、`b` 指向同一个列表，`b` 也是 `[1, 2, 3]`；`a = a + [3]` 创建了一个新列表并让 `a` 指向它，`b` 还是 `[1, 2]`。
    2. `==` 比较值（调用 `__eq__`），`is` 比较是不是同一个对象。和单例比较时必须用 `is`：`None`、自定义的哨兵对象、`True` / `False`（在需要区分类型时）。
    3. `t[1] += [3]` 分两步：先对 `t[1]` 这个列表原地 `+=`（成功，列表变了），再把结果赋值回 `t[1]`（元组不允许，报错）。所以报错了，但列表已经被改了。
    4. 浅拷贝只复制最外层的容器，里面的元素还是共享的；深拷贝（`copy.deepcopy`）递归地复制所有嵌套对象。`list(x)`、`x[:]`、`x.copy()` 都是浅拷贝。
    5. 不能：只定义 `__eq__` 时，Python 会把 `__hash__` 设为 `None`，实例变成不可哈希的，否则相等的对象可能有不同的哈希值、破坏集合和字典。要同时定义一致的 `__hash__`。

## 名字是标签，不是盒子

很多语言里，变量像一个盒子，赋值就是把值放进盒子。Python 不是这样：**对象独立存在，变量只是贴在对象上的标签（名字）**。赋值语句做的事是"让这个名字指向那个对象"，不会复制任何东西。

```pycon
>>> a = [1, 2, 3]
>>> b = a            # b 和 a 指向同一个列表
>>> b.append(4)
>>> a
[1, 2, 3, 4]
>>> a is b
True
```

想要一个独立的副本，必须显式创建新对象：

```pycon
>>> c = list(a)      # 新建一个列表，内容相同
>>> c == a, c is a
(True, False)
```

每个对象都有三样东西：

| 属性 | 怎么看 | 会不会变 |
| --- | --- | --- |
| 身份（identity） | `id(obj)`，`is` 比较的就是它 | 对象存活期间不变 |
| 类型（type） | `type(obj)` | 不变 |
| 值（value） | `==` 比较的就是它 | 可变对象会变，不可变对象不变 |

## `is` 与 `==`

- `==` 比较**值**，会调用 `__eq__`，可以被类重新定义。
- `is` 比较**身份**，也就是"是不是同一个对象"，不能被重载，速度也更快。

只有在比较**单例**时才用 `is`：`None`、`True`、`False`，以及你自己定义的哨兵对象。

```pycon
>>> x = None
>>> x is None
True
```

!!! warning "不要用 `is` 比较数字和字符串"
    CPython 会缓存 -5 到 256 的小整数和部分字符串，所以 `a is b` 有时恰好是 `True`。这是实现细节，换个数值、换个写法、换个解释器结果就可能变。比较值永远用 `==`。

### 哨兵对象

当 `None` 本身也是合法值时，用一个独一无二的对象当"没有传参"的标记：

```python
_MISSING = object()

def get_config(d, key, default=_MISSING):
    if key in d:
        return d[key]
    if default is _MISSING:
        raise KeyError(key)
    return default

cfg = {"timeout": None}
assert get_config(cfg, "timeout") is None      # None 是合法值
assert get_config(cfg, "retries", 3) == 3
```

## 可变与不可变

| 不可变（immutable） | 可变（mutable） |
| --- | --- |
| `int` `float` `complex` `bool` `str` `bytes` `tuple` `frozenset` `None` | `list` `dict` `set` `bytearray`，以及大多数自定义类的实例 |

"不可变"指的是**对象自身的值不能改**。对不可变对象做"修改"操作，其实是创建了新对象，再让名字指向它：

```pycon
>>> s = "py"
>>> before = id(s)
>>> s += "thon"
>>> id(s) == before       # 字符串不可变，+= 生成了新对象
False
>>> nums = [1]
>>> before = id(nums)
>>> nums += [2]
>>> id(nums) == before    # 列表可变，+= 原地修改
True
```

这就是自测第 1 题的答案：`a += [3]` 对列表是原地修改（调用 `__iadd__`），`b` 也能看到；而 `a = a + [3]` 创建了新列表并让 `a` 指向它，`b` 仍然指向旧列表。

### 元组里放可变对象

元组不可变，指的是它**保存的引用**不能换，但被引用的对象自己可以变：

```pycon
>>> t = (1, [2, 3])
>>> t[1].append(4)       # 没问题：改的是列表，不是元组
>>> t
(1, [2, 3, 4])
>>> t[1] += [5]
Traceback (most recent call last):
  ...
TypeError: 'tuple' object does not support item assignment
>>> t
(1, [2, 3, 4, 5])
```

`t[1] += [5]` 分两步执行：先对列表做原地的 `+=`（成功了），再执行 `t[1] = 结果`（失败了，因为元组不能赋值）。所以报了错，但列表已经改了。这也说明：**含有可变对象的元组不可哈希**，不能当字典的键。

## 函数参数：传的是对象引用

Python 的参数传递既不是"传值"也不是"传引用"，而是**传对象引用**（call by sharing）：形参是一个新名字，指向调用方传进来的同一个对象。

```python
def add_item(items, x):
    items.append(x)      # 修改传进来的对象：调用方能看到

def reset(items):
    items = []           # 只是让局部名字指向新对象：调用方不受影响

data = [1]
add_item(data, 2)
reset(data)
print(data)  # [1, 2]
```

规律很简单：**原地修改对象，调用方可见；给形参重新赋值，调用方不可见。**

## 拷贝：浅拷贝与深拷贝

- **浅拷贝**：创建新的外层容器，但里面的元素还是原来那些对象。`list(x)`、`x[:]`、`x.copy()`、`dict(d)`、`copy.copy(x)` 都是浅拷贝。
- **深拷贝**：递归复制所有层级。用 `copy.deepcopy(x)`。

```pycon
>>> import copy
>>> cfg = {"name": "svc", "ports": [80, 443]}
>>> shallow = copy.copy(cfg)
>>> deep = copy.deepcopy(cfg)
>>> cfg["ports"].append(8080)
>>> shallow["ports"]       # 和 cfg 共享同一个列表
[80, 443, 8080]
>>> deep["ports"]          # 完全独立
[80, 443]
```

### 经典陷阱：用 `*` 创建二维列表

```pycon
>>> grid = [[0] * 3] * 3      # 外层 3 个元素是同一个内层列表
>>> grid[0][0] = 1
>>> grid
[[1, 0, 0], [1, 0, 0], [1, 0, 0]]
>>> grid = [[0] * 3 for _ in range(3)]   # 每次循环新建一个内层列表
>>> grid[0][0] = 1
>>> grid
[[1, 0, 0], [0, 0, 0], [0, 0, 0]]
```

`[0] * 3` 没问题，因为 `0` 是不可变的；`[[0] * 3] * 3` 复制的是**同一个列表的引用**。

!!! tip "深拷贝不是万能药"
    深拷贝慢，还可能复制到你不想复制的东西（比如数据库连接、锁）。更好的做法通常是：尽量用不可变数据（元组、`frozenset`、`frozen=True` 的 dataclass），或者在需要时明确地构造新对象。

## 真值测试

在 `if`、`while`、`and`、`or` 里，任何对象都可以当布尔值用。以下这些是"假"的，其余都是"真"的：

```pycon
>>> [bool(x) for x in (0, 0.0, "", [], {}, set(), None, "0", [0])]
[False, False, False, False, False, False, False, True, True]
```

自定义类可以通过 `__bool__` 或 `__len__` 控制真假。地道的写法是直接用真值判断：

```py
if items:            # 而不是 if len(items) > 0:
    ...
if not name:         # 而不是 if name == "":
    ...
if result is None:   # 但要区分 None 和 0/空串时，必须显式比较
    ...
```

`and` 和 `or` 返回的是**操作数本身**，不一定是 `True`/`False`：

```pycon
>>> "" or "default"
'default'
>>> 0 or None or []
[]
>>> [1] and "ok"
'ok'
```

!!! warning "`x or default` 会吞掉合法的假值"
    `timeout = user_timeout or 30` 在用户传 `0` 时也会变成 30。当 `0`、空字符串是合法值时，写成 `30 if user_timeout is None else user_timeout`。

## 哈希与相等

能当字典键、能放进集合的对象必须是**可哈希的**（hashable）：

1. 有 `__hash__`，并且在对象生命周期内哈希值不变；
2. 有 `__eq__`；
3. 如果 `a == b`，那么 `hash(a) == hash(b)`。

不可变的内置类型都可哈希（元组要求元素也可哈希）；`list`、`dict`、`set` 不可哈希。

```pycon
>>> hash((1, 2)) == hash((1, 2))
True
>>> hash([1, 2])
Traceback (most recent call last):
  ...
TypeError: unhashable type: 'list'
```

**只定义 `__eq__` 的类，会自动把 `__hash__` 设为 `None`**，实例就不能放进集合了。这是故意的：你改变了"相等"的含义，Python 不知道哈希该怎么配合。

```pycon
>>> class Point:
...     def __init__(self, x, y):
...         self.x, self.y = x, y
...     def __eq__(self, other):
...         return isinstance(other, Point) and (self.x, self.y) == (other.x, other.y)
...
>>> Point(1, 2) == Point(1, 2)
True
>>> Point.__hash__ is None
True
```

要让它可哈希，同时定义基于同样字段的 `__hash__`，并保证这些字段不会被修改：

```python
class Point:
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return (self.x, self.y) == (other.x, other.y)

    def __hash__(self):
        return hash((self.x, self.y))

assert len({Point(1, 2), Point(1, 2), Point(3, 4)}) == 2
```

实际项目里，更省事的做法是 `@dataclass(frozen=True)`，它会帮你生成 `__eq__` 和 `__hash__`，见[数据建模](data-classes.md)。

## 一切皆对象

函数、类、模块本身都是对象：可以赋值给变量、放进容器、作为参数传递、带属性。

```pycon
>>> def greet(name):
...     """Say hi."""
...     return f"hi {name}"
...
>>> greet.__name__, greet.__doc__
('greet', 'Say hi.')
>>> handlers = {"greet": greet}
>>> handlers["greet"]("py")
'hi py'
>>> type(greet), type(int), type(type)
(<class 'function'>, <class 'type'>, <class 'type'>)
```

"函数是一等对象"是装饰器、回调、策略模式的基础，后面几章会反复用到。

## 对象什么时候被回收

CPython 主要靠**引用计数**：一个对象没有任何名字或容器引用它时，立刻被回收。循环引用（A 引用 B，B 引用 A）由周期性运行的**循环垃圾回收器**处理。

`del x` 删除的是**名字**，不是对象。对象只有在引用计数归零时才会被回收。

如果想引用一个对象、又不想阻止它被回收（比如做缓存），用 `weakref`：

```pycon
>>> import weakref
>>> class Node:
...     pass
...
>>> n = Node()
>>> r = weakref.ref(n)
>>> r() is n
True
>>> del n
>>> r() is None       # 对象已被回收
True
```

## 练习

**1. 预测输出。** 先想，再运行。

```py
def f(x, lst=[]):
    lst.append(x)
    return lst

print(f(1), f(2))
```

??? success "参考答案"
    输出 `[1, 2] [1, 2]`，不是 `[1] [1, 2]`。

    原因有两层：

    - 默认参数在**函数定义时**只求值一次，所有调用共享同一个列表。
    - `print` 先把两个参数都求值完再打印。两次调用返回的是**同一个列表对象**，打印时它已经是 `[1, 2]` 了。

    正确写法是用 `None` 当默认值：

    ```python
    def f(x, lst=None):
        if lst is None:
            lst = []
        lst.append(x)
        return lst

    assert (f(1), f(2)) == ([1], [2])
    ```

**2. 写一个 `freeze(obj)`**，把嵌套的 `list`、`dict`、`set` 转成可哈希的等价结构（`tuple`、排好序的键值对元组、`frozenset`），这样就能把任意 JSON 风格的数据当缓存的键。要求 `freeze({"b": [1, 2], "a": {3}})` 和 `freeze({"a": {3}, "b": [1, 2]})` 相等。

??? success "参考答案"
    ```python
    def freeze(obj):
        if isinstance(obj, dict):
            return tuple(sorted((k, freeze(v)) for k, v in obj.items()))
        if isinstance(obj, (list, tuple)):
            return tuple(freeze(x) for x in obj)
        if isinstance(obj, (set, frozenset)):
            return frozenset(freeze(x) for x in obj)
        return obj

    a = freeze({"b": [1, 2], "a": {3}})
    b = freeze({"a": {3}, "b": [1, 2]})
    assert a == b and hash(a) == hash(b)
    cache = {a: "cached result"}
    assert cache[b] == "cached result"
    ```

    这里对字典按键排序，是为了让键的插入顺序不影响结果。前提是键之间可以比较大小；如果键的类型混杂，可以改成 `frozenset` 存键值对。

**3. 为什么推荐写 `x is None` 而不是 `x == None`？** 写一个类，让它的实例 `== None` 为 `True`，但 `is None` 为 `False`。

??? success "参考答案"
    `==` 会调用 `__eq__`，而 `__eq__` 可以被任意重载；`is` 比较身份，无法被欺骗，而且更快。

    ```python
    class Weird:
        def __eq__(self, other):
            return True

    w = Weird()
    assert (w == None) is True
    assert (w is None) is False
    ```

    NumPy 数组、SQLAlchemy 的列对象等都重载了 `==`，对它们写 `== None` 得到的甚至不是布尔值。

## 小结

- [x] 变量是名字，赋值只是绑定，从不复制对象。
- [x] 比较值用 `==`，比较单例（`None`、哨兵）用 `is`。
- [x] 可变对象的原地修改对所有引用者可见；形参重新赋值对调用方不可见。
- [x] 浅拷贝只复制外层；嵌套的可变数据需要深拷贝或不可变设计。
- [x] 可哈希需要 `__hash__` 和 `__eq__` 一致；只定义 `__eq__` 会让实例不可哈希。
