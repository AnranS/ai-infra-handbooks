# The object model: names, references and mutability

<p class="lead">Nearly every "weird bug" in Python comes from the object model not being clear: changing one variable changes another; a default argument grows longer with every call; a copy still affects the original. This chapter lays that groundwork.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. After `a = [1, 2]; b = a; a += [3]`, what is `b`? And with `a = a + [3]` instead of `+=`?
    2. How do `is` and `==` differ? When must you use `is`?
    3. Why does `t = (1, [2]); t[1] += [3]` raise, and yet `t` changes anyway?
    4. What is the difference between a shallow and a deep copy? Which are `list(x)`, `x[:]` and `x.copy()`?
    5. A class defines only `__eq__`. Can its instances go in a `set`? Why?

??? success "Answers (try it yourself first, then expand)"
    1. `a += [3]` extends the list in place and `a` and `b` point at the same list, so `b` is `[1, 2, 3]` too; `a = a + [3]` creates a new list and points `a` at it, leaving `b` as `[1, 2]`.
    2. `==` compares values (calling `__eq__`) and `is` compares whether it is the same object. Comparing against a singleton requires `is`: `None`, your own sentinel objects, and `True` / `False` where the type has to be distinguished.
    3. `t[1] += [3]` is two steps: first the list `t[1]` is extended in place (which succeeds and changes the list), then the result is assigned back to `t[1]` (which a tuple does not allow, hence the error). So it raises, and the list has already changed.
    4. A shallow copy copies only the outermost container and the elements inside are still shared; a deep copy (`copy.deepcopy`) copies every nested object recursively. `list(x)`, `x[:]` and `x.copy()` are all shallow.
    5. No: when only `__eq__` is defined, Python sets `__hash__` to `None` and the instances become unhashable, since otherwise equal objects could have different hashes and break sets and dicts. A consistent `__hash__` has to be defined too.

## A name is a label, not a box {#名字是标签不是盒子}

In many languages a variable is a box and assignment puts a value in it. Python is not like that: **objects exist on their own and a variable is only a label (a name) stuck on one**. What an assignment does is "point this name at that object", and it copies nothing.

![Figure: how names relate to objects](../assets/figures/name-binding.svg){.aig-svg}

```pycon
>>> a = [1, 2, 3]
>>> b = a            # b and a point at the same list
>>> b.append(4)
>>> a
[1, 2, 3, 4]
>>> a is b
True
```

An independent copy has to be created explicitly:

```pycon
>>> c = list(a)      # a new list with the same contents
>>> c == a, c is a
(True, False)
```

Every object has three things:

| Property | How to see it | Does it change |
| --- | --- | --- |
| identity | `id(obj)`, which is what `is` compares | fixed for the object's life |
| type | `type(obj)` | fixed |
| value | what `==` compares | changes for a mutable object, not for an immutable one |

## `is` and `==` {#is-与-}

- `==` compares the **value**, calls `__eq__`, and a class can redefine it.
- `is` compares the **identity**, "is this the same object", cannot be overloaded and is faster.

Use `is` only when comparing against a **singleton**: `None`, `True`, `False`, and sentinel objects of your own.

```pycon
>>> x = None
>>> x is None
True
```

!!! warning "Do not compare numbers and strings with `is`"
    CPython caches the small integers from -5 to 256 and some strings, so `a is b` sometimes happens to be `True`. That is an implementation detail and the result can change with a different value, a different spelling or a different interpreter. Always compare values with `==`.

### Sentinel objects {#哨兵对象}

When `None` is itself a legal value, use a unique object to mark "no argument was passed":

```python
_MISSING = object()

def get_config(d, key, default=_MISSING):
    if key in d:
        return d[key]
    if default is _MISSING:
        raise KeyError(key)
    return default

cfg = {"timeout": None}
assert get_config(cfg, "timeout") is None      # None is a legal value
assert get_config(cfg, "retries", 3) == 3
```

## Mutable and immutable {#可变与不可变}

| immutable | mutable |
| --- | --- |
| `int` `float` `complex` `bool` `str` `bytes` `tuple` `frozenset` `None` | `list` `dict` `set` `bytearray`, and most instances of your own classes |

"Immutable" means **the object's own value cannot change**. A "modifying" operation on an immutable object in fact creates a new object and points the name at it:

```pycon
>>> s = "py"
>>> before = id(s)
>>> s += "thon"
>>> id(s) == before       # a string is immutable, so += made a new object
False
>>> nums = [1]
>>> before = id(nums)
>>> nums += [2]
>>> id(nums) == before    # a list is mutable, so += modifies in place
True
```

Which is the answer to self-test 1: `a += [3]` modifies the list in place (calling `__iadd__`) and `b` sees it too; `a = a + [3]` creates a new list and points `a` at it, while `b` still points at the old one.

### A mutable object inside a tuple {#元组里放可变对象}

A tuple being immutable means **the references it holds** cannot be swapped, while the objects referred to can change themselves:

```pycon
>>> t = (1, [2, 3])
>>> t[1].append(4)       # fine: the list changes, not the tuple
>>> t
(1, [2, 3, 4])
>>> t[1] += [5]
Traceback (most recent call last):
  ...
TypeError: 'tuple' object does not support item assignment
>>> t
(1, [2, 3, 4, 5])
```

`t[1] += [5]` runs in two steps: first the `+=` on the list in place (which succeeds), then `t[1] = the result` (which fails, because a tuple cannot be assigned to). So it raises, and the list has already changed. This also shows that **a tuple containing a mutable object is unhashable** and cannot be a dict key.

## Function parameters: an object reference is passed {#函数参数传的是对象引用}

Python's argument passing is neither "by value" nor "by reference" but **by object reference** (call by sharing): the parameter is a new name pointing at the same object the caller passed in.

```python
def add_item(items, x):
    items.append(x)      # modifying the object passed in: the caller sees it

def reset(items):
    items = []           # only pointing the local name at a new object: the caller is unaffected

data = [1]
add_item(data, 2)
reset(data)
print(data)  # [1, 2]
```

The rule is simple: **modifying the object in place is visible to the caller; rebinding the parameter is not.**

## Copying: shallow and deep {#拷贝浅拷贝与深拷贝}

- **a shallow copy**: a new outer container whose elements are still the original objects. `list(x)`, `x[:]`, `x.copy()`, `dict(d)` and `copy.copy(x)` are all shallow.
- **a deep copy**: every level copied recursively, with `copy.deepcopy(x)`.

```pycon
>>> import copy
>>> cfg = {"name": "svc", "ports": [80, 443]}
>>> shallow = copy.copy(cfg)
>>> deep = copy.deepcopy(cfg)
>>> cfg["ports"].append(8080)
>>> shallow["ports"]       # shares the same list as cfg
[80, 443, 8080]
>>> deep["ports"]          # entirely independent
[80, 443]
```

### The classic trap: building a two-dimensional list with `*` {#经典陷阱用--创建二维列表}

```pycon
>>> grid = [[0] * 3] * 3      # the outer 3 elements are the same inner list
>>> grid[0][0] = 1
>>> grid
[[1, 0, 0], [1, 0, 0], [1, 0, 0]]
>>> grid = [[0] * 3 for _ in range(3)]   # a new inner list on each iteration
>>> grid[0][0] = 1
>>> grid
[[1, 0, 0], [0, 0, 0], [0, 0, 0]]
```

`[0] * 3` is fine because `0` is immutable; `[[0] * 3] * 3` duplicates **a reference to the same list**.

!!! tip "A deep copy is not a cure-all"
    It is slow and may copy things you did not want copied (a database connection, a lock). The better approach is usually to use immutable data where you can (tuples, `frozenset`, a `frozen=True` dataclass), or to construct a new object explicitly where one is needed.

## Truth testing {#真值测试}

In an `if`, a `while`, an `and` or an `or`, any object can serve as a boolean. These are "false" and everything else is "true":

```pycon
>>> [bool(x) for x in (0, 0.0, "", [], {}, set(), None, "0", [0])]
[False, False, False, False, False, False, False, True, True]
```

A class of your own controls its truth through `__bool__` or `__len__`. The idiomatic form tests the truth directly:

```py
if items:            # rather than if len(items) > 0:
    ...
if not name:         # rather than if name == "":
    ...
if result is None:   # but distinguishing None from 0 or an empty string calls for an explicit comparison
    ...
```

`and` and `or` return **an operand itself** and not necessarily `True`/`False`:

```pycon
>>> "" or "default"
'default'
>>> 0 or None or []
[]
>>> [1] and "ok"
'ok'
```

!!! warning "`x or default` swallows a legitimate false value"
    `timeout = user_timeout or 30` becomes 30 when the user passes `0` as well. When `0` or an empty string is a legal value, write `30 if user_timeout is None else user_timeout`.

## Hashing and equality {#哈希与相等}

An object that can be a dict key or go in a set has to be **hashable**:

1. it has a `__hash__` whose value does not change over the object's life;
2. it has an `__eq__`;
3. if `a == b` then `hash(a) == hash(b)`.

The immutable built-in types are all hashable (a tuple requires its elements to be hashable too); `list`, `dict` and `set` are not.

```pycon
>>> hash((1, 2)) == hash((1, 2))
True
>>> hash([1, 2])
Traceback (most recent call last):
  ...
TypeError: unhashable type: 'list'
```

**A class that defines only `__eq__` has its `__hash__` set to `None` automatically**, and its instances can no longer go in a set. That is deliberate: you changed what "equal" means and Python cannot know how the hash should follow.

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

To make it hashable, define a `__hash__` over the same fields and guarantee those fields are never modified:

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

In a real project the easier way is `@dataclass(frozen=True)`, which generates `__eq__` and `__hash__` for you; see [modelling data](data-classes.md).

## Everything is an object {#一切皆对象}

Functions, classes and modules are objects themselves: they can be assigned to variables, put in containers, passed as arguments and given attributes.

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

"Functions are first-class objects" is the basis of decorators, callbacks and the strategy pattern, which the next few chapters use repeatedly.

## When an object is reclaimed {#对象什么时候被回收}

CPython relies mainly on **reference counting**: an object is reclaimed the moment no name or container refers to it. A reference cycle (A refers to B and B to A) is handled by the **cyclic garbage collector**, which runs periodically.

`del x` deletes the **name**, not the object. The object is reclaimed only when its reference count reaches zero.

To refer to an object without keeping it alive (a cache, say), use `weakref`:

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
>>> r() is None       # the object has been reclaimed
True
```

!!! interview "Answering in an interview"
    The one sentence at the heart of the object model question: a variable is a name stuck on an object, and assignment only binds and never copies. Everything follows: `a += [3]` modifies in place and every holder of a reference sees it, while `a = a + [3]` creates a new object; a function is passed an object reference, so modifying a mutable object is visible to the caller and rebinding the parameter is not; `is` compares identity and `==` compares value, with singletons (`None`, a sentinel) taking `is`; a shallow copy copies only the outer layer; a class defining only `__eq__` has `__hash__` set to `None` and its instances cannot go in a `set`. Explaining why `t = (1, [2]); t[1] += [3]` both raises and changes `t` (the list is extended in place first, then assigning to the tuple's element fails) earns extra credit.

## Exercises {#练习}

**1. Predict the output.** Think first, then run it.

```py
def f(x, lst=[]):
    lst.append(x)
    return lst

print(f(1), f(2))
```

??? success "Answer"
    It prints `[1, 2] [1, 2]`, not `[1] [1, 2]`.

    There are two reasons:

    - a default argument is evaluated once, **when the function is defined**, and every call shares that one list.
    - `print` evaluates both of its arguments before printing. Both calls returned **the same list object**, which by printing time is already `[1, 2]`.

    The correct form uses `None` as the default:

    ```python
    def f(x, lst=None):
        if lst is None:
            lst = []
        lst.append(x)
        return lst

    assert (f(1), f(2)) == ([1], [2])
    ```

**2. Write a `freeze(obj)`** that turns nested `list`s, `dict`s and `set`s into hashable equivalents (`tuple`, a sorted tuple of key-value pairs, `frozenset`), so that any JSON-style data can be a cache key. `freeze({"b": [1, 2], "a": {3}})` has to equal `freeze({"a": {3}, "b": [1, 2]})`.

??? success "Answer"
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

    Sorting the dict by key is what keeps the keys' insertion order from affecting the result. It assumes the keys are comparable; with keys of mixed types, store the key-value pairs in a `frozenset` instead.

**3. Why is `x is None` preferred over `x == None`?** Write a class whose instances compare `== None` as `True` while `is None` is `False`.

??? success "Answer"
    `==` calls `__eq__`, which can be overloaded arbitrarily; `is` compares identity, cannot be fooled, and is faster.

    ```python
    class Weird:
        def __eq__(self, other):
            return True

    w = Weird()
    assert (w == None) is True
    assert (w is None) is False
    ```

    NumPy arrays and SQLAlchemy's column objects overload `==`, and `== None` on them does not even give a boolean.

## Summary {#小结}

- [x] A variable is a name, assignment only binds, and no object is ever copied.
- [x] Compare values with `==` and singletons (`None`, a sentinel) with `is`.
- [x] Modifying a mutable object in place is visible to everyone holding a reference; rebinding a parameter is not visible to the caller.
- [x] A shallow copy copies only the outer layer; nested mutable data needs a deep copy or an immutable design.
- [x] Being hashable requires `__hash__` and `__eq__` to agree; defining only `__eq__` makes the instances unhashable.
