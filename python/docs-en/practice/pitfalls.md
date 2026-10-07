# Idioms and common traps

<p class="lead">This page is the book's quick reference: idiomatic forms in the first half and the traps people fall into most in the second. Every entry uses the shortest example that makes the point, with the full explanation in its own chapter. It suits checking code against after writing it, or a quick pass before an interview.</p>

## The idiomatic forms {#地道的写法}

### Iterating {#遍历}

```py
# use enumerate when the index is needed
for i, item in enumerate(items):          # rather than for i in range(len(items))
    ...

# iterate several sequences together with zip (add strict=True when they should be the same length)
for name, score in zip(names, scores, strict=True):
    ...

# iterate a dict's key-value pairs
for key, value in d.items():
    ...

# iterating in reverse and in sorted order
for x in reversed(items): ...
for x in sorted(items, key=len): ...
```

### Unpacking and swapping {#解包与交换}

```pycon
>>> a, b = 1, 2
>>> a, b = b, a
>>> head, *tail = [1, 2, 3]
>>> (a, b), head, tail
((2, 1), 1, [2, 3])
```

### Testing {#判断}

```pycon
>>> x = 5
>>> 0 < x < 10                            # a chained comparison
True
>>> x in {1, 3, 5}                        # testing several values with in plus a set
True
>>> items = [0, "", None, 3]
>>> any(items), all(items)
(True, False)
```

- Test an empty container with `if not items:` and `None` with `if x is None:`.
- `isinstance(x, (int, float))` rather than `type(x) == int`.

### Building a collection {#构建集合}

```pycon
>>> words = ["apple", "Bob", "cat"]
>>> [w.upper() for w in words if len(w) > 2]
['APPLE', 'BOB', 'CAT']
>>> {w: len(w) for w in words}
{'apple': 5, 'Bob': 3, 'cat': 3}
>>> ", ".join(words)                      # build strings with join
'apple, Bob, cat'
>>> list(dict.fromkeys([3, 1, 3, 2]))     # order-preserving deduplication
[3, 1, 2]
```

### Dicts {#字典}

```py
value = d.get(key, default)               # rather than if key in d: ... else: ...
d.setdefault(key, []).append(x)           # or use defaultdict(list)
merged = defaults | overrides             # merging
counts = Counter(words)                   # counting
```

### The walrus operator `:=` {#海象运算符-}

Assigning inside an expression, to avoid computing or calling something twice:

```pycon
>>> import re
>>> if (m := re.search(r"(\d+)", "order 42")):
...     print(m[1])
...
42
>>> data = [3, 8, 1, 9]
>>> [y for x in data if (y := x * x) > 10]
[64, 81]
```

Reading blocks until the end is its other classic use: `while chunk := f.read(8192): ...`.

### `for ... else` {#for--else}

The `else` runs when the loop **was not broken out of**, which suits "what to do when it is not found":

```pycon
>>> for n in [4, 6, 8]:
...     if n % 2:
...         print("found odd", n)
...         break
... else:
...     print("no odd number")
...
no odd number
```

### Resource management {#资源管理}

- Files, locks and connections always go in a `with`.
- Paths use `pathlib.Path`, and opening a text file carries `encoding="utf-8"`.

### The rest {#其他}

- Use `_` for a value you do not care about: `for _ in range(3)`, `_, ext = os.path.splitext(p)`.
- Put underscores in large numbers: `1_000_000`.
- Return a tuple or a dataclass for several values, not a list whose meaning is unclear.
- Make a function's switches keyword-only: `def fetch(url, *, retry=False)`.

## The common traps {#常见的坑}

### A mutable default argument {#可变默认参数}

```pycon
>>> def add(x, bucket=[]):
...     bucket.append(x)
...     return bucket
...
>>> add(1), add(2)
([1, 2], [1, 2])
```

The default is evaluated once at definition. Use `None` as the default and create the list inside the function. See [functions in depth](../core/functions.md#默认参数只求值一次).

### A closure's late binding {#闭包的延迟绑定}

```pycon
>>> fs = [lambda: i for i in range(3)]
>>> [f() for f in fs]
[2, 2, 2]
```

A closure captures the variable and not the value. Fix it with the default argument `lambda i=i: i` or with `functools.partial`. See [functions in depth](../core/functions.md#延迟绑定陷阱).

### Duplicating a nested list with `*` {#用--复制嵌套列表}

```pycon
>>> grid = [[0] * 2] * 2
>>> grid[0][0] = 1
>>> grid
[[1, 0], [1, 0]]
```

The outer duplication copies a reference to the same inner list. Use `[[0] * 2 for _ in range(2)]`.

### Modifying a container while iterating it {#遍历时修改容器}

```pycon
>>> nums = [1, 2, 2, 3]
>>> for n in nums:
...     if n == 2:
...         nums.remove(n)
...
>>> nums                                  # the second 2 was skipped
[1, 2, 3]
>>> d = {"a": 1, "b": 2}
>>> for k in d:
...     if d[k] == 1:
...         del d[k]
...
Traceback (most recent call last):
  ...
RuntimeError: dictionary changed size during iteration
```

Build a new container (`[n for n in nums if n != 2]`), or iterate a copy (`for k in list(d):`).

### Comparing values with `is` {#用-is-比较值}

`a is b` compares "is it the same object". Small integers and some strings are cached, so `is` sometimes happens to return `True` and stops with a different value. **Always compare values with `==`**, and use `is` only for `None`, `True`, `False` and sentinel objects.

### Floating-point precision {#浮点数精度}

```pycon
>>> 0.1 + 0.2 == 0.3
False
>>> import math
>>> math.isclose(0.1 + 0.2, 0.3)
True
>>> from decimal import Decimal
>>> Decimal("0.1") + Decimal("0.2") == Decimal("0.3")
True
>>> round(2.5), round(3.5), round(2.675, 2)       # banker's rounding, and the binary representation's error
(2, 4, 2.67)
```

Compare floats with `math.isclose`; use `Decimal` for money (constructed from a string) or integer "cents". `round` uses banker's rounding and not round-half-up.

### The sign of floor division and modulo {#整除和取模的符号}

```pycon
>>> -7 // 2, -7 % 2
(-4, 1)
>>> int(-7 / 2)
-3
```

`//` rounds towards negative infinity rather than truncating towards zero. Worth remembering when interoperating with other languages (comparing against C's or Java's results).

### `bool` is a subclass of `int` {#bool-是-int-的子类}

```pycon
>>> True + True, isinstance(True, int)
(2, True)
>>> {1: "one", True: "true"}
{1: 'true'}
```

`True == 1` and their hashes match, so they are the same key in a dict. When validating "is it an integer" and booleans have to be excluded, add `not isinstance(x, bool)`.

### In-place methods return `None` {#就地修改的方法返回-none}

```pycon
>>> items = [3, 1, 2]
>>> result = items.sort()
>>> print(result)
None
>>> sorted([3, 1, 2])                     # use sorted when a return value is needed
[1, 2, 3]
```

`list.sort()`, `list.append()`, `list.reverse()`, `dict.update()` and `random.shuffle()` all modify in place and return `None`.

### An iterator is single-use {#迭代器只能用一次}

```pycon
>>> squares = map(lambda x: x * x, [1, 2, 3])
>>> sum(squares), sum(squares)
(14, 0)
```

`map`, `filter`, `zip`, generators and file objects are all single-use. To use one several times, `list()` it first.

### `except`'s variable is deleted after the block {#except-的变量在块结束后被删除}

```py
try:
    1 / 0
except ZeroDivisionError as e:
    pass
print(e)          # NameError: e is deleted at the end of the except block
```

To use the exception object outside the block, assign it to another variable first.

### Catching too broadly {#捕获过宽的异常}

```py
try:
    value = compute(data)
except Exception:          # even a NameError from a typo gets swallowed
    value = None
```

Catch only the specific exceptions you expect and know how to handle, and keep the `try` block small. See [exceptions and context managers](../core/errors-context.md).

### A script named after a standard library module {#脚本和标准库模块同名}

Naming your own file `random.py`, `json.py`, `email.py` or `test.py` makes `import random` import your file and produces a baffling `AttributeError`. From 3.13 the error message suggests this as a possible cause, but it is best avoided from the start.

### Circular imports {#循环导入}

`a.py` imports `b.py` and `b.py` imports `a.py`, which can give `ImportError: cannot import name ... (most likely due to a circular import)`. The ways out:

1. pull what both depend on into a third module;
2. import inside the function (deferring it to the call);
3. when only an annotation needs it, import it inside an `if TYPE_CHECKING:` block.

A circular import usually says the modules' responsibilities are divided wrongly.

### Relative imports and how it is run {#相对导入与运行方式}

When a module inside a package uses a relative import (`from .utils import x`), running `python mypkg/cli.py` directly gives `ImportError: attempted relative import with no known parent package`. Run it as a module instead: `python -m mypkg.cli`, or through the command defined under `[project.scripts]`.

### A blocking call inside a coroutine {#在协程里调用阻塞函数}

A `time.sleep()` or `requests.get()` inside an `async def` stops the whole event loop. Use `await asyncio.sleep()`, an asynchronous HTTP library, or `await asyncio.to_thread(func)`. See [asyncio](../concurrency/asyncio.md#在异步代码里调用阻塞代码).

### Times without a zone {#时间没有时区}

`datetime.now()` and `datetime.utcnow()` (deprecated) both return times with no zone, which go wrong across zones and across servers. Use `datetime.now(UTC)`, see [the standard library in daily use](../engineering/stdlib.md#datetime-与-zoneinfo处理时间).

### The default encoding {#默认编码}

`open("f.txt")` without an `encoding` is usually not UTF-8 on Windows. Always write `encoding="utf-8"`.

## A code review checklist {#代码审查清单}

Before submitting, check against this:

- [ ] no mutable default arguments, and no mutable class attribute used as instance data
- [ ] no bare `except:`, no swallowed exceptions, and the `try` blocks are small enough
- [ ] files, locks and connections are all inside a `with`
- [ ] `None` is compared with `is`, values with `==`, floats with `isclose`, and money uses `Decimal`
- [ ] no `in` on a `list` and no `pop(0)` inside a loop
- [ ] times carry a zone and text files specify an encoding
- [ ] public functions have annotations and docstrings
- [ ] the new logic has tests, and ruff and mypy pass
