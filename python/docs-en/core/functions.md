# Functions in depth

<p class="lead">The function is Python's basic unit of organization. This chapter covers the forms a parameter can take, the scoping rules, closures, and the tools in <code>functools</code> and <code>operator</code> that make code shorter.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. In `def f(a, /, b, *, c)`, how may each of `a`, `b` and `c` be passed?
    2. Why does the code below raise `UnboundLocalError`?
       ```py
       count = 0
       def inc():
           count += 1
       ```
    3. What does each function in `[lambda: i for i in range(3)]` return when called? How do you fix it?
    4. How does `functools.partial` differ from wrapping in a `lambda`?
    5. What does `lru_cache` require of the arguments?

??? success "Answers (try it yourself first, then expand)"
    1. `a` can only be passed by position, `b` by position or by keyword, and `c` only by keyword.
    2. Assigning to `count` in the function body (`+=` is an assignment too) makes it a local variable; `count += 1` has to read that local first, and it has no value yet. Modifying a global needs a `global count` declaration (`nonlocal` inside a nested function).
    3. All three return 2: the closure captures the variable `i` itself, and by the time they are called the loop is long over. The fixes: bind the value of the moment with a default argument, `lambda i=i: i`, or use `functools.partial`.
    4. `partial` fixes the bound arguments' values when it is created, and it is an object whose `func`, `args` and `keywords` can be inspected and which can be pickled; a `lambda` looks the outer variables up when it is called, with the closure's late-binding problem.
    5. Every argument has to be hashable (they make up the cache key); and note that the cache holds those arguments and results forever (memory), and that the function should be pure (the same input always giving the same output).

## The five forms of parameter {#参数的五种形式}

```python
def api(pos_only, /, normal, *args, kw_only, **kwargs):
    return pos_only, normal, args, kw_only, kwargs

print(api(1, 2, 3, 4, kw_only=5, extra=6))
# (1, 2, (3, 4), 5, {'extra': 6})
```

| Form | How it is written | How the caller passes it |
| --- | --- | --- |
| positional-only | before the `/` | by position only |
| ordinary | between the `/` and the `*` | by position or by name |
| variadic positional | `*args` | the extra positional arguments collected into a tuple |
| keyword-only | after the `*` or `*args` | by name only |
| variadic keyword | `**kwargs` | the extra keyword arguments collected into a dict |

### When to use keyword-only parameters {#什么时候用仅关键字参数}

Boolean switches and easily confused parameters should be forced to be named, so the call site reads at a glance:

```pycon
>>> def connect(host, port, *, timeout=10, ssl=True):
...     return f"{host}:{port} timeout={timeout} ssl={ssl}"
...
>>> connect("db", 5432, timeout=3)
'db:5432 timeout=3 ssl=True'
>>> connect("db", 5432, 3)
Traceback (most recent call last):
  ...
TypeError: connect() takes 2 positional arguments but 3 were given
```

### When to use positional-only parameters {#什么时候用仅位置参数}

1. The parameter's name means nothing and callers should not depend on it (so it can be renamed freely later), `def distance(p, q, /)` say.
2. Arbitrary keyword arguments have to be accepted too, without colliding with a parameter's name:

```pycon
>>> def render(template, /, **context):
...     return template.format(**context)
...
>>> render("{template} is a word", template="this")   # cannot collide with the first parameter
'this is a word'
```

### A default argument is evaluated once {#默认参数只求值一次}

The default is evaluated when the `def` runs, and every call afterwards shares that object. A mutable default (a list, a dict) is almost always a bug, so use `None` instead without exception:

```python
def append_to(item, target=None):
    if target is None:
        target = []
    target.append(item)
    return target

assert append_to(1) == [1]
assert append_to(2) == [2]
```

Likewise, the time in `def log(msg, when=datetime.now())` is the time the function was defined, not the time it is called.

### Unpacking at the call {#调用时解包}

```pycon
>>> def point(x, y, z=0):
...     return (x, y, z)
...
>>> args = [1, 2]
>>> kwargs = {"z": 3}
>>> point(*args, **kwargs)
(1, 2, 3)
```

## Scope: LEGB {#作用域legb}

![Figure: the LEGB order of name lookup and a closure's cell](../assets/figures/legb-scope.svg){.aig-svg}

Python looks a name up in the order **L**ocal → **E**nclosing → **G**lobal → **B**uiltin:

```python
x = "global"

def outer():
    x = "enclosing"
    def inner():
        return x          # finds the enclosing x
    return inner()

assert outer() == "enclosing"
assert len("abc") == 3    # len lives in the builtin scope
```

The key rule: **assigning to a name anywhere in a function body (including `+=`) makes that name local to the whole function**. That is decided at compile time, so the code below raises when it reads:

```pycon
>>> count = 0
>>> def inc():
...     count += 1
...
>>> inc()
Traceback (most recent call last):
  ...
UnboundLocalError: cannot access local variable 'count' where it is not associated with a value
```

Modifying an outer variable needs an explicit declaration:

- a module-level variable: `global count`
- an enclosing function's variable: `nonlocal count`

!!! tip "Use `global` as little as possible"
    To keep state between calls, prefer a closure, a class instance, or passing the state in and out as arguments and return values. `global` makes a function's behaviour depend on hidden external state, which is hard to test.

## Closures {#闭包}

An inner function refers to the outer function's variables and can still reach them after the outer function has returned. That is a closure.

```pycon
>>> def make_counter():
...     count = 0
...     def inc():
...         nonlocal count
...         count += 1
...         return count
...     return inc
...
>>> c1, c2 = make_counter(), make_counter()
>>> c1(), c1(), c2()
(1, 2, 1)
>>> c1.__closure__[0].cell_contents     # a captured variable lives in a cell
2
```

### The late-binding trap {#延迟绑定陷阱}

A closure captures the **variable**, not the value it had at the time. After the loop, the variable holds the last value:

```pycon
>>> funcs = [lambda: i for i in range(3)]
>>> [f() for f in funcs]
[2, 2, 2]
```

Two fixes:

```pycon
>>> funcs = [lambda i=i: i for i in range(3)]      # a default argument is evaluated at definition
>>> [f() for f in funcs]
[0, 1, 2]
>>> from functools import partial
>>> def identity(x):
...     return x
...
>>> funcs = [partial(identity, i) for i in range(3)]
>>> [f() for f in funcs]
[0, 1, 2]
```

## lambda and higher-order functions {#lambda-与高阶函数}

A `lambda` only suits a small function of **one expression**, usually passed to another function. When it needs a name, several lines or documentation, use `def`.

`map` and `filter` can mostly be replaced by a comprehension in Python, which usually reads better:

```pycon
>>> nums = [1, 2, 3, 4]
>>> list(map(lambda x: x * x, filter(lambda x: x % 2 == 0, nums)))
[4, 16]
>>> [x * x for x in nums if x % 2 == 0]       # preferred
[4, 16]
```

Where a function already exists, `map` is concise: `list(map(int, "1 2 3".split()))`.

## `operator`: in place of a simple lambda {#operator代替简单的-lambda}

```pycon
>>> from operator import itemgetter, attrgetter, methodcaller
>>> rows = [("amy", 95), ("bob", 90)]
>>> sorted(rows, key=itemgetter(1))
[('bob', 90), ('amy', 95)]
>>> itemgetter(0, 1)({"0": "x", 0: "a", 1: "b"})
('a', 'b')
>>> list(map(methodcaller("upper"), ["a", "b"]))
['A', 'B']
```

`attrgetter("address.city")` also reaches nested attributes through a dot. These are faster than the equivalent lambda and say the intent more clearly.

## `functools`: the function toolbox {#functools函数工具箱}

### `partial`: fixing some arguments {#partial固定部分参数}

```pycon
>>> from functools import partial
>>> int_from_hex = partial(int, base=16)
>>> int_from_hex("ff")
255
>>> int_from_hex.func, int_from_hex.keywords
(<class 'int'>, {'base': 16})
```

Against a lambda, a `partial` object can be pickled (so it can go to a subprocess) and shows what it wraps.

### `cache` and `lru_cache`: memoization {#cache-与-lru_cache记忆化}

```pycon
>>> from functools import cache, lru_cache
>>> @cache
... def fib(n):
...     return n if n < 2 else fib(n - 1) + fib(n - 2)
...
>>> fib(100)
354224848179261915075
>>> fib.cache_info()
CacheInfo(hits=98, misses=101, maxsize=None, currsize=101)
```

- `@cache` has no capacity limit; `@lru_cache(maxsize=128)` evicts the least recently used result beyond its capacity.
- The arguments have to be **hashable**, so no lists and no dicts.
- It only suits a **pure function** (the same input always giving the same output, with no side effects).
- Be careful on a method: `self` is part of the cache key too, and the cache keeps the instance from ever being reclaimed. For a method-level cache, consider `functools.cached_property` (see [object orientation](oop.md)).

### `reduce`: folding {#reduce折叠}

```pycon
>>> from functools import reduce
>>> import operator
>>> reduce(operator.mul, [1, 2, 3, 4], 1)
24
```

Where `sum`, `max`, `any`, `all`, `math.prod` or `"".join` will do, do not use `reduce`; they say it more plainly.

### `singledispatch`: dispatching by type {#singledispatch按类型分派}

Choosing the implementation by **the first argument's type** is a good replacement for a long chain of `isinstance` tests:

```python
from datetime import date
from functools import singledispatch

@singledispatch
def to_json(obj):
    raise TypeError(f"cannot serialize {type(obj).__name__}")

@to_json.register
def _(obj: date):
    return obj.isoformat()

@to_json.register
def _(obj: set):
    return sorted(obj)

@to_json.register(list)
@to_json.register(tuple)
def _(obj):
    return [to_json(x) if not isinstance(x, (int, str)) else x for x in obj]

assert to_json(date(2026, 9, 24)) == "2026-09-24"
assert to_json({3, 1, 2}) == [1, 2, 3]
assert to_json((1, date(2026, 1, 1))) == [1, "2026-01-01"]
```

A class's methods use `functools.singledispatchmethod`.

### The rest {#其他}

- `functools.wraps`: essential when writing a decorator, see the next chapter.
- `functools.total_ordering`: write `__eq__` and one comparison method and the rest are filled in, see [object orientation](oop.md).
- `functools.cached_property`: an attribute computed once.

## Introspecting a function {#函数的内省}

A function object carries a wealth of metadata, which is how a framework (FastAPI, pytest) implements "injecting arguments automatically":

```pycon
>>> import inspect
>>> def handler(request, user_id: int, *, verbose: bool = False) -> dict:
...     ...
...
>>> sig = inspect.signature(handler)
>>> [(p.name, p.kind.name) for p in sig.parameters.values()]
[('request', 'POSITIONAL_OR_KEYWORD'), ('user_id', 'POSITIONAL_OR_KEYWORD'), ('verbose', 'KEYWORD_ONLY')]
>>> sig.parameters["user_id"].annotation
<class 'int'>
>>> sig.bind("req", "42").arguments
{'request': 'req', 'user_id': '42'}
```

!!! interview "How to explain it"
    Four traps come up most around functions: the kinds of parameter (before the `/` only by position, after the `*` only by keyword); a mutable default evaluated once at definition, replaced by `None`; assigning to a variable in a function makes it local, so reading an outer variable before assigning gives `UnboundLocalError` and modifying one needs `nonlocal` / `global`; and a closure captures the variable rather than the value, so `[lambda: i for i in range(3)]` all return 2, fixed by the default argument `lambda i=i: i` or by `functools.partial`. Then mention that `lru_cache` requires hashable arguments and that `singledispatch` dispatches on the first argument's type.

## Exercises {#练习}

**1. Function composition.** Write `compose(*funcs)` returning a new function that calls them right to left: `compose(f, g, h)(x) == f(g(h(x)))`. With no functions it returns the identity.

??? success "Answer"
    ```python
    from functools import reduce

    def compose(*funcs):
        def composed(x):
            return reduce(lambda acc, f: f(acc), reversed(funcs), x)
        return composed

    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert compose(inc, double)(5) == 11     # inc(double(5))
    assert compose(double, inc)(5) == 12     # double(inc(5))
    assert compose()(5) == 5
    ```

**2. Fix the button callbacks.** The code below means to bind each button a callback that "prints its own name", and they all print `C`. Fix it two ways.

```py
callbacks = {}
for name in ["A", "B", "C"]:
    callbacks[name] = lambda: print(name)
```

??? success "Answer"
    ```python
    from functools import partial

    # the first way: a default argument binds the current value at definition
    callbacks = {}
    for name in ["A", "B", "C"]:
        callbacks[name] = lambda name=name: name
    assert [f() for f in callbacks.values()] == ["A", "B", "C"]

    # the second way: partial, harder to override by a stray argument
    def echo(value):
        return value

    callbacks = {name: partial(echo, name) for name in ["A", "B", "C"]}
    assert [f() for f in callbacks.values()] == ["A", "B", "C"]
    ```

    The first way's drawback is that a caller can override `name` by passing an argument; the second says the intent more clearly.

**3. Counting paths in a grid.** Walking from the top left to the bottom right of an `m × n` grid, moving only right or down, how many paths are there? Write a recursive version and make `paths(18, 18)` finish instantly with `@cache`.

??? success "Answer"
    ```python
    from functools import cache

    @cache
    def paths(m, n):
        if m == 1 or n == 1:
            return 1
        return paths(m - 1, n) + paths(m, n - 1)

    assert paths(3, 3) == 6
    assert paths(18, 18) == 2333606220
    ```

    Without the cache the recursion tree is exponential; with it, each `(m, n)` is computed once, which is O(m·n).

**4. Design a usable API.** Write `retry_call(func, *args, retries=3, delay=0.0, exceptions=(Exception,), **kwargs)`: it calls `func(*args, **kwargs)`, retries on the given exceptions up to `retries` times, and raises if the last attempt still fails. Consider: why should `retries` and the rest be keyword-only?

??? success "Answer"
    ```python
    import time

    def retry_call(func, *args, retries=3, delay=0.0, exceptions=(Exception,), **kwargs):
        for attempt in range(1, retries + 1):
            try:
                return func(*args, **kwargs)
            except exceptions:
                if attempt == retries:
                    raise
                time.sleep(delay)

    calls = []
    def flaky(x):
        calls.append(x)
        if len(calls) < 3:
            raise ConnectionError("try again")
        return x * 10

    assert retry_call(flaky, 4) == 40
    assert len(calls) == 3
    ```

    Parameters after `*args` become keyword-only automatically. That keeps `retry_call`'s own configuration out of the called function's positional arguments, and makes it impossible for a caller to pass a business argument as `retries` by mistake.

## Summary {#小结}

- [x] Use `*` to force keyword arguments and make the call site readable; use `/` to protect parameter names you do not want to expose.
- [x] Replace a mutable default with `None`.
- [x] Assigning to a variable in a function makes it local; modifying an outer one takes `nonlocal`/`global`.
- [x] A closure captures the variable and not the value, so creating functions in a loop needs particular care.
- [x] `partial`, `cache`, `singledispatch` and `operator` make a great deal of code shorter and clearer.
