# Decorators

<p class="lead">A decorator is a function that takes a function and returns a function. Once that sentence and closures are clear, there is nothing mysterious about decorators. Logging, timing, retrying, caching, permission checks and route registration can all be done with them elegantly.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. `@deco` above a function is equivalent to which single line of code?
    2. What goes wrong without `functools.wraps`?
    3. Why does a decorator with arguments (`@retry(times=3)`, say) need three levels of function?
    4. With two decorators stacked, which runs first?
    5. How do you write a decorator usable both as `@deco` and as `@deco(level=2)`?

??? success "Answers (try it yourself first, then expand)"
    1. `f = deco(f)`, run once when the function is defined.
    2. The wrapped function loses the original's name, docstring, signature and other metadata (`__name__` becomes `wrapper`), which affects debugging, logging and every tool that relies on them (pytest, `inspect`).
    3. `retry(times=3)` has to be called first and return the actual decorator; the decorator then takes the function and returns the wrapper, which is the three levels "take the arguments → take the function → wrap the call".
    4. Decorating goes bottom to top: the one nearest the function decorates first. Calling goes top to bottom: the outermost wrapper runs first.
    5. Make the first parameter default to `None` and the rest keyword-only: used directly as `@deco`, the first argument is the function, so decorate and return it; otherwise (`@deco(level=2)`) return a decorator that takes the function.

## What it is: syntactic sugar {#本质语法糖}

![Figure: a decorator is f = deco(f) - a call enters the wrapper first, and the outermost of a stack runs first](../assets/figures/decorator-wrap.svg){.aig-svg}

```py
@deco
def f():
    ...
```

is exactly equivalent to:

```py
def f():
    ...
f = deco(f)
```

A decorator runs once **when the function is defined**, not on every call. Its return value replaces the original name.

```pycon
>>> def announce(func):
...     print(f"decorating {func.__name__}")
...     return func
...
>>> @announce
... def hello():
...     return "hi"
...
decorating hello
>>> hello()
'hi'
```

## The first real decorator: timing {#第一个真正的装饰器计时}

Most decorators return a **wrapper function** that does something before and after calling the original:

```python
import functools
import time

def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            print(f"{func.__name__} took {elapsed * 1000:.1f} ms")
    return wrapper

@timed
def slow_add(a, b):
    """Add two numbers slowly."""
    time.sleep(0.01)
    return a + b

assert slow_add(1, 2) == 3          # prints slow_add took 10.x ms
assert slow_add.__name__ == "slow_add"
assert slow_add.__doc__ == "Add two numbers slowly."
```

The key points:

- `*args, **kwargs` lets the wrapper take any arguments and forward them unchanged.
- The original's return value has to be `return`ed.
- A `try/finally` records the time even when the original raises.

### Why `functools.wraps` is necessary {#为什么必须用-functoolswraps}

Without `wraps`, the decorated function's name, documentation and signature all become the `wrapper`'s:

```pycon
>>> def bad_deco(func):
...     def wrapper(*args, **kwargs):
...         return func(*args, **kwargs)
...     return wrapper
...
>>> @bad_deco
... def greet(name):
...     """Greet someone."""
...
>>> greet.__name__, greet.__doc__
('wrapper', None)
```

That breaks logging, debugging, documentation generation and any framework that relies on a function's signature (pytest's fixtures, FastAPI's argument parsing). `wraps` copies `__name__`, `__doc__`, `__module__`, `__qualname__` and the rest, and sets `__wrapped__` to the original, which `inspect.signature` follows to find the real signature.

```pycon
>>> import functools, inspect
>>> def good_deco(func):
...     @functools.wraps(func)
...     def wrapper(*args, **kwargs):
...         return func(*args, **kwargs)
...     return wrapper
...
>>> @good_deco
... def greet(name, *, loud=False):
...     """Greet someone."""
...
>>> greet.__name__, str(inspect.signature(greet))
('greet', '(name, *, loud=False)')
```

## A decorator with arguments {#带参数的装饰器}

`@retry(times=3)` runs in this order: `retry(times=3)` is called first, and **its return value** is the actual decorator. So one more level is needed:

```python
import functools
import time

def retry(times=3, exceptions=(Exception,), delay=0.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == times:
                        raise
                    print(f"{func.__name__} failed ({e!r}), retry {attempt}/{times - 1}")
                    time.sleep(delay)
        return wrapper
    return decorator

attempts = []

@retry(times=3, exceptions=(ConnectionError,))
def fetch():
    attempts.append(1)
    if len(attempts) < 3:
        raise ConnectionError("network down")
    return "data"

assert fetch() == "data" and len(attempts) == 3
```

The three levels are: **the configuration level** (taking the decorator's arguments) → **the decorating level** (taking the function) → **the wrapping level** (taking the call's arguments).

### Supporting both `@deco` and `@deco(...)` {#同时支持-deco-和-deco}

Tell the two uses apart by "is the first argument a function", and make the configuration keyword-only:

```python
import functools

def log_calls(func=None, *, prefix="CALL"):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            print(f"{prefix} {f.__name__}{args}")
            return f(*args, **kwargs)
        return wrapper

    if func is None:          # used as @log_calls(prefix=...)
        return decorator
    return decorator(func)    # used as @log_calls

@log_calls
def add(a, b):
    return a + b

@log_calls(prefix="DEBUG")
def sub(a, b):
    return a - b

assert add(1, 2) == 3     # prints CALL add(1, 2)
assert sub(5, 3) == 2     # prints DEBUG sub(5, 3)
```

## The order of a stack {#叠加顺序}

```py
@a
@b
def f(): ...
```

is equivalent to `f = a(b(f))`: **the one nearest the function decorates first**, and when called **the outer wrapper runs first**.

```pycon
>>> def tag(name):
...     def decorator(func):
...         @functools.wraps(func)
...         def wrapper():
...             return f"<{name}>{func()}</{name}>"
...         return wrapper
...     return decorator
...
>>> @tag("b")
... @tag("i")
... def text():
...     return "hi"
...
>>> text()
'<b><i>hi</i></b>'
```

The order sometimes matters. With `@app.route` and `@login_required` together, `route` has to be outermost so that the version with the permission check is the one registered.

## A decorator written as a class {#用类实现装饰器}

To keep state, use a class implementing `__call__`:

```python
import functools

class CountCalls:
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.func(*args, **kwargs)

@CountCalls
def ping():
    return "pong"

ping(); ping()
assert ping.calls == 2
assert ping.__name__ == "ping"
```

!!! warning "A class decorator on a method"
    Applying the `CountCalls` above directly to **a method in a class** does not get `self` passed in, because a class instance is not a descriptor. Supporting methods requires implementing `__get__` (see [metaprogramming](../types/metaprogramming.md)), or simply writing it as a function closure with the state hung on the wrapper's attributes: `wrapper.calls = 0`.

## Decorating a class {#装饰类}

A decorator can also take a class and return a class. The standard library's `@dataclass` and `@functools.total_ordering` both do.

One common use is a **registry**: the decorated function or class is recorded in a dict and looked up by name later. A web framework's routes, a command line's subcommands and a plugin system all follow this pattern:

```python
COMMANDS = {}

def command(name):
    def decorator(func):
        COMMANDS[name] = func
        return func            # returned unchanged, not wrapped
    return decorator

@command("greet")
def greet(who):
    return f"hello {who}"

@command("shout")
def shout(who):
    return f"HELLO {who.upper()}!"

def dispatch(line):
    name, arg = line.split(maxsplit=1)
    return COMMANDS[name](arg)

assert dispatch("greet world") == "hello world"
assert dispatch("shout py") == "HELLO PY!"
```

## Giving a decorator type annotations {#给装饰器加类型标注}

If your project type-checks, use `ParamSpec` to have the decorator keep the original's parameter types. From 3.12 it can be written directly in the function's type parameters:

```python
import functools
from collections.abc import Callable

def logged[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        print(f"calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@logged
def area(w: float, h: float) -> float:
    return w * h

assert area(2, 3) == 6
```

The type checker then knows `area` is still `(w: float, h: float) -> float`. See [type annotations](../types/typing.md).

## The decorators in the standard library {#标准库里的装饰器}

| Decorator | What it does |
| --- | --- |
| `@property` `@classmethod` `@staticmethod` | define special methods and attributes, see [object orientation](oop.md) |
| `@functools.cache` / `@lru_cache` | memoization |
| `@functools.wraps` | for writing a decorator |
| `@functools.singledispatch` | dispatch by type |
| `@functools.total_ordering` | fill in the comparison methods |
| `@dataclasses.dataclass` | generate a data class |
| `@contextlib.contextmanager` | write a context manager as a generator |
| `@typing.override` <span class="since">3.12+</span> | mark a method as overriding the parent's |
| `@warnings.deprecated` <span class="since">3.13+</span> | mark something deprecated, warning when it is called |

!!! interview "How to explain it"
    The standard way to explain decorators: `@deco` is `f = deco(f)` run once at definition; the wrapper forwards through `*args, **kwargs`, returns the result, and uses `functools.wraps` to keep the name and documentation (without which logging, debugging and serialization all see the wrapper); a decorator with arguments is "a function returning a decorator", hence three levels; in a stack the one nearest the function decorates first and the outer wrapper runs first. Writing a `@retry(times=3)` with a retry count and an exception type on the spot, and naming the registry pattern (returning the function unchanged and only recording it) and the `ParamSpec` annotation, earns extra credit.

## Exercises {#练习}

**1. A `@debug` decorator.** It prints the arguments and the return value on each call, as `add(1, b=2) -> 3`. It has to keep the function's metadata.

??? success "Answer"
    ```python
    import functools

    def debug(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            parts = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
            result = func(*args, **kwargs)
            print(f"{func.__name__}({', '.join(parts)}) -> {result!r}")
            return result
        return wrapper

    @debug
    def add(a, b):
        return a + b

    assert add(1, b=2) == 3        # prints add(1, b=2) -> 3
    assert add.__name__ == "add"
    ```

**2. An `@once` decorator.** The decorated function runs only the first time, and later calls return the first result (commonly used for initialization).

??? success "Answer"
    ```python
    import functools

    def once(func):
        sentinel = object()
        result = sentinel

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal result
            if result is sentinel:
                result = func(*args, **kwargs)
            return result
        return wrapper

    calls = []

    @once
    def init():
        calls.append(1)
        return "ready"

    assert init() == "ready" and init() == "ready"
    assert len(calls) == 1
    ```

    A sentinel rather than `None` is used for the test because the function itself may return `None`. Several threads also need a lock, see [threads and processes](../concurrency/threads-processes.md).

**3. Exponential backoff.** Improve the `retry` above: wait `base_delay * 2**(n-1)` seconds before the nth retry, capped at `max_delay`. Make the "sleep" function an injectable argument, so tests need not really wait.

??? success "Answer"
    ```python
    import functools
    import time

    def retry(times=3, *, exceptions=(Exception,), base_delay=0.1, max_delay=2.0, sleep=time.sleep):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                for attempt in range(1, times + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions:
                        if attempt == times:
                            raise
                        sleep(min(base_delay * 2 ** (attempt - 1), max_delay))
            return wrapper
        return decorator

    waits = []

    @retry(times=5, base_delay=0.5, max_delay=1.5, sleep=waits.append)
    def always_fails():
        raise TimeoutError

    try:
        always_fails()
    except TimeoutError:
        pass
    assert waits == [0.5, 1.0, 1.5, 1.5]
    ```

    Injecting `sleep` as an argument lets a test pass a function that only records its argument, so the test finishes instantly and the backoff sequence can be asserted. That is a very practical piece of "dependency injection".

## Summary {#小结}

- [x] `@deco` is `f = deco(f)`, run once at definition.
- [x] The wrapper forwards through `*args, **kwargs`, remembers to `return` the result, and always carries `@functools.wraps`.
- [x] A decorator with arguments is "a function returning a decorator", three levels deep.
- [x] In a stack, the one nearest the function decorates first and the outer wrapper runs first.
- [x] The registry pattern needs no wrapper and returns the function unchanged.
