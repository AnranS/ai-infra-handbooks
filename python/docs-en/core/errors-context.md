# Exceptions and context managers

<p class="lead">Writing code that runs is not hard; what is hard is behaving correctly when something goes wrong: the error message is useful, the resource is always released, and the exceptions that should not be swallowed are not. This chapter covers handling exceptions properly, and managing resources with the <code>with</code> statement.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. When does each of `try`, `except`, `else` and `finally` run?
    2. How does `raise NewError(...) from e` differ from a plain `raise NewError(...)`?
    3. Why should you never write a bare `except:`?
    4. What does returning `True` from `__exit__` mean?
    5. In a generator decorated with `contextlib.contextmanager`, what do the parts before and after the `yield` correspond to?

??? success "Answers (try it yourself first, then expand)"
    1. `try` runs first; a matching exception runs the corresponding `except`; `else` runs when `try` raised nothing; and `finally` runs last whatever happened (including on a `return` or a `break`).
    2. `from e` records the original as the new exception's direct cause (`__cause__`), and the traceback says "the above exception was the direct cause of the following exception"; a plain `raise` leaves the original as implicit context (`__context__`), which reads as if another error occurred while handling the first.
    3. It catches everything, including `KeyboardInterrupt` and `SystemExit`, so the program cannot be interrupted with Ctrl+C, and it swallows real bugs silently. Write `except Exception` at minimum, and preferably catch only the specific types.
    4. The exception has been handled and is swallowed, so execution continues normally after the `with` statement.
    5. Before the `yield` is the equivalent of `__enter__` (acquiring the resource) and the value yielded is what `as` receives; after the `yield` is the equivalent of `__exit__` (releasing it), and for cleanup to happen on an exception too it has to be in a `try/finally`'s `finally`.

## The exception hierarchy {#异常的层级}

![Figure: the exception hierarchy - BaseException, Exception and the common subclasses](../assets/figures/exception-tree.svg){.aig-svg}

Every exception inherits from `BaseException`. Day to day, only subclasses of `Exception` should be caught:

<!-- i18n:diagram 5d09660f72 -->
```text
BaseException
├── SystemExit            raised by sys.exit()
├── KeyboardInterrupt     raised by Ctrl+C
├── GeneratorExit
├── BaseExceptionGroup
└── Exception             <- the only branch application code deals with
    ├── ArithmeticError → ZeroDivisionError, OverflowError
    ├── LookupError     → KeyError, IndexError
    ├── OSError         → FileNotFoundError, PermissionError, TimeoutError, ConnectionError ...
    ├── ValueError      → UnicodeDecodeError ...
    ├── TypeError, AttributeError, NameError, RuntimeError ...
    └── ExceptionGroup
```

A bare `except:` and `except BaseException:` swallow Ctrl+C and `sys.exit()` along with everything else, so the program cannot exit properly. **Always name the specific exception types you mean to catch.**

## The full `try` statement {#完整的-try-语句}

```python
def read_port(text):
    try:
        port = int(text)                 # put only the line that may fail in the try
    except ValueError:
        print(f"not a number: {text!r}")
        return None
    else:
        print("parsed ok")               # runs when nothing was raised
        return port
    finally:
        print("done")                    # runs whatever happens, for cleanup

assert read_port("8080") == 8080         # parsed ok / done
assert read_port("abc") is None          # not a number / done
```

- Keep the `try` block as **small** as possible: wrap only the code that may raise what you mean to handle, or you may catch the same kind of exception from somewhere else.
- `else` holds what should happen only on success, and an exception it raises is not caught by the `except` above.
- Do not write `return`, `break` or `continue` in a `finally`: they quietly swallow an exception in flight. From 3.14 this produces a `SyntaxWarning`.

Catching several exceptions at once:

```py
except (KeyError, IndexError) as e:     # parentheses are required when as is used
    ...
except KeyError, IndexError:            # 3.14+ allows them to be omitted without as
    ...
```

## Raising an exception {#抛出异常}

### Choosing the right type {#选对异常类型}

| Situation | Raise |
| --- | --- |
| an argument of the wrong type | `TypeError` |
| the right type with an illegal value | `ValueError` |
| a missing key or index | `KeyError` / `IndexError` (or a subclass) |
| the object's current state does not allow this operation | `RuntimeError` or your own exception |
| a method a subclass has to implement | `NotImplementedError` |

The message has to **carry the context**: `ValueError("port must be 1-65535, got 70000")` is far more useful than `ValueError("invalid")`.

### Chaining: `raise ... from ...` {#异常链raise--from-}

Raising a new exception inside an `except` block has Python record the original in `__context__` automatically, printed as "During handling of the above exception, another exception occurred", which reads as if **another bug appeared while handling the first**.

When the type is **converted deliberately**, use `from` to state the causation:

```pycon
>>> class ConfigError(Exception):
...     pass
...
>>> def load_port(cfg):
...     try:
...         return int(cfg["port"])
...     except (KeyError, ValueError) as e:
...         raise ConfigError(f"bad port in config: {cfg!r}") from e
...
>>> try:
...     load_port({"port": "http"})
... except ConfigError as e:
...     print(e, "| caused by:", repr(e.__cause__))
...
bad port in config: {'port': 'http'} | caused by: ValueError("invalid literal for int() with base 10: 'http'")
```

The traceback then says "The above exception was the direct cause of the following exception". `raise NewError(...) from None` hides the original, which is only for when the original means nothing to the caller.

### Adding information: `add_note` {#给异常补充信息add_note}

Without changing the type, only appending context:

```pycon
>>> def parse_all(lines):
...     for lineno, line in enumerate(lines, 1):
...         try:
...             int(line)
...         except ValueError as e:
...             e.add_note(f"at line {lineno}")
...             raise
...
>>> try:
...     parse_all(["1", "2", "x"])
... except ValueError as e:
...     print(e.__notes__)
...
['at line 3']
```

The notes appear at the end of the traceback. A bare `raise` re-raises the current exception and keeps the original traceback.

## Custom exceptions {#自定义异常}

A library or module should define its own exception hierarchy: one base class plus several specific subclasses. A caller can then catch coarsely or finely as needed.

```python
class PaymentError(Exception):
    """所有支付相关错误的基类。"""

class CardDeclined(PaymentError):
    def __init__(self, card_last4, reason):
        super().__init__(f"card ****{card_last4} declined: {reason}")
        self.card_last4 = card_last4
        self.reason = reason

class InsufficientFunds(CardDeclined):
    pass

try:
    raise InsufficientFunds("4242", "balance too low")
except PaymentError as e:              # the caller only cares that the payment failed
    assert isinstance(e, CardDeclined)
    assert e.reason == "balance too low"
    assert str(e) == "card ****4242 declined: balance too low"
```

Remember the `super().__init__(message)`, or `str(e)` and the traceback show no message.

## EAFP and LBYL {#eafp-与-lbyl}

- **LBYL** (Look Before You Leap): `if key in d: value = d[key]`
- **EAFP** (Easier to Ask Forgiveness than Permission): `try: value = d[key] except KeyError: ...`

The Python community prefers EAFP:

```py
# LBYL: the file may be deleted between the check and the use (a race), and it costs one more system call
if os.path.exists(path):
    with open(path) as f:
        ...

# EAFP: no race, and the intent is clearer
try:
    with open(path) as f:
        ...
except FileNotFoundError:
    ...
```

But do not be dogmatic: when "failure" is the common case, or the check is cheap and obvious (`if items:`), LBYL fits better.

## Several exceptions at once: `ExceptionGroup` {#同时处理多个异常exceptiongroup}

Concurrent tasks may fail several **at the same time**. From 3.11, an `ExceptionGroup` packs several exceptions into one and `except*` handles some of them by type:

```pycon
>>> def run_all():
...     raise ExceptionGroup("batch failed", [
...         ValueError("bad input"),
...         TimeoutError("slow service"),
...         ValueError("another bad input"),
...     ])
...
>>> try:
...     run_all()
... except* ValueError as eg:
...     print("value errors:", [str(e) for e in eg.exceptions])
... except* TimeoutError as eg:
...     print("timeouts:", len(eg.exceptions))
...
value errors: ['bad input', 'another bad input']
timeouts: 1
```

Unlike an ordinary `except`, **several `except*` branches can all run**. You meet it most often with `asyncio.TaskGroup`, see [asyncio](../concurrency/asyncio.md).

## Logging an exception {#记录异常}

```py
import logging
logger = logging.getLogger(__name__)

try:
    process()
except Exception:
    logger.exception("processing failed")   # attaches the full traceback automatically
    raise                                   # after logging it, usually re-raise
```

**Either handle it or re-raise it, and never "log it and pretend nothing happened".** Swallowing an exception makes the problem surface further away in a stranger form.

## Context managers {#上下文管理器}

The `with` statement guarantees "prepare on entry, clean up on exit", **whether it leaves normally or through an exception**. Files, locks, database transactions and temporary directories should all use it.

```py
with open("data.txt", encoding="utf-8") as f:
    data = f.read()
# the file is certainly closed on leaving the with block, even if read() raised
```

Several resources go in one `with`, and from 3.10 parentheses allow a line break:

```py
with (
    open("in.txt", encoding="utf-8") as src,
    open("out.txt", "w", encoding="utf-8") as dst,
):
    dst.write(src.read())
```

### Written as a class {#用类实现}

Implement two methods, `__enter__` and `__exit__`:

```python
import time

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self                          # the name after as receives this return value

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start
        return False                         # does not swallow the exception

with Timer() as t:
    sum(range(100_000))
assert t.elapsed > 0
```

The three arguments `__exit__` receives describe the exception raised in the block (all `None` when there was none). **Returning a true value means "the exception is handled, do not re-raise it"**, so return `False` or `None` unless you really do mean to swallow it.

### Written as a generator: `@contextmanager` {#用生成器实现contextmanager}

Most of the time, a generator through `contextlib.contextmanager` is simpler: before the `yield` is `__enter__`, the value yielded goes to `as`, and after the `yield` is `__exit__`.

```python
import os
from contextlib import contextmanager

@contextmanager
def temp_env(**overrides):
    old = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:                                 # the finally is required, or an exception in the block skips the restore
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

with temp_env(APP_MODE="test"):
    assert os.environ["APP_MODE"] == "test"
assert "APP_MODE" not in os.environ
```

### The other tools in `contextlib` {#contextlib-里的其他工具}

```pycon
>>> from contextlib import suppress, redirect_stdout, ExitStack, nullcontext
>>> import io, os
>>> with suppress(FileNotFoundError):         # ignore one kind of exception explicitly
...     os.remove("/tmp/definitely-not-exists-12345")
...
>>> buf = io.StringIO()
>>> with redirect_stdout(buf):                # redirect print's output temporarily
...     print("captured")
...
>>> buf.getvalue()
'captured\n'
```

- `ExitStack`: a group of resources whose count is only known at run time, opening every file in a list say:

```python
import tempfile
from contextlib import ExitStack
from pathlib import Path

with tempfile.TemporaryDirectory() as d:
    paths = [Path(d) / f"part{i}.txt" for i in range(3)]
    with ExitStack() as stack:
        files = [stack.enter_context(p.open("w", encoding="utf-8")) for p in paths]
        for i, f in enumerate(files):
            f.write(f"chunk {i}\n")
    # leaving the ExitStack closes every file in reverse order
    assert all(f.closed for f in files)
    assert (Path(d) / "part2.txt").read_text(encoding="utf-8") == "chunk 2\n"
```

- `nullcontext()`: using a context manager conditionally, `with (lock if threaded else nullcontext()):`.
- `contextlib.chdir(path)`: change the working directory temporarily.
- `contextlib.closing(obj)`: give an object that has only a `close()` method support for `with`.

!!! interview "Answering in an interview"
    On exceptions: keep the `try` block small and catch only specific exceptions, since a bare `except:` swallows `KeyboardInterrupt` and `SystemExit` too; `else` runs when nothing was raised and `finally` always runs (with no `return` inside, which swallows the exception); converting an exception uses `raise NewError(...) from e` to keep the chain of causes; either handle it or re-raise it. Resource management always uses `with`: `__exit__` returning a true value swallows the exception; in a `@contextmanager`, before the `yield` is the acquisition and after it the release, which belongs in a `finally`. Several exceptions arising at once in concurrent code are handled with `ExceptionGroup` and `except*`.

## Exercises {#练习}

**1. A database transaction.** Write a context manager `transaction(conn)`: it does nothing on entry, calls `conn.commit()` on a normal exit, and on an exception calls `conn.rollback()` and lets the exception propagate. Verify it with an in-memory `sqlite3` database.

??? success "Answer"
    ```python
    import sqlite3
    from contextlib import contextmanager

    @contextmanager
    def transaction(conn):
        try:
            yield conn
        except BaseException:
            conn.rollback()
            raise
        else:
            conn.commit()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (x INTEGER)")

    with transaction(conn):
        conn.execute("INSERT INTO t VALUES (1)")

    try:
        with transaction(conn):
            conn.execute("INSERT INTO t VALUES (2)")
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert conn.execute("SELECT x FROM t").fetchall() == [(1,)]
    ```

    Catching `BaseException` is reasonable here: even a Ctrl+C has to roll back, and the immediate `raise` re-raises it without swallowing it. In fact `sqlite3.Connection` itself supports `with conn:` with the same semantics.

**2. Designing a configuration loader's exceptions.** Write `load_config(text)` parsing multi-line `key=value` text, requiring `port` to be present and an integer from 1 to 65535. Define a `ConfigError` base class with `MissingKey` and `InvalidValue` subclasses; use `raise ... from` to keep the cause when a conversion fails; and include the line number or the key's name in the message.

??? success "Answer"
    ```python
    class ConfigError(Exception):
        pass

    class MissingKey(ConfigError):
        pass

    class InvalidValue(ConfigError):
        pass

    def load_config(text):
        cfg = {}
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition("=")
            if not sep:
                raise InvalidValue(f"line {lineno}: expected key=value, got {line!r}")
            cfg[key.strip()] = value.strip()

        if "port" not in cfg:
            raise MissingKey("'port' is required")
        try:
            port = int(cfg["port"])
        except ValueError as e:
            raise InvalidValue(f"port must be an integer, got {cfg['port']!r}") from e
        if not 1 <= port <= 65535:
            raise InvalidValue(f"port must be 1-65535, got {port}")
        cfg["port"] = port
        return cfg

    assert load_config("host = db\nport = 5432")["port"] == 5432
    for bad, exc in [("host=db", MissingKey), ("port=http", InvalidValue), ("oops", InvalidValue)]:
        try:
            load_config(bad)
        except exc as e:
            assert isinstance(e, ConfigError)
        else:
            raise AssertionError(f"{bad!r} should fail")
    ```

**3. A timer with a slowness warning.** Extend the `Timer` above: it takes a `warn_after` in seconds and warns through `warnings.warn` when the block takes longer; an exception in the block has to propagate normally.

??? success "Answer"
    ```python
    import time
    import warnings

    class Timer:
        def __init__(self, label="block", warn_after=None):
            self.label, self.warn_after = label, warn_after

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb):
            self.elapsed = time.perf_counter() - self.start
            if self.warn_after is not None and self.elapsed > self.warn_after:
                warnings.warn(f"{self.label} took {self.elapsed:.3f}s (> {self.warn_after}s)", stacklevel=2)
            return False

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with Timer("sleep", warn_after=0.01):
            time.sleep(0.02)
    assert "sleep took" in str(caught[0].message)
    ```

## Summary {#小结}

- [x] Catch only specific exception types; keep the `try` block small; no `return` in a `finally`.
- [x] Convert an exception with `raise ... from e`; add context with `add_note`.
- [x] A library needs its own exception hierarchy; a custom exception remembers to call `super().__init__`.
- [x] Either handle an exception or re-raise it, never swallow it.
- [x] Resource management always uses `with`; writing your own context manager prefers `@contextmanager`, with the cleanup in a `finally`.
