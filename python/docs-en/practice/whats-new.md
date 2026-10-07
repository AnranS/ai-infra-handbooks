# What is new in recent versions

<p class="lead">Python releases a new version every October. Many an experienced developer's knowledge stopped a few years ago, and the code they write runs while missing the shorter, safer newer forms. This page lists the changes from 3.10 to 3.14 most worth knowing, version by version, and ends with a preview of 3.15.</p>

!!! note "The support cycle"
    Each version gets 2 years of full support plus 3 more of security fixes, 5 in all. As of September 2026, 3.10 reaches its end of life in October 2026, so **a new project should start on 3.13 or 3.14**.

## Python 3.10 (2021) {#python-3102021}

**Structural pattern matching**: `match`/`case`, see [structural pattern matching](../core/pattern-matching.md).

**The new spelling of a union**:

```py
def f(x: int | None) -> str | bytes: ...     # in place of Optional[int] and Union[str, bytes]
isinstance(x, int | str)                     # isinstance accepts it too
```

**Parenthesized context managers**, which allow line breaks:

```py
with (
    open("a") as fa,
    open("b") as fb,
):
    ...
```

**`zip(strict=True)`**, **`itertools.pairwise`**, and markedly better syntax error messages.

## Python 3.11 (2022) {#python-3112022}

**A large performance gain**: about 25% faster than 3.10 on average.

**Exception groups and `except*`**, **`asyncio.TaskGroup`**, **`asyncio.timeout`**: see [exceptions and context managers](../core/errors-context.md#同时处理多个异常exceptiongroup) and [asyncio](../concurrency/asyncio.md).

**`exception.add_note()`**: attaching context to an exception.

**Tracebacks pointing at the expression**:

```text
Traceback (most recent call last):
  File "demo.py", line 2, in <module>
    x = data["user"]["address"]["city"]
        ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^
TypeError: 'NoneType' object is not subscriptable
```

**New in the standard library**: `tomllib` (reading TOML); **`typing.Self`**, **`enum.StrEnum`**; and the `datetime.UTC` alias.

## Python 3.12 (2023) {#python-3122023}

**The new type parameter syntax** (PEP 695):

```py
def first[T](xs: list[T]) -> T: ...
class Box[T]:
    def __init__(self, item: T) -> None: ...
type Pair[T] = tuple[T, T]
```

See [type annotations](../types/typing.md#泛型).

**Looser rules for f-strings** (PEP 701): an f-string can use the same quotes as the outside, can contain backslashes and comments, and can nest arbitrarily:

```py
f"{", ".join(names)}"            # before 3.12 the inner and outer quotes could not match
```

**The rest**: `itertools.batched`, `Path.walk()`, `typing.override`, the low-overhead monitoring API `sys.monitoring` (PEP 669), friendlier error messages (a `NameError` suggesting "did you forget to import it"), and a GIL per subinterpreter (PEP 684); `distutils` was removed.

## Python 3.13 (2024) {#python-3132024}

**A new interactive interpreter (REPL)**: multi-line editing, colour, history browsing, `help` and `exit` without parentheses, and ++f3++ for paste mode.

**The free-threaded build (experimental)**: a build that runs without the GIL (`python3.13t`), see [threads, processes and the GIL](../concurrency/threads-processes.md#gil全局解释器锁).

**An experimental JIT compiler** (which has to be enabled at build time).

**`locals()`'s semantics defined** (PEP 667), making debuggers and tools behave predictably.

**The type system**: `typing.TypeIs`, `TypedDict`'s `ReadOnly`, defaults for type parameters (PEP 696), and the `warnings.deprecated` decorator.

**The rest**: `copy.replace()`, `itertools.batched(strict=True)`, an error message suggesting "your script has the same name as a standard library module"; and the batch of obsolete modules listed in PEP 594 was removed (`cgi`, `telnetlib`, `crypt`, `imghdr` and others).

## Python 3.14 (2025) {#python-3142025}

**Lazily evaluated annotations** (PEP 649/749): a type annotation is computed only when it is really read, so a forward reference no longer needs quotes and `from __future__ import annotations` is no longer necessary. The new `annotationlib` module reads annotations.

**Template strings, t-strings** (PEP 750): `t"hello {name}"` gives a `Template` object rather than a string, so a library can handle the interpolations safely (escaping SQL, HTML and so on), see [the standard library in daily use](../engineering/stdlib.md#字符串格式化).

**The free-threaded build officially supported** (PEP 779): from "experimental" to "officially supported", though still not the default build.

**Multiple interpreters in the standard library** (PEP 734): the new `concurrent.interpreters` module and `concurrent.futures.InterpreterPoolExecutor`.

**`except` without parentheses** (PEP 758):

```py
except TimeoutError, ConnectionError:      # when there is no as
    ...
```

**A `return`/`break`/`continue` in a `finally` now warns** (PEP 765).

**New modules and features**:

- `compression.zstd`: the Zstandard compression algorithm
- `uuid.uuid7()`: a time-ordered UUID, which suits a database primary key
- `pathlib.Path.copy()` and `move()`
- `python -m asyncio ps PID` and `pstree PID`: the asyncio tasks of a running process
- `pdb -p PID`: attaching to a running process (PEP 768)
- syntax highlighting in the REPL; colour in the command-line output of `argparse`, `unittest`, `json` and others

**The rest**: `multiprocessing`'s default start method on Linux changed from `fork` to `forkserver`; and the official macOS and Windows installers began including the experimental JIT.

## Python 3.15 (expected October 2026) {#python-315预计-2026-年-10-月}

As this page is written, 3.15 is at the release candidate stage (3.15.0rc2) and the following can already be tried (`uv python install 3.15`):

- **Explicit lazy imports** (PEP 810): `lazy import json` imports the module only on first use, which markedly speeds up the startup of large programs and command-line tools.
- **UTF-8 mode by default** (PEP 686): `open()` and friends default to UTF-8 without an explicit encoding, finally ending the mojibake on Windows.
- **A built-in statistical sampling profiler**, `profiling.sampling`: low overhead, and able to profile a running program.

## How to keep up {#怎么跟上变化}

- When a version is released each year, read the official [What's New](https://docs.python.org/3/whatsnew/index.html) and pick out what you can use.
- Use ruff's `UP` rules (pyupgrade) to upgrade old spellings automatically: raise `target-version` and run `ruff check --fix`.
- A month or two after a release, once the main dependencies support it, upgrade the project's Python version and take the free performance along with it.
