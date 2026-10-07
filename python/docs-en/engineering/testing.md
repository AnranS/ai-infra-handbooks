# Writing tests with pytest

<p class="lead">Tests are not a burden but what gives you the nerve to change the code. pytest is Python's de facto standard test framework: assertions are plain <code>assert</code> statements, fixtures manage test data and resources, and parametrization covers many cases at once. This chapter ties these together through a small project.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How does pytest find tests? Why does a plain `assert` give a detailed failure message?
    2. What does a fixture's `yield` do? What does `scope="session"` mean?
    3. How do you cover 20 input-output pairs with one test function?
    4. Should the path in `mock.patch("a.b.func")` be where it is defined or where it is used?
    5. What kind of code is easy to test?

??? success "Answers (try it yourself first, then expand)"
    1. By default it collects functions starting with `test_` in files starting with `test_` under the current directory (and methods in classes starting with `Test`). pytest rewrites the `assert` statements as it imports the test module, so a failure can show the value of every part of the expression.
    2. Before the `yield` prepares the resource, the value yielded goes to the test, and the code after it cleans up when the test ends (whether it passed or failed). `scope="session"` means it is created once for the whole test session and shared by every test.
    3. `@pytest.mark.parametrize("x, expected", [(1, 2), (3, 4), ...])`, where each set of parameters becomes its own test case.
    4. Where it is used: patching replaces a name in some module, and the code under test looks it up through the name in its own module (after `from a.b import func`, patch the `func` in the module that uses it).
    5. Code with clear inputs and outputs, no hidden global state or side effects, dependencies passed in from outside (dependency injection), and small functions with one responsibility; being hard to test usually means the design needs adjusting.

## Starting from one function {#从一个函数开始}

This chapter's example is a small project called `shop` in the src layout (see [projects and the toolchain](tooling.md)). First the function under test:

```py title="src/shop/text.py"
import re
import unicodedata


def slugify(text: str, max_len: int = 50) -> str:
    """Convert text into a URL-friendly slug."""
    if max_len <= 0:
        raise ValueError(f"max_len must be positive, got {max_len}")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[-\s_]+", "-", text).strip("-")
    return slug[:max_len].rstrip("-")
```

The test files go under `tests/`, with names starting with `test_`, and so do the test functions:

```py title="tests/test_text.py"
import pytest

from shop.text import slugify


def test_basic():
    assert slugify("Hello World") == "hello-world"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Leading and trailing  ", "leading-and-trailing"),
        ("Python 3.14 is out!", "python-314-is-out"),
        ("Crème brûlée", "creme-brulee"),
        ("a---b___c", "a-b-c"),
        ("", ""),
    ],
    ids=["strip", "punctuation", "accents", "separators", "empty"],
)
def test_slugify_cases(raw, expected):
    assert slugify(raw) == expected


def test_truncated_slug_does_not_end_with_dash():
    assert slugify("hello world again", max_len=6) == "hello"


def test_rejects_non_positive_max_len():
    with pytest.raises(ValueError, match="must be positive"):
        slugify("x", max_len=0)
```

Running it:

```bash
uv run pytest                 # run every test
uv run pytest -q              # concise output
uv run pytest tests/test_text.py::test_basic   # run one test only
uv run pytest -k "slugify and not empty"       # filter by name
uv run pytest -x              # stop at the first failure
uv run pytest --lf            # rerun only last run's failures
uv run pytest -vv             # a more detailed diff on failure
uv run pytest -s              # do not capture output, so print is visible
uv run pytest --pdb           # drop into the debugger on failure
```

pytest **rewrites the `assert` statements** and prints every value in the expression on failure, so no `assertEqual` and the like are needed:

```text
    def test_basic():
>       assert slugify("Hello World") == "hello_world"
E       AssertionError: assert 'hello-world' == 'hello_world'
E         - hello_world
E         ?      ^
E         + hello-world
E         ?      ^
```

## Parametrization: one test, many cases {#参数化一个测试很多用例}

The `@pytest.mark.parametrize` above runs the same test function once per set of data, each a separate test, so a failure says exactly which set. `ids` gives each set a readable name.

Parametrizations can be stacked, giving the Cartesian product; and `pytest.param(..., marks=pytest.mark.xfail)` marks one set on its own.

## Exceptions, floats and markers {#异常浮点数与标记}

```py title="tests/test_misc.py"
import os
import sys

import pytest


def test_float_math():
    assert 0.1 + 0.2 == pytest.approx(0.3)                    # do not compare floats with == directly
    assert [0.1 + 0.2, 1 / 3] == pytest.approx([0.3, 0.333], abs=1e-3)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX only")
def test_posix_path_separator():
    assert os.sep == "/"


@pytest.mark.xfail(reason="float cannot represent 2.675 exactly", strict=True)
def test_known_rounding_surprise():
    assert round(2.675, 2) == 2.68
```

- `pytest.raises(Exc, match=regex)` asserts that an exception is raised, with `match` checking its message.
- `pytest.approx` compares floats, with a relative or absolute tolerance.
- `skip`/`skipif` skip a test; `xfail` marks one as known to fail, and `strict=True` makes an unexpected pass a failure too (the bug is fixed and the marker should go).

## Fixtures: preparing what a test needs {#fixture准备测试所需的东西}

Tests often need something prepared: a database connection, a set of test data, a temporary directory. A fixture pulls that preparation out, and a test function only has to **name the fixture among its parameters** for pytest to create it and pass it in.

The code under test:

```py title="src/shop/inventory.py"
import sqlite3


class Inventory:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        conn.execute("CREATE TABLE IF NOT EXISTS stock (sku TEXT PRIMARY KEY, qty INTEGER NOT NULL)")

    def add(self, sku: str, qty: int) -> None:
        if qty <= 0:
            raise ValueError("qty must be positive")
        self.conn.execute(
            "INSERT INTO stock (sku, qty) VALUES (?, ?) "
            "ON CONFLICT(sku) DO UPDATE SET qty = qty + excluded.qty",
            (sku, qty),
        )

    def remove(self, sku: str, qty: int) -> None:
        current = self.quantity(sku)
        if qty > current:
            raise ValueError(f"only {current} of {sku} left")
        self.conn.execute("UPDATE stock SET qty = qty - ? WHERE sku = ?", (qty, sku))

    def quantity(self, sku: str) -> int:
        row = self.conn.execute("SELECT qty FROM stock WHERE sku = ?", (sku,)).fetchone()
        return row[0] if row else 0
```

Fixtures usually go in `tests/conftest.py`, where every test in that directory and below can use them without importing:

```py title="tests/conftest.py"
import sqlite3

import pytest

from shop.inventory import Inventory


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    yield connection            # before the yield is the preparation, and the test runs here
    connection.close()          # after the yield is the cleanup, which runs even when the test fails


@pytest.fixture
def inventory(conn):            # a fixture can depend on another fixture
    inv = Inventory(conn)
    inv.add("apple", 10)
    return inv
```

```py title="tests/test_inventory.py"
import pytest


def test_add_accumulates(inventory):
    inventory.add("apple", 5)
    assert inventory.quantity("apple") == 15


def test_cannot_remove_more_than_stock(inventory):
    with pytest.raises(ValueError, match="only 10 of apple left"):
        inventory.remove("apple", 11)


def test_unknown_sku_has_zero_quantity(inventory):
    assert inventory.quantity("pear") == 0
```

Every test gets a **fresh** `inventory` and no test affects another. That matters a great deal: tests have to run in any order, alone or together.

### A fixture's scope {#fixture-的作用域}

`@pytest.fixture(scope=...)` controls how often the fixture is created:

| scope | How often | Typical use |
| --- | --- | --- |
| `function` (the default) | once per test | most cases, keeping them isolated |
| `module` | once per test file | a resource that is expensive to create and that the tests do not modify |
| `session` | once per run | starting a test database container, loading a large model |

A wider scope is faster and less isolated. A shared fixture that the tests modify gives the "passes alone, fails together" problem.

### The built-in fixtures worth knowing {#内置的实用-fixture}

```py title="src/shop/report.py"
import os
from pathlib import Path


def write_report(lines: list[str], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} lines to {path.name}")
    return path


def currency() -> str:
    return os.environ.get("SHOP_CURRENCY", "CNY")
```

```py title="tests/test_report.py"
from shop.report import currency, write_report


def test_write_report(tmp_path, capsys):
    path = write_report(["a", "b"], tmp_path / "out")      # tmp_path: a temporary directory of its own per test
    assert path.read_text(encoding="utf-8") == "a\nb\n"
    assert capsys.readouterr().out == "wrote 2 lines to report.txt\n"   # capsys: captures the output


def test_currency_defaults_to_cny(monkeypatch):
    monkeypatch.delenv("SHOP_CURRENCY", raising=False)    # monkeypatch: restored automatically when the test ends
    assert currency() == "CNY"


def test_currency_from_env(monkeypatch):
    monkeypatch.setenv("SHOP_CURRENCY", "USD")
    assert currency() == "USD"
```

| Fixture | Use |
| --- | --- |
| `tmp_path` | a temporary directory of its own (a `pathlib.Path`), cleaned up afterwards |
| `monkeypatch` | temporarily change an environment variable, an object's attribute, a dict item or `sys.path`, restored when the test ends |
| `capsys` / `capfd` | capture `stdout` and `stderr` |
| `caplog` | capture log records, so what was logged can be asserted |
| `request` | information about the current test, from inside a fixture |

## Replacing external dependencies: mocks and fakes {#替换外部依赖mock-与-fake}

A unit test should not really reach the network, really send an email or really depend on the current time. There are two ways to isolate such dependencies.

```py title="src/shop/rates.py"
import json
import urllib.request


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.load(resp)


def convert(amount: float, base: str, target: str) -> float:
    if base == target:
        return amount
    data = get_json(f"https://rates.example.com/{base}/{target}")
    return round(amount * data["rate"], 2)
```

### The first way: `unittest.mock.patch` {#办法一unittestmockpatch}

```py title="tests/test_rates.py"
from unittest.mock import patch

import pytest

from shop import rates


def test_convert_uses_fetched_rate():
    with patch("shop.rates.get_json", return_value={"rate": 7.1}) as fake:
        assert rates.convert(10, "USD", "CNY") == 71.0
    fake.assert_called_once_with("https://rates.example.com/USD/CNY")


def test_same_currency_skips_network():
    with patch("shop.rates.get_json") as fake:
        assert rates.convert(10, "CNY", "CNY") == 10
    fake.assert_not_called()


def test_network_errors_propagate(monkeypatch):
    def boom(url):
        raise TimeoutError("rate service too slow")

    monkeypatch.setattr(rates, "get_json", boom)
    with pytest.raises(TimeoutError):
        rates.convert(1, "USD", "EUR")
```

!!! warning "Patch where it is used, not where it is defined"
    `patch` replaces **a name in a module**. If `shop/rates.py` says `from shop.http import get_json`, then the `rates` module has its own name `get_json`, and what you patch is `"shop.rates.get_json"` and not `"shop.http.get_json"`.

    Besides, `patch(..., autospec=True)` gives the mock the original function's signature, so wrong arguments fail the test rather than passing quietly.

### The second way: dependency injection plus a fake {#办法二依赖注入--fake}

Used heavily, mocks tie the tests tightly to the implementation: refactor an internal call and the tests break. The better design **passes the dependency in as an argument**:

```py
def convert(amount, base, target, *, get_json=get_json):
    ...

def test_convert():
    rates = {"https://rates.example.com/USD/CNY": {"rate": 7.1}}
    assert convert(10, "USD", "CNY", get_json=rates.__getitem__) == 71.0
```

For a more complicated dependency (a database, a message queue), write a simple **in-memory implementation** (a fake), like the `FakeNotifier` in [protocols and abstract base classes](../types/protocols.md#练习). A fake behaves close to the real thing and the tests read more naturally.

**The current time** is an external dependency too. With `datetime.now()` hard-coded inside a function, the "expired" logic cannot be tested. Take `now` as an argument, or inject a clock function.

## The other tools in common use {#其他常用工具}

- **Coverage**: `uv add --dev pytest-cov`, then `uv run pytest --cov=shop --cov-report=term-missing` lists the lines no test reached. Coverage is a tool for finding gaps and not a goal in itself: 100% coverage does not mean the tests are good.
- **Running in parallel**: `pytest-xdist` provides `pytest -n auto`, running the tests over several processes.
- **Asynchronous tests**: `pytest-asyncio` (`@pytest.mark.asyncio`), or anyio's own pytest plugin (`@pytest.mark.anyio`).
- **Property-based testing**: [Hypothesis](https://hypothesis.readthedocs.io/) generates many inputs automatically and finds the boundary cases you did not think of:

```py
from hypothesis import given, strategies as st

@given(st.text())
def test_slugify_is_idempotent(s):
    once = slugify(s)
    assert slugify(once) == once          # a "property" that should hold for any string
```

- **doctest**: the `>>>` examples in docstrings can run as tests too: `pytest --doctest-modules`.

## How to write good tests {#怎样写出好测试}

- **One test verifies one thing**, and its name says which behaviour: `test_cannot_remove_more_than_stock` beats `test_remove_2`.
- **Arrange, Act, Assert** in three clear parts.
- **Test behaviour, not implementation**: verify the result through the public interface rather than asserting which private method was called.
- **Tests have to be fast and stable**: no network, no dependence on order, no `sleep`. Keep slow integration tests apart with a marker (`@pytest.mark.slow`, registered in the configuration).
- **Write a failing test before fixing a bug**: it confirms the test really catches it and keeps it from coming back.
- **Code that is hard to test usually has a design problem**: the function does too much, a dependency is hard-coded inside it, or it relies on global state. Separate the pure logic from the I/O and testing gets much simpler.

!!! interview "Answering in an interview"
    On testing: pytest finds tests by the `test_` prefix and rewrites the `assert` statements, so a failure shows both sides' values; `parametrize` covers many cases with one test; a fixture's `yield` separates preparation from cleanup, `scope="session"` creates it once for the whole session, and shared ones go in `conftest.py`; `pytest.raises` and `pytest.approx` handle exceptions and floats. `mock.patch` goes "where it is used" (the name the module under test imported) and not where it is defined; the better way is dependency injection, so an external dependency can be swapped for a fake. Code that is hard to test usually says the design needs adjusting. One common approach to testing an inference engine: compare greedy decoding token by token against a reference implementation.

## Exercises {#练习}

**1. Write tests for `parse_duration`.** The function turns strings like `"1h30m"`, `"45s"`, `"2h"` and `"1h2m3s"` into seconds, raising `ValueError` on an empty string or a bad format. Write the tests first (parametrized over the valid and invalid cases), then implement the function and make them pass.

??? success "Answer"
    ```py title="tests/test_duration.py"
    import re

    import pytest

    _PART = re.compile(r"(\d+)([hms])")
    _FULL = re.compile(r"(?:\d+[hms])+")
    _SECONDS = {"h": 3600, "m": 60, "s": 1}


    def parse_duration(text: str) -> int:
        if not _FULL.fullmatch(text):
            raise ValueError(f"invalid duration: {text!r}")
        return sum(int(n) * _SECONDS[unit] for n, unit in _PART.findall(text))


    @pytest.mark.parametrize(
        ("text", "seconds"),
        [("45s", 45), ("2h", 7200), ("1h30m", 5400), ("1h2m3s", 3723), ("90m", 5400)],
    )
    def test_valid(text, seconds):
        assert parse_duration(text) == seconds


    @pytest.mark.parametrize("text", ["", "1x", "h1", "1h 30m", "-5s", "1.5h"])
    def test_invalid(text):
        with pytest.raises(ValueError, match="invalid duration"):
            parse_duration(text)
    ```

    Note the invalid cases `"1h 30m"` (with a space), `"-5s"` and `"1.5h"`: thinking through "what should not be accepted" first often uncovers more of the requirement's vagueness than the implementation does.

**2. Testing logic that depends on time.** A function decides whether a token has expired. Rework it so the test does not depend on the real current time, and test the boundary of "expiring exactly now".

```py
def is_expired(token):
    return datetime.now(UTC) >= token.expires_at
```

??? success "Answer"
    ```py title="tests/test_token.py"
    from dataclasses import dataclass
    from datetime import UTC, datetime, timedelta


    @dataclass(frozen=True)
    class Token:
        value: str
        expires_at: datetime


    def is_expired(token: Token, now: datetime | None = None) -> bool:
        now = now or datetime.now(UTC)
        return now >= token.expires_at


    T0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    TOKEN = Token("abc", expires_at=T0)


    def test_not_expired_before_deadline():
        assert not is_expired(TOKEN, now=T0 - timedelta(seconds=1))


    def test_expired_exactly_at_deadline():
        assert is_expired(TOKEN, now=T0)


    def test_uses_current_time_by_default():
        fresh = Token("x", expires_at=datetime.now(UTC) + timedelta(hours=1))
        assert not is_expired(fresh)
    ```

    Giving `now` a default leaves the callers' code unchanged while the tests control the time exactly. The other approach is freezing time with a library like `time-machine` or `freezegun`, but injecting the argument is simpler and more explicit.

## Summary {#小结}

- [x] Test files and functions start with `test_`, and assertions are plain `assert`.
- [x] `parametrize` covers many cases, and `pytest.raises` and `pytest.approx` handle exceptions and floats.
- [x] Fixtures prepare resources, with cleanup after the `yield`; shared ones go in `conftest.py`.
- [x] Make good use of `tmp_path`, `monkeypatch`, `capsys` and `caplog`.
- [x] Isolate external dependencies with a patch or a fake; patch where it is used; prefer dependency injection.
- [x] Code that is hard to test usually says the design needs adjusting.
