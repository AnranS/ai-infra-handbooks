# 用 pytest 写测试

<p class="lead">测试不是负担，而是让你敢于修改代码的底气。pytest 是 Python 事实上的标准测试框架：用普通的 <code>assert</code> 写断言，用 fixture 管理测试数据和资源，用参数化覆盖大量用例。这一章用一个小项目把这些串起来。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. pytest 怎么发现测试？为什么用普通的 `assert` 就能看到详细的失败信息？
    2. fixture 里的 `yield` 起什么作用？`scope="session"` 意味着什么？
    3. 怎么用一个测试函数覆盖 20 组输入输出？
    4. `mock.patch("a.b.func")` 里的路径应该写"定义的地方"还是"使用的地方"？
    5. 什么样的代码容易测试？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 默认收集当前目录下名字以 `test_` 开头的文件里以 `test_` 开头的函数（和 `Test` 开头的类里的方法）。pytest 在导入测试模块时改写了 `assert` 语句，失败时能把表达式里每一部分的值都显示出来。
    2. `yield` 之前准备资源、`yield` 出去的值交给测试使用、之后的代码在测试结束时做清理（不管测试成功还是失败）。`scope="session"` 表示整个测试会话只创建一次，所有测试共用。
    3. `@pytest.mark.parametrize("x, expected", [(1, 2), (3, 4), ...])`，每组参数生成一个独立的测试用例。
    4. 使用的地方：patch 替换的是某个模块里的名字，被测代码是通过它所在模块的名字去查找的（`from a.b import func` 之后，要 patch 使用它的那个模块里的 `func`）。
    5. 输入输出清楚、没有隐藏的全局状态和副作用、依赖从外面传进来（依赖注入）、小而单一职责的函数；难以测试往往说明设计需要调整。

## 从一个函数开始

本章的示例是一个小项目 `shop`，采用 src 布局（见[项目与工具链](tooling.md)）。先有一个被测函数：

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

测试文件放在 `tests/` 下，文件名以 `test_` 开头，测试函数也以 `test_` 开头：

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

运行：

```bash
uv run pytest                 # 运行所有测试
uv run pytest -q              # 简洁输出
uv run pytest tests/test_text.py::test_basic   # 只跑一个测试
uv run pytest -k "slugify and not empty"       # 按名字筛选
uv run pytest -x              # 遇到第一个失败就停止
uv run pytest --lf            # 只重跑上次失败的测试
uv run pytest -vv             # 失败时显示更详细的差异
uv run pytest -s              # 不捕获输出，能看到 print
uv run pytest --pdb           # 失败时进入调试器
```

pytest 会**改写 `assert` 语句**，失败时把表达式里每个值都打印出来，所以不需要 `assertEqual` 之类的方法：

```text
    def test_basic():
>       assert slugify("Hello World") == "hello_world"
E       AssertionError: assert 'hello-world' == 'hello_world'
E         - hello_world
E         ?      ^
E         + hello-world
E         ?      ^
```

## 参数化：一个测试，很多用例

上面的 `@pytest.mark.parametrize` 让同一个测试函数用多组数据各运行一次，每组都是一个独立的测试，失败时能精确看到是哪一组。`ids` 给每组起个可读的名字。

参数化可以叠加，得到笛卡尔积；也可以用 `pytest.param(..., marks=pytest.mark.xfail)` 给某一组单独打标记。

## 异常、浮点数与标记

```py title="tests/test_misc.py"
import os
import sys

import pytest


def test_float_math():
    assert 0.1 + 0.2 == pytest.approx(0.3)                    # 浮点数不要用 == 直接比较
    assert [0.1 + 0.2, 1 / 3] == pytest.approx([0.3, 0.333], abs=1e-3)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX only")
def test_posix_path_separator():
    assert os.sep == "/"


@pytest.mark.xfail(reason="float cannot represent 2.675 exactly", strict=True)
def test_known_rounding_surprise():
    assert round(2.675, 2) == 2.68
```

- `pytest.raises(Exc, match=正则)` 断言会抛出某个异常，`match` 检查异常消息。
- `pytest.approx` 比较浮点数，可以设置相对或绝对误差。
- `skip`/`skipif` 跳过测试；`xfail` 标记"已知会失败"，`strict=True` 表示如果它意外通过了，也算失败（说明 bug 修好了，该去掉标记了）。

## Fixture：准备测试所需的东西

测试经常需要一些准备工作：一个数据库连接、一组测试数据、一个临时目录。fixture 把这些准备工作抽出来，测试函数只要**在参数里写上 fixture 的名字**，pytest 就会自动创建并传进来。

被测的代码：

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

fixture 通常放在 `tests/conftest.py` 里，同目录及子目录下的所有测试都能直接使用，不需要 import：

```py title="tests/conftest.py"
import sqlite3

import pytest

from shop.inventory import Inventory


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    yield connection            # yield 之前是准备，测试在这里运行
    connection.close()          # yield 之后是清理，即使测试失败也会执行


@pytest.fixture
def inventory(conn):            # fixture 可以依赖别的 fixture
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

每个测试都会拿到**全新的** `inventory`，测试之间互不影响。这一点非常重要：测试必须能以任意顺序、单独或一起运行。

### fixture 的作用域

`@pytest.fixture(scope=...)` 控制 fixture 多久创建一次：

| scope | 创建频率 | 典型用途 |
| --- | --- | --- |
| `function`（默认） | 每个测试一次 | 大多数情况，保证隔离 |
| `module` | 每个测试文件一次 | 创建代价较高、测试不会修改它的资源 |
| `session` | 整个测试运行一次 | 启动测试数据库容器、加载大模型 |

作用域越大越快，但隔离性越差。共享的 fixture 如果会被测试修改，就会出现"单独跑能过、一起跑就挂"的问题。

### 内置的实用 fixture

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
    path = write_report(["a", "b"], tmp_path / "out")      # tmp_path：每个测试独立的临时目录
    assert path.read_text(encoding="utf-8") == "a\nb\n"
    assert capsys.readouterr().out == "wrote 2 lines to report.txt\n"   # capsys：捕获输出


def test_currency_defaults_to_cny(monkeypatch):
    monkeypatch.delenv("SHOP_CURRENCY", raising=False)    # monkeypatch：测试结束后自动还原
    assert currency() == "CNY"


def test_currency_from_env(monkeypatch):
    monkeypatch.setenv("SHOP_CURRENCY", "USD")
    assert currency() == "USD"
```

| fixture | 用途 |
| --- | --- |
| `tmp_path` | 独立的临时目录（`pathlib.Path`），测试完自动清理 |
| `monkeypatch` | 临时修改环境变量、对象属性、字典项、`sys.path`，测试结束自动还原 |
| `capsys` / `capfd` | 捕获 `stdout`、`stderr` |
| `caplog` | 捕获日志记录，可以断言打出了哪些日志 |
| `request` | 在 fixture 里获取当前测试的信息 |

## 替换外部依赖：mock 与 fake

单元测试不应该真的访问网络、真的发邮件、真的依赖当前时间。有两种办法隔离这些依赖。

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

### 办法一：`unittest.mock.patch`

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

!!! warning "patch 使用的地方，不是定义的地方"
    `patch` 替换的是**某个模块里的某个名字**。如果 `shop/rates.py` 写的是 `from shop.http import get_json`，那么 `rates` 模块里有一个自己的 `get_json` 名字，你要 patch 的是 `"shop.rates.get_json"`，而不是 `"shop.http.get_json"`。

    另外，`patch(..., autospec=True)` 会让 mock 对象拥有和原函数一样的签名，调用参数写错时测试会失败，而不是悄悄通过。

### 办法二：依赖注入 + fake

mock 用多了，测试会和实现细节绑得很紧：重构一下内部调用方式，测试就挂了。更好的设计是**把依赖作为参数传进来**：

```py
def convert(amount, base, target, *, get_json=get_json):
    ...

def test_convert():
    rates = {"https://rates.example.com/USD/CNY": {"rate": 7.1}}
    assert convert(10, "USD", "CNY", get_json=rates.__getitem__) == 71.0
```

对于更复杂的依赖（数据库、消息队列），写一个简单的**内存版实现**（fake），比如[协议与抽象基类](../types/protocols.md#练习)里的 `FakeNotifier`。fake 的行为接近真实实现，测试读起来也更自然。

**当前时间**也是一种外部依赖。`datetime.now()` 直接写死在函数里，就没法测试"过期"的逻辑。把 `now` 作为参数，或者注入一个时钟函数。

## 其他常用工具

- **覆盖率**：`uv add --dev pytest-cov`，然后 `uv run pytest --cov=shop --cov-report=term-missing`，会列出哪些行没有被测试执行到。覆盖率是发现遗漏的工具，不是目标本身：100% 覆盖不代表测得好。
- **并行运行**：`pytest-xdist` 提供 `pytest -n auto`，用多个进程并行跑测试。
- **异步测试**：`pytest-asyncio`（`@pytest.mark.asyncio`），或者 anyio 自带的 pytest 插件（`@pytest.mark.anyio`）。
- **基于属性的测试**：[Hypothesis](https://hypothesis.readthedocs.io/) 自动生成大量输入，找出你没想到的边界情况：

```py
from hypothesis import given, strategies as st

@given(st.text())
def test_slugify_is_idempotent(s):
    once = slugify(s)
    assert slugify(once) == once          # 对任意字符串都应该成立的"性质"
```

- **doctest**：文档字符串里的 `>>>` 示例也可以当测试运行：`pytest --doctest-modules`。

## 怎样写出好测试

- **一个测试只验证一件事**，测试名说清楚是什么行为：`test_cannot_remove_more_than_stock` 好过 `test_remove_2`。
- **准备、执行、断言**（Arrange / Act / Assert）三段分明。
- **测试行为，不测实现**：通过公开的接口验证结果，不要断言内部调用了哪个私有方法。
- **测试要快、要稳定**：不依赖网络、不依赖执行顺序、不 `sleep`。慢的集成测试用标记分开（`@pytest.mark.slow`，并在配置里注册这个标记）。
- **先写一个失败的测试再修 bug**：确认测试真的能捕获这个 bug，也防止它再次出现。
- **难测的代码往往是设计有问题**：函数做了太多事、依赖写死在内部、依赖全局状态。把纯逻辑和 I/O 分开，测试会简单得多。

!!! interview "面试怎么答"
    测试题：pytest 按 `test_` 前缀发现测试，改写 `assert` 语句所以失败时能显示两边的值；`parametrize` 让一个测试覆盖多组用例；fixture 用 `yield` 分隔准备和清理，`scope="session"` 表示整个测试会话只建一次，共享的放 `conftest.py`；`pytest.raises`、`pytest.approx` 处理异常和浮点数。`mock.patch` 要打在"使用的地方"（被测模块导入进来的名字），而不是定义的地方；更好的办法是依赖注入，让外部依赖可以换成 fake。难测的代码通常说明设计要调整。给推理引擎写测试的一个常用思路：贪心解码逐 token 比对参考实现。

## 练习

**1. 为 `parse_duration` 写测试。** 函数把 `"1h30m"`、`"45s"`、`"2h"`、`"1h2m3s"` 这样的字符串转换成秒数，空字符串或格式错误时抛 `ValueError`。先写测试（用参数化覆盖正常用例和错误用例），再实现函数，让测试通过。

??? success "参考答案"
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

    注意错误用例里的 `"1h 30m"`（有空格）、`"-5s"`、`"1.5h"`：先想清楚"什么不应该被接受"，往往比实现本身更能发现需求里的模糊之处。

**2. 测试和时间有关的逻辑。** 有一个判断令牌是否过期的函数。改造它，让测试不依赖真实的当前时间，并测试"刚好到期"的边界。

```py
def is_expired(token):
    return datetime.now(UTC) >= token.expires_at
```

??? success "参考答案"
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

    给 `now` 一个默认值，调用方的代码不用改，测试却可以精确控制时间。另一种做法是用 `time-machine` 或 `freezegun` 这类库冻结时间，但参数注入更简单、更明确。

## 小结

- [x] 测试文件和函数以 `test_` 开头，用普通 `assert` 断言。
- [x] 用 `parametrize` 覆盖多组用例，用 `pytest.raises`、`pytest.approx` 处理异常和浮点数。
- [x] 用 fixture 准备资源，`yield` 之后写清理；共享的 fixture 放 `conftest.py`。
- [x] 善用 `tmp_path`、`monkeypatch`、`capsys`、`caplog`。
- [x] 外部依赖用 patch 或 fake 隔离；patch 使用的地方；优先考虑依赖注入。
- [x] 难测的代码通常说明设计需要调整。
