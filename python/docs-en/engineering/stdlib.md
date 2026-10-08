# The standard library in daily use

<p class="lead">"Batteries included" is one of Python's great advantages, but many people use only a small part of it and then install a third-party package or build their own. This chapter picks the modules used most in everyday development and covers their correct use and the common mistakes.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Which module makes joining paths and listing every `.py` file in a directory easiest?
    2. What is the hidden danger in what `datetime.now()` returns?
    3. `json.dumps` turned a non-ASCII character into an escape like `\u4e2d`. How do you fix it?
    4. Why is `shell=True` discouraged when calling an external command?
    5. Should library code call `logging.basicConfig()`?

??? success "Answers (try it yourself first, then expand)"
    1. `pathlib`: `Path("a") / "b"` joins paths and `Path("src").rglob("*.py")` finds every `.py` recursively.
    2. It gives local time with no time zone, which goes wrong on a different machine, in a different zone or around daylight saving, and cannot be compared with a time that has one. Use `datetime.now(timezone.utc)`, and store in UTC in ISO 8601.
    3. Pass `ensure_ascii=False`, and specify `encoding="utf-8"` when writing the file.
    4. The command goes through the shell, so external input interpolated into it can inject any command; quotes, spaces and other special characters go wrong easily too. Pass a list of arguments: `subprocess.run(["ls", path], check=True)`.
    5. No. How logs are emitted is the application's decision; a library only records through `logging.getLogger(__name__)` (adding a `NullHandler` at most) and configures no handlers or levels.

## `pathlib`: paths as objects {#pathlib面向对象的路径}

`pathlib` can replace most uses of `os.path` entirely, and the code is shorter and harder to get wrong:

```python
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    src = root / "project" / "src"                 # join paths with /
    src.mkdir(parents=True, exist_ok=True)         # like mkdir -p

    (src / "app.py").write_text("print('hi')\n", encoding="utf-8")
    (src / "util.py").write_text("", encoding="utf-8")
    (root / "project" / "README.md").write_text("# demo\n", encoding="utf-8")

    f = src / "app.py"
    assert (f.name, f.stem, f.suffix) == ("app.py", "app", ".py")
    assert f.parent.name == "src"
    assert f.with_suffix(".txt").name == "app.txt"
    assert f.read_text(encoding="utf-8") == "print('hi')\n"

    py_files = sorted(p.name for p in (root / "project").rglob("*.py"))   # a recursive search
    assert py_files == ["app.py", "util.py"]
    assert f.relative_to(root).as_posix() == "project/src/app.py"
    assert f.exists() and f.is_file() and not f.is_dir()
```

| Need | The `pathlib` form |
| --- | --- |
| the working directory / the home directory | `Path.cwd()` / `Path.home()` |
| the script's own directory | `Path(__file__).resolve().parent` |
| walking a directory | `p.iterdir()`, `p.glob("*.csv")`, `p.rglob("*.py")`, `p.walk()` <span class="since">3.12+</span> |
| reading and writing a small file | `read_text()`, `write_text()`, `read_bytes()`, `write_bytes()` |
| opening a file | `p.open("r", encoding="utf-8")` |
| deleting a file / an empty directory | `p.unlink(missing_ok=True)` / `p.rmdir()` |
| copying, moving | `p.copy(dst)`, `p.move(dst)` <span class="since">3.14+</span>, and `shutil.copy2`, `shutil.move` before that |
| a file's information | `p.stat().st_size`, `p.stat().st_mtime` |

!!! warning "Always specify `encoding` explicitly"
    Without an encoding, `open()` and `read_text()` use the system's "locale encoding" on 3.14 and earlier, which on Windows is usually not UTF-8, so the same code produces mojibake on another machine. From 3.15 the default becomes UTF-8, but for compatibility with older versions, **always write `encoding="utf-8"` for a text file**.

## `json` {#json}

```pycon
>>> import json
>>> data = {"name": "张三", "tags": ["py", "go"], "score": 9.5, "active": True, "extra": None}
>>> json.dumps(data)
'{"name": "\\u5f20\\u4e09", "tags": ["py", "go"], "score": 9.5, "active": true, "extra": null}'
>>> print(json.dumps(data, ensure_ascii=False, indent=2))
{
  "name": "张三",
  "tags": [
    "py",
    "go"
  ],
  "score": 9.5,
  "active": true,
  "extra": null
}
>>> json.loads('{"a": [1, 2.5, "x"]}')
{'a': [1, 2.5, 'x']}
```

- `dumps`/`loads` work on strings and `dump`/`load` on file objects.
- Non-ASCII text needs `ensure_ascii=False`, with `encoding="utf-8"` when writing the file.
- Tuples become lists, dict keys become strings, and `datetime`, `set`, `Decimal` and your own objects cannot be serialized by default. The `default` argument handles them:

```pycon
>>> from datetime import date
>>> from decimal import Decimal
>>> def to_jsonable(obj):
...     if isinstance(obj, date):
...         return obj.isoformat()
...     if isinstance(obj, (set, frozenset)):
...         return sorted(obj)
...     if isinstance(obj, Decimal):
...         return str(obj)
...     raise TypeError(f"{type(obj).__name__} is not JSON serializable")
...
>>> json.dumps({"day": date(2026, 9, 24), "ids": {3, 1}, "price": Decimal("9.90")}, default=to_jsonable)
'{"day": "2026-09-24", "ids": [1, 3], "price": "9.90"}'
```

To format JSON from the command line: `python -m json.tool data.json`, or `cat data.json | python -m json` (3.14+).

## `datetime` and `zoneinfo`: handling time {#datetime-与-zoneinfo处理时间}

The most important rule about time: **do not use "naive" times**, that is a `datetime` without time zone information. `datetime.now()` returns exactly that, a naive local time, and one change to the server's time zone scrambles the data.

```pycon
>>> from datetime import datetime, timedelta, UTC
>>> from zoneinfo import ZoneInfo
>>> t = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)          # an aware time, carrying a zone
>>> t.isoformat()
'2026-09-24T08:00:00+00:00'
>>> t.astimezone(ZoneInfo("Asia/Shanghai"))
datetime.datetime(2026, 9, 24, 16, 0, tzinfo=zoneinfo.ZoneInfo(key='Asia/Shanghai'))
>>> t.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M %Z")
'2026-09-24 04:00 EDT'
>>> datetime.fromisoformat("2026-09-24T16:00:00+08:00") == t
True
>>> t + timedelta(days=1, hours=2)
datetime.datetime(2026, 9, 25, 10, 0, tzinfo=datetime.timezone.utc)
```

The recommended practice:

- the current time: `datetime.now(UTC)`.
- **store and transmit** in UTC without exception, in ISO 8601 (`isoformat()` / `fromisoformat()`).
- convert to the user's zone only when **displaying it**.
- name time zones by their IANA names, `zoneinfo.ZoneInfo("Asia/Shanghai")`, and never hard-code a `+8` offset (many regions have daylight saving).
- measure elapsed time with `time.perf_counter()` rather than subtracting `datetime`s.

## `re`: regular expressions {#re正则表达式}

```pycon
>>> import re
>>> LOG = re.compile(r"(?P<ip>\d{1,3}(?:\.\d{1,3}){3}) .* \"(?P<method>GET|POST) (?P<path>\S+)")
>>> m = LOG.search('10.0.0.1 - - [24/Sep/2026] "GET /api/users?id=3 HTTP/1.1" 200')
>>> m["ip"], m["method"], m["path"]
('10.0.0.1', 'GET', '/api/users?id=3')
>>> m.groupdict()
{'ip': '10.0.0.1', 'method': 'GET', 'path': '/api/users?id=3'}
>>> re.findall(r"\d+", "a1b22c333")
['1', '22', '333']
>>> re.sub(r"(\d{3})\d{4}(\d{4})", r"\1****\2", "call 13812345678")
'call 138****5678'
>>> re.sub(r"\d+", lambda m: str(int(m[0]) * 2), "3 apples, 10 pears")   # the replacement may be a function
'6 apples, 20 pears'
>>> bool(re.fullmatch(r"[a-z_][a-z0-9_]*", "user_id")), bool(re.fullmatch(r"[a-z_][a-z0-9_]*", "1st"))
(True, False)
```

The key points:

- always write a regular expression as a **raw string** `r"..."`, or Python processes `\d`, `\b` and the other backslashes first.
- `match` only matches from the start, `search` finds it anywhere, and `fullmatch` requires the whole string to match. Validating an input's format takes `fullmatch`.
- use **named groups** `(?P<name>...)`, which read far better than counting parentheses.
- a complicated expression takes the `re.VERBOSE` flag, which allows line breaks and comments.
- prefer the string methods for simple work: `startswith`, `endswith`, `in`, `split`, `replace` are faster and clearer.

## `logging`: logs {#logging日志}

**Do not use `print` for debugging or for recording what is happening.** `logging` has levels, a consistent format, output to a file or a log system, and its verbosity can be adjusted without changing the code.

```python
import logging
import sys

# configured once at the application's entry point (library code does not call basicConfig)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,                                  # overrides any earlier configuration (only so this example can be rerun)
)

log = logging.getLogger(__name__)                # one logger per module, layered by module name

def transfer(amount):
    log.debug("this is hidden at INFO level")
    log.info("transferring %s", amount)          # a %s placeholder rather than an f-string
    try:
        1 / 0
    except ZeroDivisionError:
        log.exception("transfer failed")         # at ERROR level, with the traceback attached

transfer(100)
```

The key points:

- every module starts with `log = logging.getLogger(__name__)`, which puts the module's name in the log and allows levels to be set per package.
- pass arguments as `log.info("user %s", user_id)` rather than an f-string: the string is formatted only when it really has to be emitted.
- **a library only obtains a logger and logs, and configures no handlers**. The configuration (the level, the format, the destination) belongs to the application.
- a production configuration is usually loaded from a file through `logging.config.dictConfig`; for structured logs (JSON), `structlog` is worth a look.

| Level | Use |
| --- | --- |
| `DEBUG` | debugging detail, usually off |
| `INFO` | the key normal flow: a service started, a task finished |
| `WARNING` | unexpected but still able to continue: a retry, a fallback, a missing setting using the default |
| `ERROR` | an operation failed |
| `CRITICAL` | the whole program cannot continue |

## `argparse`: command-line arguments {#argparse命令行参数}

```python
import argparse

def build_parser():
    parser = argparse.ArgumentParser(prog="logtool", description="Analyze log files.")
    parser.add_argument("paths", nargs="+", help="log files to read")
    parser.add_argument("-l", "--level", choices=["INFO", "WARN", "ERROR"], default="ERROR")
    parser.add_argument("-n", "--top", type=int, default=10, help="show top N (default: %(default)s)")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser

args = build_parser().parse_args(["a.log", "b.log", "--level", "WARN", "-n", "3"])
assert args.paths == ["a.log", "b.log"]
assert (args.level, args.top, args.verbose) == ("WARN", 3, False)
```

`parse_args()` reads `sys.argv` when given nothing; passing a list makes it testable. The `--help` it generates is already quite good. Subcommands (like `git commit` and `git push`) use `parser.add_subparsers()`.

The third-party **Typer** (based on type annotations) and **Click** (based on decorators) are more concise to write and suit complicated command-line tools.

## `subprocess`: calling an external command {#subprocess调用外部命令}

```python
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-c", "import sys; print('hello'); sys.exit(0)"],   # the arguments as a list
    capture_output=True,       # capture stdout and stderr
    text=True,                 # returned as a string rather than bytes
    check=True,                # a non-zero return code raises CalledProcessError
    timeout=10,
)
assert result.stdout == "hello\n" and result.returncode == 0

try:
    subprocess.run([sys.executable, "-c", "raise SystemExit(3)"], check=True)
except subprocess.CalledProcessError as e:
    assert e.returncode == 3
```

- **Pass a list of arguments and do not build a string with `shell=True`.** When user input goes into the string, any command can be injected (`filename = "a.txt; rm -rf ~"`). Where shell features really are needed, escape every argument with `shlex.quote()`.
- Always add `check=True`, or a failed command goes unnoticed.
- To read the output live (for a long-running command), use `subprocess.Popen`.

## The other common modules at a glance {#其他常用模块速览}

| Module | Use | A one-line example |
| --- | --- | --- |
| `shutil` | copying, moving and deleting directory trees, archives | `shutil.rmtree(p)`, `shutil.make_archive("out", "zip", src)` |
| `tempfile` | temporary files and directories, cleaned up automatically | `with tempfile.TemporaryDirectory() as d:` |
| `os` / `os.environ` | environment variables, process information | `os.environ.get("DB_URL", "sqlite://")` |
| `tomllib` | reading TOML configuration (read-only) | `tomllib.loads(text)["tool"]` |
| `csv` | reading and writing CSV, handling quotes and commas correctly | `csv.DictReader(f)` |
| `sqlite3` | the built-in embedded database | `sqlite3.connect("app.db")` |
| `secrets` | generating secure random tokens and passwords | `secrets.token_urlsafe(32)` |
| `hashlib` | hashing | `hashlib.sha256(data).hexdigest()` |
| `uuid` | unique ids; `uuid7()` is time-ordered, which suits a database primary key <span class="since">3.14+</span> | `uuid.uuid4()`, `uuid.uuid7()` |
| `decimal` | exact decimal arithmetic (money) | `Decimal("0.1") + Decimal("0.2")` |
| `textwrap` | indenting, wrapping, removing a common indent | `textwrap.dedent(s)` |
| `pprint` | printing nested data nicely | `pprint.pp(data)` |
| `urllib.parse` | parsing and building URLs | `urlparse(url).netloc`, `urlencode(params)` |
| `http.server` | a throwaway static file server | `python -m http.server 8000` |
| `compression.zstd` | Zstandard compression <span class="since">3.14+</span> | `zstd.compress(data)` |

The `random` module's numbers are predictable and **must not** be used for passwords, tokens or verification codes, which all require `secrets`.

Small examples of `csv` and `tomllib`:

```python
import csv
import io
import tomllib

buf = io.StringIO('name,city\n"Doe, John",Paris\namy,"New York"\n')
rows = list(csv.DictReader(buf))
assert rows[0] == {"name": "Doe, John", "city": "Paris"}      # the comma inside the quotes is handled correctly

config = tomllib.loads("""
[server]
host = "0.0.0.0"
port = 8080

[database]
url = "postgres://localhost/app"
""")
assert config["server"]["port"] == 8080
```

## Formatting strings {#字符串格式化}

An f-string's format specifiers are powerful and a few are worth remembering:

```pycon
>>> price, ratio, n, name = 1234567.891, 0.4567, 42, "amy"
>>> f"{price:,.2f} | {ratio:.1%} | {n:05d} | {n:#x} | {n:b}"
'1,234,567.89 | 45.7% | 00042 | 0x2a | 101010'
>>> f"[{name:>8}] [{name:<8}] [{name:^8}] [{name:*^8}]"
'[     amy] [amy     ] [  amy   ] [**amy***]'
>>> f"{n = }, {name=}, {price=:.1f}"                      # = is for debugging output
"n = 42, name='amy', price=1234567.9"
>>> from datetime import date
>>> f"{date(2026, 9, 24):%Y/%m/%d}"
'2026/09/24'
```

3.14 adds the **t-string** (template string): written like an f-string with a `t` prefix, but it does **not** give a string directly and gives a `Template` object instead, holding the literal text and the interpolations separately. A library can then handle the interpolations safely, escaping SQL parameters or HTML say:

```pycon
>>> name = "<b>amy</b>"
>>> tpl = t"hello {name}!"
>>> tpl.strings, [i.value for i in tpl.interpolations]
(('hello ', '!'), ['<b>amy</b>'])
>>> import html
>>> "".join(part if isinstance(part, str) else html.escape(part.value) for part in tpl)
'hello &lt;b&gt;amy&lt;/b&gt;!'
```

Everyday code keeps using f-strings; t-strings are mainly for library authors, and you will meet them in newer template engines and database drivers.

!!! interview "How to explain it"
    The standard library is about getting it right: paths go to `pathlib` and text always carries `encoding="utf-8"`; times carry a time zone, storage is UTC in ISO 8601, and `datetime.now()` without a zone is the hazard; `json.dumps(..., ensure_ascii=False)` emits non-ASCII text; regular expressions use raw strings and named groups, and validating a whole string uses `fullmatch`; logging goes to `logging` rather than `print` and library code does not call `basicConfig`; `subprocess.run` takes a list of arguments with `check=True` and avoids `shell=True`'s injection risk; cryptographic randomness goes to `secrets` and money to `Decimal`.

## Exercises {#练习}

**1. A log statistics command-line tool.** Write a `main(argv)`: it takes several log file paths and a `--top N`; each log line is `2026-09-24T10:00:01+08:00 ERROR [module] message`; it counts the ERRORs per module and emits the top N as JSON (with `ensure_ascii=False`); and finally, converting every time to UTC, it reports the earliest and latest ERROR times. Test it against two files in a temporary directory.

??? success "Answer"
    ```python
    import argparse
    import json
    import re
    import tempfile
    from collections import Counter
    from datetime import UTC, datetime
    from pathlib import Path

    LINE = re.compile(r"(?P<ts>\S+) (?P<level>[A-Z]+) \[(?P<module>[^\]]+)\] (?P<msg>.*)")

    def main(argv=None):
        parser = argparse.ArgumentParser(prog="errstat")
        parser.add_argument("paths", nargs="+", type=Path)
        parser.add_argument("--top", type=int, default=3)
        args = parser.parse_args(argv)

        counts, times = Counter(), []
        for path in args.paths:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    m = LINE.match(line)
                    if m and m["level"] == "ERROR":
                        counts[m["module"]] += 1
                        times.append(datetime.fromisoformat(m["ts"]).astimezone(UTC))

        report = {
            "top": counts.most_common(args.top),
            "first": min(times).isoformat() if times else None,
            "last": max(times).isoformat() if times else None,
        }
        return json.dumps(report, ensure_ascii=False)

    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp, "a.log"), Path(tmp, "b.log")
        a.write_text(
            "2026-09-24T10:00:01+08:00 ERROR [支付] timeout\n"
            "2026-09-24T10:00:02+08:00 INFO [auth] ok\n"
            "2026-09-24T10:05:00+08:00 ERROR [auth] bad token\n",
            encoding="utf-8",
        )
        b.write_text("2026-09-24T03:00:00+00:00 ERROR [支付] declined\n", encoding="utf-8")
        out = json.loads(main([str(a), str(b), "--top", "1"]))

    assert out["top"] == [["支付", 2]]
    assert out["first"] == "2026-09-24T02:00:01+00:00"
    assert out["last"] == "2026-09-24T03:00:00+00:00"
    ```

    Putting the logic in `main(argv)` and returning the result, rather than printing directly and reading `sys.argv` directly, is what makes a command-line tool **testable**. The real entry point is only `if __name__ == "__main__": print(main())`.

## Summary {#小结}

- [x] Paths go to `pathlib`, and a text file always carries `encoding="utf-8"`.
- [x] Times always carry a time zone, and storage is UTC in ISO 8601.
- [x] Regular expressions use raw strings and named groups, and validating a format uses `fullmatch`.
- [x] Use `logging` rather than `print`; a library configures no handlers.
- [x] `subprocess.run` takes a list with `check=True` and never `shell=True`.
- [x] Security-related randomness goes to `secrets` and money to `Decimal`.
