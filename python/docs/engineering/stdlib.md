# 常用标准库

<p class="lead">"自带电池"是 Python 的一大优势，但很多人只用到了其中很小一部分，然后去装第三方包、甚至自己造轮子。这一章挑出日常开发最常用的模块，讲它们的正确用法和常见误区。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 拼接路径、列出目录下所有 `.py` 文件，用哪个模块最方便？
    2. `datetime.now()` 返回的时间有什么隐患？
    3. `json.dumps` 输出中文时变成了 `\u4e2d` 这样的转义，怎么解决？
    4. 调用外部命令时，为什么不推荐 `shell=True`？
    5. 库代码里应该调用 `logging.basicConfig()` 吗？

??? success "自测参考答案（先自己答，再展开对照）"
    1. `pathlib`：`Path("a") / "b"` 拼接路径，`Path("src").rglob("*.py")` 递归地找出所有 `.py` 文件。
    2. 返回的是不带时区的本地时间，换机器、换时区、遇到夏令时就会出错，和带时区的时间也没法比较。用 `datetime.now(timezone.utc)`，存储用 UTC 和 ISO 8601 格式。
    3. 传 `ensure_ascii=False`，写文件时同时指定 `encoding="utf-8"`。
    4. 命令要经过 shell 解析，拼进去的外部输入可能被注入任意命令；引号、空格等特殊字符也容易出错。应该传参数列表：`subprocess.run(["ls", path], check=True)`。
    5. 不应该。日志怎么输出由应用程序决定；库只用 `logging.getLogger(__name__)` 记录日志（顶多加一个 `NullHandler`），不配置 handler 和级别。

## `pathlib`：面向对象的路径

`pathlib` 已经完全可以取代 `os.path` 的大部分用法，代码更短也更不容易出错：

```python
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    src = root / "project" / "src"                 # 用 / 拼接路径
    src.mkdir(parents=True, exist_ok=True)         # 类似 mkdir -p

    (src / "app.py").write_text("print('hi')\n", encoding="utf-8")
    (src / "util.py").write_text("", encoding="utf-8")
    (root / "project" / "README.md").write_text("# demo\n", encoding="utf-8")

    f = src / "app.py"
    assert (f.name, f.stem, f.suffix) == ("app.py", "app", ".py")
    assert f.parent.name == "src"
    assert f.with_suffix(".txt").name == "app.txt"
    assert f.read_text(encoding="utf-8") == "print('hi')\n"

    py_files = sorted(p.name for p in (root / "project").rglob("*.py"))   # 递归查找
    assert py_files == ["app.py", "util.py"]
    assert f.relative_to(root).as_posix() == "project/src/app.py"
    assert f.exists() and f.is_file() and not f.is_dir()
```

| 需求 | `pathlib` 写法 |
| --- | --- |
| 当前工作目录 / 用户主目录 | `Path.cwd()` / `Path.home()` |
| 脚本所在目录 | `Path(__file__).resolve().parent` |
| 遍历目录 | `p.iterdir()`、`p.glob("*.csv")`、`p.rglob("*.py")`、`p.walk()` <span class="since">3.12+</span> |
| 读写小文件 | `read_text()`、`write_text()`、`read_bytes()`、`write_bytes()` |
| 打开文件 | `p.open("r", encoding="utf-8")` |
| 删除文件 / 空目录 | `p.unlink(missing_ok=True)` / `p.rmdir()` |
| 复制、移动 | `p.copy(dst)`、`p.move(dst)` <span class="since">3.14+</span>，之前用 `shutil.copy2`、`shutil.move` |
| 文件信息 | `p.stat().st_size`、`p.stat().st_mtime` |

!!! warning "永远显式指定 `encoding`"
    `open()` 和 `read_text()` 不指定编码时，在 3.14 及以前使用系统的"本地编码"，在 Windows 上通常不是 UTF-8，同一份代码换台机器就乱码。3.15 起默认改为 UTF-8，但为了兼容旧版本，**文本文件一律写上 `encoding="utf-8"`**。

## `json`

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

- `dumps`/`loads` 处理字符串，`dump`/`load` 处理文件对象。
- 中文要 `ensure_ascii=False`，写文件时配合 `encoding="utf-8"`。
- 元组会变成列表，字典的键会变成字符串，`datetime`、`set`、`Decimal`、自定义对象默认不能序列化。用 `default` 参数处理它们：

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

命令行里格式化 JSON：`python -m json.tool data.json`，或者 `cat data.json | python -m json`（3.14+）。

## `datetime` 与 `zoneinfo`：处理时间

时间处理最重要的一条规则：**不要使用"天真"（naive）的时间**，也就是不带时区信息的 `datetime`。`datetime.now()` 返回的就是本地时间的 naive 对象，服务器换个时区设置，数据就全乱了。

```pycon
>>> from datetime import datetime, timedelta, UTC
>>> from zoneinfo import ZoneInfo
>>> t = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)          # 带时区的 aware 时间
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

推荐的做法：

- 获取当前时间：`datetime.now(UTC)`。
- **存储和传输**一律用 UTC，格式用 ISO 8601（`isoformat()` / `fromisoformat()`）。
- 只在**展示给用户**时转换成用户的时区。
- 时区用 `zoneinfo.ZoneInfo("Asia/Shanghai")` 这样的 IANA 名字，不要自己写死 `+8` 的偏移量（很多地区有夏令时）。
- 测量耗时用 `time.perf_counter()`，不要用 `datetime` 相减。

## `re`：正则表达式

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
>>> re.sub(r"\d+", lambda m: str(int(m[0]) * 2), "3 apples, 10 pears")   # 替换可以是函数
'6 apples, 20 pears'
>>> bool(re.fullmatch(r"[a-z_][a-z0-9_]*", "user_id")), bool(re.fullmatch(r"[a-z_][a-z0-9_]*", "1st"))
(True, False)
```

要点：

- 正则字符串一律用**原始字符串** `r"..."`，否则 `\d`、`\b` 这些反斜杠会被 Python 先处理一遍。
- `match` 只从开头匹配，`search` 在任意位置找，`fullmatch` 要求整个字符串匹配。校验输入格式时用 `fullmatch`。
- 用**命名分组** `(?P<name>...)`，比数第几个括号可读得多。
- 复杂的正则用 `re.VERBOSE` 标志，可以换行和写注释。
- 简单的字符串操作优先用字符串方法：`startswith`、`endswith`、`in`、`split`、`replace`，更快也更清楚。

## `logging`：日志

**不要用 `print` 调试和记录运行信息。** `logging` 可以分级别、统一格式、输出到文件或日志系统，并且可以在不改代码的情况下调整详细程度。

```python
import logging
import sys

# 应用程序的入口处配置一次（库代码不要调用 basicConfig）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,                                  # 覆盖之前的配置（这里只是为了示例能重复运行）
)

log = logging.getLogger(__name__)                # 每个模块一个 logger，按模块名分层

def transfer(amount):
    log.debug("this is hidden at INFO level")
    log.info("transferring %s", amount)          # 用 %s 占位，而不是 f-string
    try:
        1 / 0
    except ZeroDivisionError:
        log.exception("transfer failed")         # ERROR 级别，并附带回溯信息

transfer(100)
```

要点：

- 每个模块开头 `log = logging.getLogger(__name__)`，日志会带上模块名，可以按包的层级分别设置级别。
- 传参用 `log.info("user %s", user_id)` 而不是 f-string：只有真正需要输出时才会格式化字符串。
- **库只获取 logger、打日志，不配置 handler**。配置（级别、格式、输出位置）是应用程序的事。
- 生产环境的配置通常用 `logging.config.dictConfig` 从配置文件加载；需要结构化日志（JSON）时可以看看 `structlog`。

| 级别 | 用途 |
| --- | --- |
| `DEBUG` | 调试细节，平时关闭 |
| `INFO` | 正常的关键流程：服务启动、任务完成 |
| `WARNING` | 意料之外但还能继续：重试、降级、配置缺失用了默认值 |
| `ERROR` | 某个操作失败了 |
| `CRITICAL` | 整个程序无法继续 |

## `argparse`：命令行参数

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

`parse_args()` 不传参数时读取 `sys.argv`；传入列表方便测试。自动生成的 `--help` 已经相当好用。有子命令（像 `git commit`、`git push`）时用 `parser.add_subparsers()`。

第三方的 **Typer**（基于类型标注）和 **Click**（基于装饰器）写起来更简洁，适合复杂的命令行工具。

## `subprocess`：调用外部命令

```python
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-c", "import sys; print('hello'); sys.exit(0)"],   # 参数用列表
    capture_output=True,       # 捕获 stdout 和 stderr
    text=True,                 # 以字符串而不是字节返回
    check=True,                # 返回码非 0 时抛 CalledProcessError
    timeout=10,
)
assert result.stdout == "hello\n" and result.returncode == 0

try:
    subprocess.run([sys.executable, "-c", "raise SystemExit(3)"], check=True)
except subprocess.CalledProcessError as e:
    assert e.returncode == 3
```

- **参数传列表，不要用 `shell=True` 拼字符串。** 拼接的字符串里如果包含用户输入，就可能被注入任意命令（`filename = "a.txt; rm -rf ~"`）。确实需要 shell 特性时，用 `shlex.quote()` 转义每一个参数。
- 永远加上 `check=True`，否则命令失败了你也不知道。
- 需要实时读取输出（长时间运行的命令）时，用 `subprocess.Popen`。

## 其他常用模块速览

| 模块 | 用途 | 一行示例 |
| --- | --- | --- |
| `shutil` | 复制、移动、删除目录树，压缩归档 | `shutil.rmtree(p)`、`shutil.make_archive("out", "zip", src)` |
| `tempfile` | 临时文件和目录，自动清理 | `with tempfile.TemporaryDirectory() as d:` |
| `os` / `os.environ` | 环境变量、进程信息 | `os.environ.get("DB_URL", "sqlite://")` |
| `tomllib` | 读取 TOML 配置（只读） | `tomllib.loads(text)["tool"]` |
| `csv` | 读写 CSV，正确处理引号和逗号 | `csv.DictReader(f)` |
| `sqlite3` | 内置的嵌入式数据库 | `sqlite3.connect("app.db")` |
| `secrets` | 生成安全的随机令牌、密码 | `secrets.token_urlsafe(32)` |
| `hashlib` | 哈希 | `hashlib.sha256(data).hexdigest()` |
| `uuid` | 唯一 ID；`uuid7()` 按时间有序，适合做数据库主键 <span class="since">3.14+</span> | `uuid.uuid4()`、`uuid.uuid7()` |
| `decimal` | 精确的十进制运算（金额） | `Decimal("0.1") + Decimal("0.2")` |
| `textwrap` | 缩进、换行、去除公共缩进 | `textwrap.dedent(s)` |
| `pprint` | 漂亮地打印嵌套数据 | `pprint.pp(data)` |
| `urllib.parse` | 解析和构建 URL | `urlparse(url).netloc`、`urlencode(params)` |
| `http.server` | 临时起一个静态文件服务器 | `python -m http.server 8000` |
| `compression.zstd` | Zstandard 压缩 <span class="since">3.14+</span> | `zstd.compress(data)` |

`random` 模块的随机数是可预测的，**不能**用于生成密码、令牌、验证码，这些场景必须用 `secrets`。

`csv` 和 `tomllib` 的小例子：

```python
import csv
import io
import tomllib

buf = io.StringIO('name,city\n"Doe, John",Paris\namy,"New York"\n')
rows = list(csv.DictReader(buf))
assert rows[0] == {"name": "Doe, John", "city": "Paris"}      # 引号里的逗号被正确处理

config = tomllib.loads("""
[server]
host = "0.0.0.0"
port = 8080

[database]
url = "postgres://localhost/app"
""")
assert config["server"]["port"] == 8080
```

## 字符串格式化

f-string 的格式说明符很强大，值得记住几个常用的：

```pycon
>>> price, ratio, n, name = 1234567.891, 0.4567, 42, "amy"
>>> f"{price:,.2f} | {ratio:.1%} | {n:05d} | {n:#x} | {n:b}"
'1,234,567.89 | 45.7% | 00042 | 0x2a | 101010'
>>> f"[{name:>8}] [{name:<8}] [{name:^8}] [{name:*^8}]"
'[     amy] [amy     ] [  amy   ] [**amy***]'
>>> f"{n = }, {name=}, {price=:.1f}"                      # = 用于调试输出
"n = 42, name='amy', price=1234567.9"
>>> from datetime import date
>>> f"{date(2026, 9, 24):%Y/%m/%d}"
'2026/09/24'
```

3.14 新增了 **t-string**（模板字符串）：写法和 f-string 一样，只是前缀换成 `t`，但它**不会**直接得到字符串，而是得到一个 `Template` 对象，里面分开保存了字面文本和插值。库可以据此安全地处理插值，比如给 SQL 参数做转义、给 HTML 做转义：

```pycon
>>> name = "<b>amy</b>"
>>> tpl = t"hello {name}!"
>>> tpl.strings, [i.value for i in tpl.interpolations]
(('hello ', '!'), ['<b>amy</b>'])
>>> import html
>>> "".join(part if isinstance(part, str) else html.escape(part.value) for part in tpl)
'hello &lt;b&gt;amy&lt;/b&gt;!'
```

日常代码继续用 f-string；t-string 主要是给库作者用的，你会在新版本的模板引擎、数据库驱动里遇到它。

!!! interview "怎么讲清楚"
    标准库讲的是"写对"：路径用 `pathlib`，读写文本总是 `encoding="utf-8"`；时间用带时区的 `datetime`，存储用 UTC 和 ISO 8601，`datetime.now()` 不带时区是隐患；`json.dumps(..., ensure_ascii=False)` 输出中文；正则用原始字符串、命名分组，校验整串用 `fullmatch`；日志用 `logging` 而不是 `print`，库代码不调用 `basicConfig`；`subprocess.run` 传参数列表、加 `check=True`，避免 `shell=True` 的注入风险；密码学随机数用 `secrets`，金额用 `Decimal`。

## 练习

**1. 日志统计命令行工具。** 写一个 `main(argv)` 函数：接收若干日志文件路径和 `--top N` 参数；日志每行格式是 `2026-09-24T10:00:01+08:00 ERROR [module] message`；统计每个模块的 ERROR 数量，以 JSON 输出前 N 名（`ensure_ascii=False`）；最后把所有时间转换成 UTC 后，输出最早和最晚的 ERROR 时间。用临时目录里的两个文件测试它。

??? success "参考答案"
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

    把逻辑写在 `main(argv)` 里并返回结果，而不是直接打印、直接读 `sys.argv`，是让命令行工具**可测试**的关键。真正的脚本入口只需要 `if __name__ == "__main__": print(main())`。

## 小结

- [x] 路径用 `pathlib`，文本文件总是写 `encoding="utf-8"`。
- [x] 时间一律用带时区的 `datetime`，存储用 UTC 和 ISO 8601。
- [x] 正则用原始字符串和命名分组，校验格式用 `fullmatch`。
- [x] 用 `logging` 而不是 `print`；库不配置 handler。
- [x] `subprocess.run` 传列表、加 `check=True`，不要 `shell=True`。
- [x] 安全相关的随机数用 `secrets`，金额用 `Decimal`。
