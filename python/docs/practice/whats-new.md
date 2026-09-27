# 新版本特性速览

<p class="lead">Python 每年 10 月发布一个新版本。很多"老手"的知识停留在几年前，写出来的代码虽然能跑，但错过了更简洁、更安全的新写法。这一页按版本列出 3.10 到 3.14 里最值得掌握的变化，最后附上 3.15 的预告。</p>

!!! note "版本支持周期"
    每个版本提供 2 年的完整支持和额外 3 年的安全修复，总共 5 年。截至 2026 年 9 月，3.10 将在 2026 年 10 月停止维护，**新项目建议直接使用 3.13 或 3.14**。

## Python 3.10（2021）

**结构化模式匹配**：`match`/`case`，见[结构化模式匹配](../core/pattern-matching.md)。

**联合类型的新写法**：

```py
def f(x: int | None) -> str | bytes: ...     # 代替 Optional[int]、Union[str, bytes]
isinstance(x, int | str)                     # isinstance 也支持
```

**带括号的上下文管理器**，可以换行：

```py
with (
    open("a") as fa,
    open("b") as fb,
):
    ...
```

**`zip(strict=True)`**、**`itertools.pairwise`**，以及明显更好的语法错误提示。

## Python 3.11（2022）

**性能大幅提升**：平均比 3.10 快约 25%。

**异常组与 `except*`**、**`asyncio.TaskGroup`**、**`asyncio.timeout`**：见[异常与上下文管理器](../core/errors-context.md#同时处理多个异常exceptiongroup)和 [asyncio](../concurrency/asyncio.md)。

**`exception.add_note()`**：给异常附加上下文信息。

**回溯信息精确到表达式**：

```text
Traceback (most recent call last):
  File "demo.py", line 2, in <module>
    x = data["user"]["address"]["city"]
        ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^
TypeError: 'NoneType' object is not subscriptable
```

**标准库新增 `tomllib`**（读取 TOML）；**`typing.Self`**、**`enum.StrEnum`**；`datetime.UTC` 别名。

## Python 3.12（2023）

**新的类型参数语法**（PEP 695）：

```py
def first[T](xs: list[T]) -> T: ...
class Box[T]:
    def __init__(self, item: T) -> None: ...
type Pair[T] = tuple[T, T]
```

见[类型标注](../types/typing.md#泛型)。

**f-string 限制放宽**（PEP 701）：f-string 里可以使用和外层相同的引号、可以写反斜杠和注释、可以任意嵌套：

```py
f"{", ".join(names)}"            # 3.12 之前外层和内层引号不能相同
```

**其他**：`itertools.batched`、`Path.walk()`、`typing.override`、低开销的监控 API `sys.monitoring`（PEP 669）、更友好的错误提示（比如 `NameError` 会提示"你是不是忘了 import"）、每个子解释器可以有自己的 GIL（PEP 684）；移除了 `distutils`。

## Python 3.13（2024）

**全新的交互式解释器（REPL）**：支持多行编辑、彩色输出、历史浏览，`help`、`exit` 不用加括号，++f3++ 进入粘贴模式。

**自由线程版本（实验性）**：可以去掉 GIL 运行的构建版本（`python3.13t`），见[线程、进程与 GIL](../concurrency/threads-processes.md#gil全局解释器锁)。

**实验性的 JIT 编译器**（需要编译时开启）。

**`locals()` 的语义被明确定义**（PEP 667），调试器和各种工具的行为更可预测。

**类型系统**：`typing.TypeIs`、`TypedDict` 的 `ReadOnly`、类型参数默认值（PEP 696）、`warnings.deprecated` 装饰器。

**其他**：`copy.replace()`、`itertools.batched(strict=True)`、错误信息会提示"你的脚本和标准库模块重名了"；移除了 PEP 594 列出的一批陈旧模块（`cgi`、`telnetlib`、`crypt`、`imghdr` 等）。

## Python 3.14（2025）

**标注延迟求值**（PEP 649/749）：类型标注在真正读取时才计算，前向引用不再需要加引号，也不再需要 `from __future__ import annotations`。新增 `annotationlib` 模块用于读取标注。

**模板字符串 t-string**（PEP 750）：`t"hello {name}"` 得到 `Template` 对象而不是字符串，让库可以安全地处理插值（SQL、HTML 转义等），见[常用标准库](../engineering/stdlib.md#字符串格式化)。

**自由线程版本正式支持**（PEP 779）：从"实验性"变为"官方支持"，但仍不是默认构建。

**标准库支持多解释器**（PEP 734）：新增 `concurrent.interpreters` 模块和 `concurrent.futures.InterpreterPoolExecutor`。

**`except` 可以省略括号**（PEP 758）：

```py
except TimeoutError, ConnectionError:      # 不带 as 时
    ...
```

**`finally` 里的 `return`/`break`/`continue` 会产生警告**（PEP 765）。

**新模块和新功能**：

- `compression.zstd`：Zstandard 压缩算法
- `uuid.uuid7()`：按时间排序的 UUID，适合做数据库主键
- `pathlib.Path.copy()`、`move()`
- `python -m asyncio ps PID`、`pstree PID`：查看运行中进程的 asyncio 任务
- `pdb -p PID`：附加调试一个正在运行的进程（PEP 768）
- REPL 支持语法高亮；`argparse`、`unittest`、`json` 等命令行输出支持颜色

**其他**：Linux 上 `multiprocessing` 的默认启动方式从 `fork` 改为 `forkserver`；官方的 macOS 和 Windows 安装包开始包含实验性的 JIT。

## Python 3.15（预计 2026 年 10 月）

写作本页时 3.15 处于候选发布阶段（3.15.0rc2），以下特性已经可以试用（`uv python install 3.15`）：

- **显式延迟导入**（PEP 810）：`lazy import json` 在第一次使用时才真正导入模块，可以显著加快大型程序和命令行工具的启动速度。
- **默认使用 UTF-8 模式**（PEP 686）：`open()` 等函数不指定编码时默认使用 UTF-8，终于告别 Windows 上的乱码问题。
- **内置统计采样分析器** `profiling.sampling`：低开销，可以对正在运行的程序进行性能分析。

## 怎么跟上变化

- 每年新版本发布时，读一遍官方的 [What's New](https://docs.python.org/zh-cn/3/whatsnew/index.html)，挑自己用得上的部分。
- 用 `ruff` 的 `UP` 规则（pyupgrade）自动把旧写法升级为新写法：把 `target-version` 调高后运行 `ruff check --fix`。
- 新版本发布后一两个月，等主要依赖都支持了，就可以升级项目的 Python 版本，顺便获得免费的性能提升。
